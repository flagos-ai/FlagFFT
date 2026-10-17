# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Parameterized tensor FFTs with explicit logical input/output layouts.

All device work in this module is emitted as Triton/TLE source.  The same
butterflies handle contiguous rows, strided axes, compact real boundaries and
permuted stores.  Exchange is explicit so an implementation can be qualified
against the capabilities of its compiler backend.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source


@dataclass(frozen=True)
class TensorFFTConfig:
    length: int
    rows_per_program: int = 8
    direction: str = "forward"
    mode: str = "complex"
    input_stride: int = 1
    output_stride: int = 1
    output_rows_per_batch: int = 0
    exchange: str = "gather"
    num_warps: int = 4

    def __post_init__(self):
        if self.length < 2 or self.length & (self.length - 1):
            raise ValueError("tensor FFT length must be a power of two >= 2")
        pack = self.rows_per_program
        if pack < 1 or pack & (pack - 1):
            raise ValueError("rows_per_program must be a positive power of two")
        if self.direction not in {"forward", "inverse"}:
            raise ValueError("invalid tensor FFT direction")
        if self.mode not in {"complex", "r2c", "c2r"}:
            raise ValueError("invalid tensor FFT IO mode")
        if ((self.mode == "r2c" and self.direction != "forward") or
                (self.mode == "c2r" and self.direction != "inverse")):
            raise ValueError("real FFT mode disagrees with direction")
        if self.input_stride < 1 or self.output_stride < 1:
            raise ValueError("tensor FFT strides must be positive")
        if self.output_rows_per_batch < 0:
            raise ValueError("output_rows_per_batch must be nonnegative")
        if self.output_rows_per_batch and self.output_stride != 1:
            raise ValueError("permuted output uses output_stride=1")
        if self.exchange not in {"gather", "split", "dsa", "buffer", "static"}:
            raise ValueError("unsupported tensor FFT exchange")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("unsupported tensor FFT launch width")


def _build_static_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    """Route a complete radix graph at emission time, vectorizing over rows.

    Values remain separate Triton tensors across stages.  This reuses the
    register codelets without asking a backend to lower a tensor gather or
    reconstruct a succession of reshaped, sliced local-memory layouts.
    """
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(json.dumps({"version": 3, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    input_length = n // 2 + 1 if config.mode == "c2r" else n
    output_length = n // 2 + 1 if config.mode == "r2c" else n
    lines = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch):",
        f"    row = tl.program_id(0).to(tl.int64) * {pack} + tl.arange(0, {pack})",
        "    mask = row < nbatch",
        f"    input_base = (row // {config.input_stride}) * {input_length * config.input_stride} + row % {config.input_stride}",
    ]
    bits = n.bit_length() - 1
    for index in range(n):
        source_index = int(f"{index:0{bits}b}"[::-1], 2)
        conjugate = config.mode == "c2r" and source_index > n // 2
        if conjugate:
            source_index = n - source_index
        position = f"(input_base + {source_index * config.input_stride})"
        if config.mode == "r2c":
            lines += [
                f"    r{index} = tl.load(in_ptr + {position}, mask=mask, other=0.0)",
                f"    i{index} = tl.full(({pack},), 0.0, tl.float32)",
            ]
        else:
            lines += [f"    r{index} = tl.load(in_ptr + {position} * 2, mask=mask, other=0.0)"]
            if config.mode == "c2r" and source_index in {0, n // 2}:
                lines += [f"    i{index} = tl.full(({pack},), 0.0, tl.float32)"]
            else:
                sign = "-" if conjugate else ""
                lines += [f"    i{index} = {sign}tl.load(in_ptr + {position} * 2 + 1, mask=mask, other=0.0)"]
    for stage in range(1, bits + 1):
        span = 1 << stage
        half = span // 2
        for base in range(0, n, span):
            for lane in range(half):
                lo, hi = base + lane, base + lane + half
                angle = (-1 if config.direction == "forward" else 1) * 2 * math.pi * lane / span
                c, s = math.cos(angle), math.sin(angle)
                c = 0.0 if abs(c) < 1e-15 else c
                s = 0.0 if abs(s) < 1e-15 else s
                if lane == 0:
                    real, imag = f"r{hi}", f"i{hi}"
                elif c == 0.0:
                    real = f"i{hi}" if s < 0 else f"-i{hi}"
                    imag = f"r{hi}" if s > 0 else f"-r{hi}"
                else:
                    lines += [
                        f"    tr = r{hi} * {c!r} - i{hi} * {s!r}",
                        f"    ti = r{hi} * {s!r} + i{hi} * {c!r}",
                    ]
                    real, imag = "tr", "ti"
                lines += [f"    r{lo}, r{hi}, i{lo}, i{hi} = _fwd_rad2_b1(r{lo}, {real}, i{lo}, {imag})"]
    if config.output_rows_per_batch:
        extent = config.output_rows_per_batch
        lines += [f"    output_base = (row // {extent}) * {extent * output_length} + row % {extent}"]
        step = extent
    else:
        lines += [f"    output_base = (row // {config.output_stride}) * {output_length * config.output_stride} + row % {config.output_stride}"]
        step = config.output_stride
    for index in range(output_length):
        position = f"(output_base + {index * step})"
        if config.mode == "c2r":
            lines += [f"    tl.store(out_ptr + {position}, r{index}, mask=mask)"]
        else:
            lines += [
                f"    tl.store(out_ptr + {position} * 2, r{index}, mask=mask)",
                f"    tl.store(out_ptr + {position} * 2 + 1, i{index}, mask=mask)",
            ]
    return name, "\n".join(lines) + "\n"


def _build_tensor_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    if config.exchange == "static":
        return _build_static_fft_source(config)
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(
        json.dumps({"version": 2, **asdict(config)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    bits = n.bit_length() - 1
    reversed_index = " | ".join(
        f"(((frequency >> {bit}) & 1) << {bits - bit - 1})"
        for bit in range(bits)
    )
    input_length = n // 2 + 1 if config.mode == "c2r" else n
    output_length = n // 2 + 1 if config.mode == "r2c" else n
    # The innermost tensor axis groups independent rows. Eight FP32 rows
    # form one aligned Ascend vector block even at the two-point butterfly.
    # Keeping this axis through every stage avoids padding a succession of
    # one/two-element inner axes to the hardware's memory alignment.
    lines = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch):",
        f"    row = tl.program_id(0).to(tl.int64) * {pack} + tl.arange(0, {pack})",
        f"    frequency = tl.arange(0, {n})",
        f"    source_frequency = {reversed_index}",
        f"    input_base = (row // {config.input_stride}) * {input_length * config.input_stride} + row % {config.input_stride}",
    ]
    if config.mode == "c2r":
        lines += [
            f"    conjugate = source_frequency > {n // 2}",
            f"    source_frequency = tl.where(conjugate, {n} - source_frequency, source_frequency)",
        ]
    lines += [
        f"    source_index = input_base[None, :] + source_frequency[:, None] * {config.input_stride}",
        "    mask = row[None, :] < nbatch",
    ]
    if config.mode == "r2c":
        lines += [
            "    real = tl.load(in_ptr + source_index, mask=mask, other=0.0)",
            f"    imag = tl.full(({n}, {pack}), 0.0, tl.float32)",
        ]
    else:
        lines += [
            "    real = tl.load(in_ptr + 2 * source_index, mask=mask, other=0.0)",
            "    imag = tl.load(in_ptr + 2 * source_index + 1, mask=mask, other=0.0)",
        ]
        if config.mode == "c2r":
            lines += [
                "    imag = tl.where(conjugate[:, None], -imag, imag)",
                f"    imag = tl.where((source_frequency[:, None] == 0) | (source_frequency[:, None] == {n // 2}), 0.0, imag)",
            ]

    for stage in range(1, bits + 1):
        span = 1 << stage
        half = span // 2
        if config.exchange == "gather":
            lines += [
                f"    lo_index = tl.broadcast_to((frequency & ~{half})[:, None], ({n}, {pack}))",
                f"    hi_index = lo_index + {half}",
                "    ar = tl.gather(real, lo_index, 0)",
                "    ai = tl.gather(imag, lo_index, 0)",
                "    br = tl.gather(real, hi_index, 0)",
                "    bi = tl.gather(imag, hi_index, 0)",
            ]
            if half > 1:
                lines += [
                    f"    twiddle_index = (frequency % {half}) * {n // span} * 2",
                    "    wr = tl.load(twiddle_ptr + twiddle_index)[:, None]",
                    "    wi = tl.load(twiddle_ptr + twiddle_index + 1)[:, None]",
                    "    tr = br * wr - bi * wi",
                    "    ti = br * wi + bi * wr",
                ]
            else:
                lines += ["    tr = br", "    ti = bi"]
            lines += [
                f"    real = ar + tl.where((frequency & {half})[:, None] == 0, tr, -tr)",
                f"    imag = ai + tl.where((frequency & {half})[:, None] == 0, ti, -ti)",
            ]
            continue

        groups = n // span
        shape = f"[{groups}, {span}, {pack}]"
        part = f"[{groups}, {half}, {pack}]"
        zero = "tl.full((), 0, tl.int32)"
        high = f"tl.full((), {half}, tl.int32)"
        for component, lower, upper in (("real", "ar", "br"), ("imag", "ai", "bi")):
            if config.exchange == "split":
                lines += [
                    f"    stage_{component} = tl.reshape({component}, ({groups}, 2, {half}, {pack}))",
                    f"    {lower}, {upper} = tl.split(tl.trans(stage_{component}, (0, 2, 3, 1)))",
                ]
            elif config.exchange == "buffer":
                lines += [
                    f"    stage_{component} = tle.dsa.to_buffer(tl.reshape({component}, {shape}), space=tle.dsa.ascend.UB)",
                    f"    {lower} = tle.dsa.to_tensor(tle.dsa.subview(stage_{component}, [0, 0, 0], {part}, [1, 1, 1]), writable=False)",
                    f"    {upper} = tle.dsa.to_tensor(tle.dsa.subview(stage_{component}, [0, {half}, 0], {part}, [1, 1, 1]), writable=False)",
                ]
            else:
                lines += [
                    f"    stage_{component} = tl.reshape({component}, {shape})",
                    f"    {lower} = tle.dsa.extract_slice(stage_{component}, [{zero}, {zero}, {zero}], {part}, [1, 1, 1])",
                    f"    {upper} = tle.dsa.extract_slice(stage_{component}, [{zero}, {high}, {zero}], {part}, [1, 1, 1])",
                ]
        if half > 1:
            lines += [
                f"    twiddle_index = tl.arange(0, {half}) * {n // span} * 2",
                "    wr = tl.load(twiddle_ptr + twiddle_index)[None, :, None]",
                "    wi = tl.load(twiddle_ptr + twiddle_index + 1)[None, :, None]",
                "    tr = br * wr - bi * wi",
                "    ti = br * wi + bi * wr",
            ]
        else:
            lines += ["    tr = br", "    ti = bi"]
        for component, base, term in (("real", "ar", "tr"), ("imag", "ai", "ti")):
            if config.exchange == "split":
                lines += [
                    f"    joined_{component} = tl.trans(tl.join({base} + {term}, {base} - {term}), (0, 3, 1, 2))",
                    f"    {component} = tl.reshape(joined_{component}, ({n}, {pack}))",
                ]
            elif config.exchange == "buffer":
                lines += [
                    f"    joined_{component} = tle.dsa.alloc({shape}, tl.float32, tle.dsa.ascend.UB)",
                    f"    lower_{component} = tle.dsa.to_buffer({base} + {term}, bind_buffer=tle.dsa.subview(joined_{component}, [0, 0, 0], {part}, [1, 1, 1]))",
                    f"    upper_{component} = tle.dsa.to_buffer({base} - {term}, bind_buffer=tle.dsa.subview(joined_{component}, [0, {half}, 0], {part}, [1, 1, 1]))",
                    f"    {component} = tl.reshape(tle.dsa.to_tensor(joined_{component}, writable=False), ({n}, {pack}))",
                ]
            else:
                lines += [
                    f"    joined_{component} = tl.full({shape}, 0.0, tl.float32)",
                    f"    joined_{component} = tle.dsa.insert_slice(joined_{component}, {base} + {term}, [{zero}, {zero}, {zero}], {part}, [1, 1, 1])",
                    f"    joined_{component} = tle.dsa.insert_slice(joined_{component}, {base} - {term}, [{zero}, {high}, {zero}], {part}, [1, 1, 1])",
                    f"    {component} = tl.reshape(joined_{component}, ({n}, {pack}))",
                ]

    if config.output_rows_per_batch:
        extent = config.output_rows_per_batch
        lines += [
            f"    output_base = (row // {extent}) * {extent * output_length} + row % {extent}",
            f"    output_index = output_base[None, :] + frequency[:, None] * {extent}",
        ]
    else:
        lines += [
            f"    output_base = (row // {config.output_stride}) * {output_length * config.output_stride} + row % {config.output_stride}",
            f"    output_index = output_base[None, :] + frequency[:, None] * {config.output_stride}",
        ]
    lines += [f"    output_mask = mask & (frequency[:, None] < {output_length})"]
    if config.mode == "c2r":
        lines += ["    tl.store(out_ptr + output_index, real, mask=output_mask)"]
    else:
        lines += [
            "    tl.store(out_ptr + 2 * output_index, real, mask=output_mask)",
            "    tl.store(out_ptr + 2 * output_index + 1, imag, mask=output_mask)",
        ]
    return name, "\n".join(lines) + "\n"


def emit_tensor_fft(config: TensorFFTConfig, out_dir: Path) -> dict:
    name, source = _build_tensor_fft_source(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source, radices=(2,) if config.exchange == "static" else ()))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,i32",
        "num_warps": config.num_warps,
        "num_stages": 1,
        "batch_per_block": config.rows_per_program,
        "arg_names": ["in_ptr", "out_ptr", "twiddle_ptr", "nbatch"],
        "kernel_type": "tensor_fft",
        "dtype": "complex64",
        "implementation": "triton",
        "tensor_fft_config": asdict(config),
        "generator_version": 3,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
