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
from .kernels_common import _NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import _emit_natural_order_codelet_call


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
    factors: tuple[int, ...] = ()
    twiddle: str = "table"

    def __post_init__(self):
        if self.length < 2:
            raise ValueError("tensor FFT length must be >= 2")
        if self.exchange != "static" and self.length & (self.length - 1):
            raise ValueError("tensor exchange requires a power-of-two length")
        factors = tuple(self.factors) or _static_factors(self.length)
        if math.prod(factors) != self.length or any(
            radix not in _NATURAL_ORDER_CODELET_RADICES for radix in factors
        ):
            raise ValueError("unsupported tensor FFT factorization")
        object.__setattr__(self, "factors", factors)
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
        if self.exchange not in {"gather", "flat", "interleaved", "split", "dsa", "buffer", "static"}:
            raise ValueError("unsupported tensor FFT exchange")
        if self.twiddle not in {"table", "local_table", "compute"}:
            raise ValueError("unsupported twiddle strategy")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("unsupported tensor FFT launch width")


def _static_factors(length: int) -> tuple[int, ...]:
    """Use small natural-order codelets to bound the generated arithmetic DAG."""
    remaining = length
    factors = []
    for radix in (4, 2, 3, 5, 7, 11, 13, 17, 19):
        while remaining % radix == 0:
            factors.append(radix)
            remaining //= radix
    if remaining != 1:
        raise ValueError(f"no register-codelet factorization for length {length}")
    return tuple(factors)


def _digit_reversed_index(index: int, factors: tuple[int, ...]) -> int:
    source, stride = 0, math.prod(factors)
    for radix in factors:
        stride //= radix
        source += (index % radix) * stride
        index //= radix
    return source


def _build_static_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    """Route a complete radix graph at emission time, vectorizing over rows.

    Values remain separate Triton tensors across stages.  This reuses the
    register codelets without asking a backend to lower a tensor gather or
    reconstruct a succession of reshaped, sliced local-memory layouts.
    """
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(json.dumps({"version": 6, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    input_length = n // 2 + 1 if config.mode == "c2r" else n
    output_length = n // 2 + 1 if config.mode == "r2c" else n
    lines = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, row_offset):",
        f"    row = row_offset + tl.program_id(0).to(tl.int64) * {pack} + tl.arange(0, {pack})",
        "    mask = row < nbatch",
        f"    input_base = (row // {config.input_stride}) * {input_length * config.input_stride} + row % {config.input_stride}",
    ]
    for index in range(n):
        source_index = _digit_reversed_index(index, config.factors)
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
            if config.mode == "c2r" and (source_index == 0 or (n % 2 == 0 and source_index == n // 2)):
                lines += [f"    i{index} = tl.full(({pack},), 0.0, tl.float32)"]
            else:
                sign = "-" if conjugate else ""
                lines += [f"    i{index} = {sign}tl.load(in_ptr + {position} * 2 + 1, mask=mask, other=0.0)"]
    previous_span = 1
    for radix in config.factors:
        span = previous_span * radix
        for base in range(0, n, span):
            for lane in range(previous_span):
                indices = [base + lane + digit * previous_span for digit in range(radix)]
                for digit, register in enumerate(indices[1:], start=1):
                    angle = (-1 if config.direction == "forward" else 1) * 2 * math.pi * digit * lane / span
                    if lane == 0:
                        continue
                    c, s = math.cos(angle), math.sin(angle)
                    c = 0.0 if abs(c) < 1e-15 else c
                    s = 0.0 if abs(s) < 1e-15 else s
                    lines += [
                        f"    r{register}, i{register} = _cmul(r{register}, i{register}, {c!r}, {s!r})",
                    ]
                lines.extend(_emit_natural_order_codelet_call("    ", radix, config.direction, indices))
        previous_span = span
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


def _build_flat_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    """Keep exchange indices in one explicit logical vector address space."""
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(json.dumps({"version": 6, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    bits = n.bit_length() - 1
    reversed_index = " | ".join(
        f"(((frequency >> {bit}) & 1) << {bits - bit - 1})" for bit in range(bits))
    ilen = n // 2 + 1 if config.mode == "c2r" else n
    olen = n // 2 + 1 if config.mode == "r2c" else n
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, row_offset):",
        f"    flat = tl.arange(0, {n * pack})",
        f"    lane = flat // {n}", f"    frequency = flat % {n}",
        f"    row = row_offset + tl.program_id(0).to(tl.int64) * {pack} + lane",
        "    mask = row < nbatch", "    source_frequency = frequency",
        f"    input_base = (row // {config.input_stride}) * {ilen * config.input_stride} + row % {config.input_stride}"]
    if config.mode == "c2r":
        lines += [f"    conjugate = source_frequency > {n // 2}",
            f"    source_frequency = tl.where(conjugate, {n} - source_frequency, source_frequency)"]
    lines += [f"    source_index = input_base + source_frequency * {config.input_stride}"]
    if config.mode == "r2c":
        lines += ["    real = tl.load(in_ptr + source_index, mask, 0.0)",
            f"    imag = tl.full(({n * pack},), 0.0, tl.float32)"]
    else:
        lines += ["    real = tl.load(in_ptr + source_index * 2, mask, 0.0)",
            "    imag = tl.load(in_ptr + source_index * 2 + 1, mask, 0.0)"]
        if config.mode == "c2r":
            lines += ["    imag = tl.where(conjugate, -imag, imag)",
                f"    imag = tl.where((source_frequency == 0) | (source_frequency == {n // 2}), 0.0, imag)"]
    lines += [f"    reverse = lane * {n} + ({reversed_index})",
        "    real = tl.gather(real, reverse, 0)", "    imag = tl.gather(imag, reverse, 0)"]
    if config.twiddle == "local_table":
        lines += [f"    roots = tl.load(twiddle_ptr + tl.arange(0, {n}))"]
    for stage in range(1, bits + 1):
        half, span = 1 << (stage - 1), 1 << stage
        lines += [f"    partner = flat ^ {half}",
            "    pr = tl.gather(real, partner, 0)", "    pi = tl.gather(imag, partner, 0)",
            f"    upper = (frequency & {half}) != 0",
            "    ar = tl.where(upper, pr, real)", "    ai = tl.where(upper, pi, imag)",
            "    br = tl.where(upper, real, pr)", "    bi = tl.where(upper, imag, pi)"]
        if half == 1:
            lines += ["    tr = br", "    ti = bi"]
        elif half == 2:
            rr, ri = ("bi", "-br") if config.direction == "forward" else ("-bi", "br")
            lines += [f"    tr = tl.where((frequency & 1) == 0, br, {rr})",
                f"    ti = tl.where((frequency & 1) == 0, bi, {ri})"]
        else:
            if config.twiddle == "compute":
                lines += [f"    angle = (frequency % {half}).to(tl.float32) * {(-1 if config.direction == 'forward' else 1) * 2 * math.pi / span!r}",
                    "    wr = tl.cos(angle)", "    wi = tl.sin(angle)"]
            else:
                lines += [f"    tw = (frequency % {half}) * {n // span} * 2"]
                if config.twiddle == "local_table":
                    lines += ["    wr = tl.gather(roots, tw, 0)", "    wi = tl.gather(roots, tw + 1, 0)"]
                else:
                    lines += ["    wr = tl.load(twiddle_ptr + tw)", "    wi = tl.load(twiddle_ptr + tw + 1)"]
            lines += ["    tr, ti = _cmul(br, bi, wr, wi)"]
        lines += ["    real = ar + tl.where(upper, -tr, tr)",
                  "    imag = ai + tl.where(upper, -ti, ti)"]
    step = config.output_rows_per_batch or config.output_stride
    lines += [f"    output_base = (row // {step}) * {olen * step} + row % {step}",
        f"    target = output_base + frequency * {step}", f"    valid = mask & (frequency < {olen})"]
    if config.mode == "c2r":
        lines += ["    tl.store(out_ptr + target, real, valid)"]
    else:
        lines += ["    tl.store(out_ptr + target * 2, real, valid)",
            "    tl.store(out_ptr + target * 2 + 1, imag, valid)"]
    return name, "\n".join(lines) + "\n"


def _build_interleaved_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    """Operate on interleaved complex vectors with contiguous GM transactions."""
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(json.dumps({"version": 6, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    bits = n.bit_length() - 1
    reverse = " | ".join(f"(((frequency >> {b}) & 1) << {bits-b-1})" for b in range(bits))
    ilen = n // 2 + 1 if config.mode == "c2r" else n
    olen = n // 2 + 1 if config.mode == "r2c" else n
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, row_offset):",
        f"    flat = tl.arange(0, {2 * n * pack})", "    component = flat % 2",
        f"    frequency = (flat // 2) % {n}", f"    lane = flat // {2 * n}",
        f"    row = row_offset + tl.program_id(0).to(tl.int64) * {pack} + lane",
        "    mask = row < nbatch", "    source_frequency = frequency",
        f"    base = (row // {config.input_stride}) * {ilen * config.input_stride} + row % {config.input_stride}"]
    if config.mode == "c2r":
        lines += [f"    conjugate = frequency > {n // 2}",
            f"    source_frequency = tl.where(conjugate, {n} - frequency, frequency)"]
    lines += [f"    source = base + source_frequency * {config.input_stride}"]
    if config.mode == "r2c":
        lines += ["    value = tl.load(in_ptr + source, mask & (component == 0), 0.0)"]
    else:
        lines += ["    value = tl.load(in_ptr + source * 2 + component, mask, 0.0)"]
        if config.mode == "c2r":
            lines += ["    value = tl.where(conjugate & (component != 0), -value, value)",
                f"    value = tl.where(((frequency == 0) | (frequency == {n // 2})) & (component != 0), 0.0, value)"]
    lines += [f"    reverse = lane * {2 * n} + ({reverse}) * 2 + component",
        "    value = tl.gather(value, reverse, 0)"]
    if config.twiddle == "local_table":
        lines += [f"    roots = tl.load(twiddle_ptr + tl.arange(0, {n}))"]
    for stage in range(1, bits + 1):
        half, span = 1 << (stage - 1), 1 << stage
        lines += [f"    partner = tl.gather(value, flat ^ {half * 2}, 0)",
            f"    upper = (frequency & {half}) != 0",
            "    a = tl.where(upper, partner, value)", "    b = tl.where(upper, value, partner)"]
        if half == 1:
            lines += ["    term = b"]
        else:
            lines += ["    rotated = tl.gather(b, flat ^ 1, 0)"]
            if half == 2:
                sign = 1.0 if config.direction == "forward" else -1.0
                lines += [f"    rotated = rotated * tl.where(component == 0, {sign}, {-sign})",
                    "    term = tl.where((frequency & 1) == 0, b, rotated)"]
            else:
                if config.twiddle == "compute":
                    lines += [f"    angle = (frequency % {half}).to(tl.float32) * {(-1 if config.direction == 'forward' else 1) * 2 * math.pi / span!r}",
                        "    wr = tl.cos(angle)", "    wi = tl.sin(angle)"]
                else:
                    lines += [f"    tw = (frequency % {half}) * {n // span} * 2"]
                    if config.twiddle == "local_table":
                        lines += ["    wr = tl.gather(roots, tw, 0)", "    wi = tl.gather(roots, tw + 1, 0)"]
                    else:
                        lines += ["    wr = tl.load(twiddle_ptr + tw)", "    wi = tl.load(twiddle_ptr + tw + 1)"]
                lines += ["    term = b * wr + rotated * tl.where(component == 0, -wi, wi)"]
        lines += ["    value = a + tl.where(upper, -term, term)"]
    step = config.output_rows_per_batch or config.output_stride
    lines += [f"    base_out = (row // {step}) * {olen * step} + row % {step}",
        f"    target = base_out + frequency * {step}", f"    valid = mask & (frequency < {olen})"]
    if config.mode == "c2r":
        lines += ["    tl.store(out_ptr + target, value, valid & (component == 0))"]
    else:
        lines += ["    tl.store(out_ptr + target * 2 + component, value, valid)"]
    return name, "\n".join(lines) + "\n"


def _build_tensor_fft_source(config: TensorFFTConfig) -> tuple[str, str]:
    if config.exchange == "static":
        return _build_static_fft_source(config)
    if config.exchange == "flat":
        return _build_flat_fft_source(config)
    if config.exchange == "interleaved":
        return _build_interleaved_fft_source(config)
    n, pack = config.length, config.rows_per_program
    identity = hashlib.sha256(
        json.dumps({"version": 6, **asdict(config)}, sort_keys=True).encode()
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
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, row_offset):",
        f"    row = row_offset + tl.program_id(0).to(tl.int64) * {pack} + tl.arange(0, {pack})",
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
    write_text_atomic(module_path, _module_source(source, radices=tuple(sorted(set(config.factors))) if config.exchange == "static" else ()))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,i32,i64",
        "num_warps": config.num_warps,
        "num_stages": 1,
        "batch_per_block": config.rows_per_program,
        "arg_names": ["in_ptr", "out_ptr", "twiddle_ptr", "nbatch", "row_offset"],
        "kernel_type": "tensor_fft",
        "dtype": "complex64",
        "implementation": "triton",
        "tensor_fft_config": json.loads(json.dumps(asdict(config))),
        "generator_version": 6,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
