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
    exchange: str = "split_register"
    num_warps: int = 4
    factors: tuple[int, ...] = ()

    def __post_init__(self):
        if self.length < 2:
            raise ValueError("tensor FFT length must be >= 2")
        if self.exchange == "split_register" and self.length & (self.length - 1):
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
        if self.exchange not in {"static", "split_register"}:
            raise ValueError("unsupported tensor FFT exchange")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("unsupported tensor FFT launch width")


    @property
    def row_tile_extent(self) -> int:
        """Keep every program within one strided/permuted layout group."""
        extents = [value for value in (self.input_stride, self.output_stride,
                   self.output_rows_per_batch) if value > 1]
        return math.gcd(*extents) if extents else 0


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
    identity = hashlib.sha256(json.dumps({"version": 7, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_fft_n{n}_{config.mode}_{config.direction}_{identity}"
    input_length = n // 2 + 1 if config.mode == "c2r" else n
    output_length = n // 2 + 1 if config.mode == "r2c" else n
    lines = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, program_offset):",
        "    program = program_offset + tl.program_id(0).to(tl.int64)",
        f"    lane = tl.arange(0, {pack})",
    ]
    extent = config.row_tile_extent
    if extent:
        tiles = (extent + pack - 1) // pack
        lines += [f"    inner_row = (program % {tiles}) * {pack} + lane",
            f"    group_row = (program // {tiles}) * {extent}",
            "    row = group_row + inner_row",
            f"    mask = (row < nbatch) & (inner_row < {extent})"]
    else:
        lines += [f"    row = program * {pack} + lane", "    mask = row < nbatch"]
    if extent and config.input_stride > 1:
        lines += [f"    input_base = (group_row // {config.input_stride}) * {input_length * config.input_stride} + group_row % {config.input_stride} + inner_row"]
    else:
        lines += [f"    input_base = row * {input_length}"]
    if config.exchange == "split_register" and config.mode != "c2r":
        frequency = "tl.arange(0, " + str(n) + ")"
        lines += [f"    frequency = {frequency}", "    source_frequency = frequency"]
        if config.input_stride == 1:
            lines += ["    source = input_base[:, None] + source_frequency[None, :]"]
            load_mask = "mask[:, None]"
            transpose = False
        else:
            lines += [f"    source = input_base[None, :] + source_frequency[:, None] * {config.input_stride}"]
            load_mask = "mask[None, :]"
            transpose = True
        if config.mode == "r2c":
            lines += [f"    loaded_r = tl.load(in_ptr + source, {load_mask}, 0.0)",
                f"    loaded_i = tl.full(({n}, {pack}), 0.0, tl.float32)" if transpose else f"    loaded_i = tl.full(({pack}, {n}), 0.0, tl.float32)"]
        else:
            lines += [f"    loaded_r = tl.load(in_ptr + source * 2, {load_mask}, 0.0)",
                f"    loaded_i = tl.load(in_ptr + source * 2 + 1, {load_mask}, 0.0)"]
        if transpose:
            lines += ["    loaded_r = tl.trans(loaded_r, (1, 0))", "    loaded_i = tl.trans(loaded_i, (1, 0))"]
        inverse_indices = { _digit_reversed_index(i, config.factors): i for i in range(n) }
        def split_input(name, points, prefix, path=""):
            if len(points) == 1:
                lines.append(f"    {prefix}{inverse_indices[points[0]]} = tl.reshape({name}, ({pack},))")
                return
            even, odd = f"split_{prefix}{path}e", f"split_{prefix}{path}o"
            lines.append(f"    {even}, {odd} = tl.split(tl.reshape({name}, ({pack}, {len(points) // 2}, 2)))")
            split_input(even, points[::2], prefix, path + "e")
            split_input(odd, points[1::2], prefix, path + "o")
        split_input("loaded_r", tuple(range(n)), "r")
        split_input("loaded_i", tuple(range(n)), "i")
    else:
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
        step = extent
    else:
        step = config.output_stride
    if config.row_tile_extent and step > 1:
        lines += [f"    output_base = (group_row // {step}) * {output_length * step} + group_row % {step} + inner_row"]
    else:
        lines += [f"    output_base = row * {output_length}"]
    if config.exchange == "split_register":
        lines += [f"    frequency = tl.arange(0, {n})"]
        def join_output(points, prefix, path=""):
            if len(points) == 1:
                return f"tl.reshape({prefix}{points[0]}, ({pack}, 1))"
            even = join_output(points[::2], prefix, path + "e")
            odd = join_output(points[1::2], prefix, path + "o")
            name = f"joined_{prefix}{path}"
            lines.append(f"    {name} = tl.reshape(tl.join({even}, {odd}), ({pack}, {len(points)}))")
            return name
        real = join_output(tuple(range(n)), "r")
        imag = join_output(tuple(range(n)), "i") if config.mode != "c2r" else ""
        if step == 1:
            lines += ["    target = output_base[:, None] + frequency[None, :]",
                f"    valid = mask[:, None] & (frequency[None, :] < {output_length})"]
        else:
            lines += [f"    target = output_base[None, :] + frequency[:, None] * {step}",
                f"    valid = mask[None, :] & (frequency[:, None] < {output_length})",
                f"    output_r = tl.trans({real}, (1, 0))"]
            real = "output_r"
            if imag:
                lines += [f"    output_i = tl.trans({imag}, (1, 0))"]
                imag = "output_i"
        if config.mode == "c2r":
            lines += [f"    tl.store(out_ptr + target, {real}, valid)"]
        else:
            lines += [f"    tl.store(out_ptr + target * 2, {real}, valid)",
                f"    tl.store(out_ptr + target * 2 + 1, {imag}, valid)"]
    else:
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


def emit_tensor_fft(config: TensorFFTConfig, out_dir: Path) -> dict:
    name, source = _build_static_fft_source(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source, radices=tuple(sorted(set(config.factors)))))
    extent = config.row_tile_extent
    metadata = {
        "module_path": str(module_path), "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,i32,i64",
        "num_warps": config.num_warps, "num_stages": 1,
        "batch_per_block": config.rows_per_program,
        "row_group_extent": extent or config.rows_per_program,
        "programs_per_row_group": (extent + config.rows_per_program - 1) // config.rows_per_program if extent else 1,
        "arg_names": ["in_ptr", "out_ptr", "twiddle_ptr", "nbatch", "program_offset"],
        "kernel_type": "tensor_fft", "dtype": "complex64", "implementation": "triton",
        "tensor_fft_config": json.loads(json.dumps(asdict(config))), "generator_version": 7,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
