"""Vector DIF autosort stages using the shared natural-order radix codelets.

Each stage writes contiguous frequency planes. Twiddle tiles are loaded once
and broadcast over independent FFT rows, avoiding repeated indirect GM reads.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .artifacts import write_text_atomic
from .kernels_common import _NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import _emit_natural_order_codelet_call
from .metadata import _module_source


def _paired_radix(radix: int) -> bool:
    return 13 <= radix <= 31 and radix % 2 == 1


@dataclass(frozen=True)
class FFTStageConfig:
    length: int
    radix: int
    span: int
    direction: str = "forward"
    mode: str = "complex"
    block: int = 128
    rows_per_program: int = 1
    input_stride: int = 1
    output_stride: int = 1
    output_rows_per_batch: int = 0

    def __post_init__(self):
        if self.radix not in _NATURAL_ORDER_CODELET_RADICES and not _paired_radix(self.radix):
            raise ValueError("unsupported DIF radix")
        if self.length < 2 or self.span < self.radix or self.length % self.span or self.span % self.radix:
            raise ValueError("invalid DIF transform or stage span")
        if self.direction not in {"forward", "inverse"}:
            raise ValueError("invalid DIF direction")
        if self.mode not in {"complex", "r2c", "c2r"} or (
            self.mode == "r2c" and self.direction != "forward" or
            self.mode == "c2r" and self.direction != "inverse"
        ):
            raise ValueError("invalid DIF real IO mode")
        for value in (self.block, self.rows_per_program):
            if value < 1 or value & (value - 1):
                raise ValueError("DIF tile extents must be powers of two")
        if self.input_stride < 1 or self.output_stride < 1 or self.output_rows_per_batch < 0:
            raise ValueError("invalid DIF strides")
        if self.output_rows_per_batch and self.output_stride != 1:
            raise ValueError("permuted DIF output uses output_stride=1")


def stage_twiddle_values(config: FFTStageConfig) -> list[float]:
    """Interleaved FP32 roots, replicated over groups for affine tile reads."""
    ndiv = config.length // config.radix
    inner = config.span // config.radix
    sign = -1 if config.direction == "forward" else 1
    values = []
    for digit in range(config.radix):
        for k in range(ndiv):
            angle = sign * 2 * math.pi * digit * (k % inner) / config.span
            values.extend((math.cos(angle), math.sin(angle)))
    # Odd-radix paired-vector codelets use a small contiguous root matrix.
    width = 1 << (config.radix // 2).bit_length()
    for digit in range(config.radix):
        for output in range(width):
            angle = sign * 2 * math.pi * output * digit / config.radix
            values.extend((math.cos(angle), math.sin(angle)))
    return values


def build_fft_stage(config: FFTStageConfig) -> tuple[str, str]:
    n, radix, span, block, pack = (config.length, config.radix, config.span,
                                  config.block, config.rows_per_program)
    ndiv, inner = n // radix, span // radix
    first, last = span == n, span == radix
    ilen = n // 2 + 1 if first and config.mode == "c2r" else n
    olen = n // 2 + 1 if last and config.mode == "r2c" else n
    identity = hashlib.sha256(json.dumps({"version": 2, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_fft_dif_n{n}_r{radix}_s{span}_{identity}"
    programs = (ndiv + block - 1) // block
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, program_offset):",
        "    program = program_offset + tl.program_id(0).to(tl.int64)",
        f"    row = (program // {programs}) * {pack} + tl.arange(0, {pack})",
        f"    k = (program % {programs}) * {block} + tl.arange(0, {block})",
        f"    valid_k = k < {ndiv}", "    valid_row = row < nbatch",
        "    mask = valid_row[:, None] & valid_k[None, :]",
        f"    group = k // {inner}", f"    j = k % {inner}",
        f"    base_in = (row // {config.input_stride}) * {ilen * config.input_stride} + row % {config.input_stride}",
        f"    base_out = (row // {config.output_rows_per_batch or config.output_stride}) * {olen * (config.output_rows_per_batch or config.output_stride)} + row % {config.output_rows_per_batch or config.output_stride}"]

    def input_lines(digit, real, imag, indent="    "):
        result = [f"{indent}frequency = group * {span} + j + {digit} * {inner}"]
        if first and config.mode == "c2r":
            result += [f"{indent}conjugate = frequency > {n // 2}",
                f"{indent}frequency = tl.where(conjugate, {n} - frequency, frequency)"]
        result += [f"{indent}source = base_in[:, None] + frequency[None, :] * {config.input_stride}"]
        if first and config.mode == "r2c":
            result += [f"{indent}{real} = tl.load(in_ptr + source, mask, 0.0)",
                f"{indent}{imag} = tl.full(({pack}, {block}), 0.0, tl.float32)"]
        else:
            result += [f"{indent}point = tl.load(in_ptr + source[:, :, None] * 2 + tl.arange(0, 2)[None, None, :], mask[:, :, None], 0.0)",
                f"{indent}{real}, {imag} = tl.split(point)"]
            if first and config.mode == "c2r":
                boundary = f"(frequency[None, :] == 0)"
                if n % 2 == 0:
                    boundary += f" | (frequency[None, :] == {n // 2})"
                result += [f"{indent}{imag} = tl.where(conjugate[None, :], -{imag}, {imag})",
                    f"{indent}{imag} = tl.where({boundary}, 0.0, {imag})"]
        return result

    def output_lines(digit, real, imag, indent="    ", expanded=False):
        result = []
        if not last:
            if expanded:
                result += [f"{indent}root = ({digit}[:, None] * {ndiv} + k[None, :]) * 2",
                    f"{indent}root_mask = ({digit}[:, None] < {radix}) & valid_k[None, :]",
                    f"{indent}root_point = tl.load(twiddle_ptr + root[:, :, None] + tl.arange(0, 2)[None, None, :], root_mask[:, :, None], 0.0)",
                    f"{indent}wr, wi = tl.split(root_point)",
                    f"{indent}wr, wi = wr[:, None, :], wi[:, None, :]"]
            else:
                result += [f"{indent}root = ({digit} * {ndiv} + k) * 2",
                    f"{indent}root_point = tl.load(twiddle_ptr + root[:, None] + tl.arange(0, 2)[None, :], valid_k[:, None], 0.0)",
                    f"{indent}wr, wi = tl.split(root_point)",
                    f"{indent}wr, wi = wr[None, :], wi[None, :]"]
            result += [f"{indent}{real}, {imag} = _cmul({real}, {imag}, wr, wi)"]
        step = config.output_rows_per_batch or config.output_stride
        if expanded:
            result += [f"{indent}frequency_out = {digit}[:, None, None] * {ndiv} + k[None, None, :]",
                f"{indent}target = base_out[None, :, None] + frequency_out * {step}",
                f"{indent}valid = mask[None, :, :] & ({digit}[:, None, None] < {radix}) & (frequency_out < {olen})"]
        else:
            result += [f"{indent}frequency_out = {digit} * {ndiv} + k",
                f"{indent}target = base_out[:, None] + frequency_out[None, :] * {step}",
                f"{indent}valid = mask & (frequency_out[None, :] < {olen})"]
        if last and config.mode == "c2r":
            result += [f"{indent}tl.store(out_ptr + target, {real}, valid)"]
        else:
            result += [f"{indent}point = tl.join({real}, {imag})"]
            if expanded:
                result += [f"{indent}tl.store(out_ptr + target[:, :, :, None] * 2 + tl.arange(0, 2)[None, None, None, :], point, valid[:, :, :, None])"]
            else:
                result += [f"{indent}tl.store(out_ptr + target[:, :, None] * 2 + tl.arange(0, 2)[None, None, :], point, valid[:, :, None])"]
        return result

    if not _paired_radix(radix):
        for digit in range(radix):
            lines += input_lines(str(digit), f"r{digit}", f"i{digit}")
        lines += _emit_natural_order_codelet_call("    ", radix, config.direction)
        for digit in range(radix):
            lines += output_lines(str(digit), f"r{digit}", f"i{digit}")
    else:
        half = radix // 2
        width = 1 << half.bit_length()
        lines += [f"    output = tl.arange(0, {width})"]
        lines += input_lines("0", "x0r", "x0i")
        lines += [f"    cr = tl.full(({width}, {pack}, {block}), 0.0, tl.float32) + x0r[None, :, :]",
            f"    ci = tl.full(({width}, {pack}, {block}), 0.0, tl.float32) + x0i[None, :, :]",
            f"    sr = tl.full(({width}, {pack}, {block}), 0.0, tl.float32)",
            f"    si = tl.full(({width}, {pack}, {block}), 0.0, tl.float32)",
            f"    for digit in range(1, {half + 1}):"]
        lines += input_lines("digit", "xr", "xi", "        ")
        lines += input_lines(f"({radix} - digit)", "yr", "yi", "        ")
        lines += [f"        root = ({n} + digit * {width} + output) * 2",
            "        root_point = tl.load(twiddle_ptr + root[:, None] + tl.arange(0, 2)[None, :])",
            "        wr, wi = tl.split(root_point)",
            "        wr, wi = wr[:, None, None], wi[:, None, None]",
            "        cr = cr + (xr + yr)[None, :, :] * wr",
            "        ci = ci + (xi + yi)[None, :, :] * wr",
            "        sr = sr + (xr - yr)[None, :, :] * wi",
            "        si = si + (xi - yi)[None, :, :] * wi",
            "    low_r, low_i = cr - si, ci + sr",
            "    high_r, high_i = cr + si, ci - sr",
            f"    output = tl.where(output <= {half}, output, {radix})"]
        lines += output_lines("output", "low_r", "low_i", expanded=True)
        lines += [f"    output = tl.where((output > 0) & (output <= {half}), {radix} - output, {radix})"]
        lines += output_lines("output", "high_r", "high_i", expanded=True)
    return name, "\n".join(lines) + "\n"


def emit_fft_stage(config: FFTStageConfig, out_dir: Path) -> dict:
    name, source = build_fft_stage(config)
    path = out_dir / f"{name}.py"
    write_text_atomic(path, _module_source(source, radices=(config.radix,)))
    metadata = {"module_path": str(path), "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,i32,i64", "num_warps": 4,
        "num_stages": 1, "arg_names": ["in_ptr", "out_ptr", "twiddle_ptr", "nbatch", "program_offset"],
        "batch_per_block": config.rows_per_program,
        "row_group_extent": config.rows_per_program,
        "programs_per_row_group": (config.length // config.radix + config.block - 1) // config.block,
        "kernel_type": "fft_dif_stage", "dtype": "complex64", "implementation": "triton",
        "generator_version": 2, "fft_stage_config": asdict(config)}
    write_text_atomic(path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
