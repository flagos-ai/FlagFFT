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
from .kernels_fft_prime import emit_rader_inner, emit_rader_call, rader_factors


def _paired_radix(radix: int, strategy: str = "auto") -> bool:
    eligible = 3 <= radix <= 31 and radix % 2 == 1
    return eligible and (strategy in {"paired", "paired_loop"} or
        strategy == "auto" and radix not in _NATURAL_ORDER_CODELET_RADICES)


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
    prime_strategy: str = "auto"

    def __post_init__(self):
        if self.prime_strategy not in {"auto", "natural", "paired", "paired_loop", "rader"}:
            raise ValueError("unsupported prime codelet strategy")
        if self.prime_strategy == "rader":
            rader_factors(self.radix)
        elif self.radix not in _NATURAL_ORDER_CODELET_RADICES and not _paired_radix(self.radix, self.prime_strategy):
            raise ValueError("unsupported DIF radix")
        if self.prime_strategy in {"paired", "paired_loop"} and not _paired_radix(self.radix, "paired"):
            raise ValueError("paired codelet requires an odd radix from 3 through 31")
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
    width = (1 << (config.radix - 1).bit_length()) if _paired_radix(config.radix, config.prime_strategy) else (1 << (config.radix // 2).bit_length())
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
    identity = hashlib.sha256(json.dumps({"version": 14, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
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
    transform_extent = 1 << (pack * ilen - 1).bit_length()
    cached_transform = (config.input_stride == 1 and
        (programs == 1 or first and config.mode == "c2r") and
        transform_extent * (4 if first and config.mode == "r2c" else 8) <= 16 * 1024)
    cache_groups = (block + 2 * inner - 2) // inner
    window_extent = 1 << (cache_groups * span - 1).bit_length()
    cached_window = (not cached_transform and
        config.input_stride == 1 and not (first and config.mode == "c2r") and
        pack * window_extent * 8 <= 32 * 1024 and
        window_extent <= 8 * block * radix)
    if cached_transform or cached_window:
        cache_extent = transform_extent if cached_transform else window_extent
        lines += [f"    cache_offset = tl.arange(0, {cache_extent})"]
        if cached_transform:
            lines += [f"    cache_row_begin = (program // {programs}) * {pack}",
                f"    cache_rows = tl.minimum({pack}, tl.maximum(nbatch - cache_row_begin, 0)).to(tl.int32)",
                f"    cache_mask = cache_offset < cache_rows * {ilen}",
                f"    cache_source = cache_row_begin * {ilen} + cache_offset"]
        else:
            lines += [f"    cache_k_begin = (program % {programs}) * {block}",
                f"    cache_j = cache_k_begin % {inner}",
                f"    cache_begin = (cache_k_begin // {inner}) * {span}",
                f"    cache_count = ((cache_j + {block + inner - 1}) // {inner}) * {span}",
                "    cache_frequency = cache_begin + cache_offset",
                f"    cache_mask = valid_row[:, None] & (cache_frequency[None, :] < {n}) & (cache_offset[None, :] < cache_count)"]
            lines += ["    cache_source = base_in[:, None] + cache_frequency[None, :]"]
        if first and config.mode == "r2c":
            lines += ["    cached_r = tl.load(in_ptr + cache_source, cache_mask, 0.0)",
                f"    cached_i = tl.full({(cache_extent,) if cached_transform else (pack, cache_extent)}, 0.0, tl.float32)"]
        else:
            point = ("    cached_point = tl.load(in_ptr + cache_source[:, None] * 2 + tl.arange(0, 2)[None, :], cache_mask[:, None], 0.0)" if cached_transform else
                "    cached_point = tl.load(in_ptr + cache_source[:, :, None] * 2 + tl.arange(0, 2)[None, None, :], cache_mask[:, :, None], 0.0)")
            lines += [point,
                "    cached_r, cached_i = tl.split(cached_point)"]

    def input_lines(digit, real, imag, indent="    "):
        if cached_window:
            return [f"{indent}cache_k = cache_j + tl.arange(0, {block})",
                f"{indent}cache_index = (cache_k // {inner}) * {span} + cache_k % {inner} + {digit} * {inner}",
                f"{indent}cache_index = tl.broadcast_to(cache_index[None, :], ({pack}, {block}))",
                f"{indent}{real} = tl.gather(cached_r, cache_index, 1)",
                f"{indent}{imag} = tl.gather(cached_i, cache_index, 1)"]
        # Every valid first-stage butterfly belongs to the sole full-length
        # group. Keep its GM address affine instead of obscuring the dense
        # read with a quotient/remainder pair that only cancels under a mask.
        result = [f"{indent}frequency = k + {digit} * {inner}" if first else
            f"{indent}frequency = group * {span} + j + {digit} * {inner}"]
        if first and config.mode == "c2r":
            result += [f"{indent}conjugate = frequency > {n // 2}",
                f"{indent}frequency = tl.where(conjugate, {n} - frequency, frequency)"]
        if cached_transform:
            result += [f"{indent}cache_index = tl.where(valid_k, frequency, 0)",
                f"{indent}cache_index = tl.arange(0, {pack})[:, None] * {ilen} + cache_index[None, :]",
                f"{indent}cache_index = tl.reshape(cache_index, ({pack * block},))",
                f"{indent}{real} = tl.reshape(tl.gather(cached_r, cache_index, 0), ({pack}, {block}))",
                f"{indent}{imag} = tl.reshape(tl.gather(cached_i, cache_index, 0), ({pack}, {block}))"]
        elif first and config.mode == "r2c":
            result += [f"{indent}source = base_in[:, None] + frequency[None, :] * {config.input_stride}"]
            result += [f"{indent}{real} = tl.load(in_ptr + source, mask, 0.0)",
                f"{indent}{imag} = tl.full(({pack}, {block}), 0.0, tl.float32)"]
        else:
            result += [f"{indent}source = base_in[:, None] + frequency[None, :] * {config.input_stride}"]
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
                f"{indent}valid = mask[None, :, :] & ({digit}[:, None, None] < {radix})"]
            if last and config.mode == "r2c":
                result += [f"{indent}valid = valid & (frequency_out < {olen})"]
        else:
            result += [f"{indent}frequency_out = {digit} * {ndiv} + k",
                f"{indent}target = base_out[:, None] + frequency_out[None, :] * {step}",
                f"{indent}valid = mask & (frequency_out[None, :] < {olen})"]
        if last and config.mode == "c2r":
            result += [f"{indent}tl.store(out_ptr + target, {real}, valid)"]
        elif expanded:
            result += [f"{indent}point = tl.join({real}, {imag})",
                f"{indent}tl.store(out_ptr + target[:, :, :, None] * 2 + tl.arange(0, 2)[None, None, None, :], point, valid[:, :, :, None])"]
        else:
            result += [f"{indent}point = tl.join({real}, {imag})"]
            result += [f"{indent}tl.store(out_ptr + target[:, :, None] * 2 + tl.arange(0, 2)[None, None, :], point, valid[:, :, None])"]
        return result

    if not _paired_radix(radix, config.prime_strategy):
        for digit in range(radix):
            lines += input_lines(str(digit), f"r{digit}", f"i{digit}")
        lines += emit_rader_call(radix, config.direction) if config.prime_strategy == "rader" else _emit_natural_order_codelet_call("    ", radix, config.direction)
        for digit in range(radix):
            lines += output_lines(str(digit), f"r{digit}", f"i{digit}")
    else:
        half = radix // 2
        width = 1 << (radix - 1).bit_length()
        lines += [f"    output = tl.arange(0, {width})"]
        lines += input_lines("0", "x0r", "x0i")
        lines += [f"    cr = tl.full(({width}, {pack}, {block}), 0.0, tl.float32) + x0r[None, :, :]",
            f"    ci = tl.full(({width}, {pack}, {block}), 0.0, tl.float32) + x0i[None, :, :]",
            f"    sr = tl.full(({width}, {pack}, {block}), 0.0, tl.float32)",
            f"    si = tl.full(({width}, {pack}, {block}), 0.0, tl.float32)"]
        loop = config.prime_strategy == "paired_loop"
        indent = "        " if loop else "    "
        if loop:
            lines += [f"    for digit in range(1, {half + 1}):"]
        for digit in (["digit"] if loop else range(1, half + 1)):
            lines += input_lines(str(digit), "xr", "xi", indent)
            lines += input_lines(f"({radix} - {digit})", "yr", "yi", indent)
            lines += [f"{indent}root = ({n} + {digit} * {width} + output) * 2",
                f"{indent}root_point = tl.load(twiddle_ptr + root[:, None] + tl.arange(0, 2)[None, :])",
                f"{indent}wr, wi = tl.split(root_point)",
                f"{indent}wr, wi = wr[:, None, None], wi[:, None, None]",
                f"{indent}cr = cr + (xr + yr)[None, :, :] * wr",
                f"{indent}ci = ci + (xi + yi)[None, :, :] * wr",
                f"{indent}sr = sr + (xr - yr)[None, :, :] * wi",
                f"{indent}si = si + (xi - yi)[None, :, :] * wi"]
        # Keep every output frequency in one tensor. This avoids separate
        # mirrored stores sharing accumulator buffers in the backend.
        lines += ["    result_r, result_i = cr - si, ci + sr"]
        lines += output_lines("output", "result_r", "result_i", expanded=True)
    helpers = emit_rader_inner(radix) if config.prime_strategy == "rader" else ""
    return name, helpers + "\n".join(lines) + "\n"


def emit_fft_stage(config: FFTStageConfig, out_dir: Path) -> dict:
    name, source = build_fft_stage(config)
    path = out_dir / f"{name}.py"
    radices = rader_factors(config.radix) if config.prime_strategy == "rader" else (config.radix,)
    write_text_atomic(path, _module_source(source, radices=tuple(sorted(set(radices)))))
    metadata = {"module_path": str(path), "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,i32,i64", "num_warps": 4,
        "num_stages": 1, "arg_names": ["in_ptr", "out_ptr", "twiddle_ptr", "nbatch", "program_offset"],
        "batch_per_block": config.rows_per_program,
        "row_group_extent": config.rows_per_program,
        "programs_per_row_group": (config.length // config.radix + config.block - 1) // config.block,
        "kernel_type": "fft_dif_stage", "dtype": "complex64", "implementation": "triton",
        "generator_version": 14, "fft_stage_config": asdict(config)}
    write_text_atomic(path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
