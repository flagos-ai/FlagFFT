"""Global-memory Stockham mapping of the shared register radix codelets."""

from .kernels_common import _NATURAL_ORDER_CODELET_RADICES, _dtype_suffix, _ix_backend_active, _maca_knob
from .kernels_leaf import (
    _emit_natural_order_codelet_call, _emit_radix16_codelet_call,
    _emit_natural_order_radix32_codelet_call,
    _distributed_join_tree,
)


def build_stockham_stage(n: int, radix: int, direction: str, dtype: str, stage_span: int = 0,
                        block: int = 128):
    if radix not in _NATURAL_ORDER_CODELET_RADICES | {16, 32} or n % radix:
        raise ValueError(f"unsupported Stockham stage n={n}, radix={radix}")
    if stage_span < 0 or (stage_span and n % (radix * stage_span)):
        raise ValueError(f"invalid Stockham span {stage_span} for n={n}, radix={radix}")
    if block < 1 or block > 128 or block & (block - 1):
        raise ValueError(f"invalid Stockham butterfly block {block}")
    if radix in (13, 17, 19):
        return _build_vector_stage(n, radix, direction, dtype, stage_span, block)
    name = f"stockham_{direction}_n{n}_r{radix}_s{stage_span}_b{block}_{_dtype_suffix(dtype)}"
    body = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, span, nbatch):",
        *([f"    span = {stage_span}"] if stage_span else []),
        f"    index = tl.program_id(0).to(tl.int64) * {block} + tl.arange(0, {block})",
        f"    mask = index < nbatch.to(tl.int64) * {n // radix}",
        f"    batch = index // {n // radix}",
        f"    k = index % {n // radix}",
        "    j = k % span",
    ]
    for digit in range(radix):
        body.extend(
            [
                f"    src{digit} = (batch * {n} + k + {digit * (n // radix)}) * 2",
                f"    r{digit} = tl.load(in_ptr + src{digit}, mask, 0)",
                f"    i{digit} = tl.load(in_ptr + src{digit} + 1, mask, 0)",
            ]
        )
        if digit and stage_span != 1:
            body.extend(
                [
                    f"    tw{digit} = {digit} * j * ({n} // ({radix} * span)) * 2",
                    f"    wr{digit} = tl.load(twiddle_ptr + tw{digit}, mask, 0)",
                    f"    wi{digit} = tl.load(twiddle_ptr + tw{digit} + 1, mask, 0)",
                    f"    r{digit}, i{digit} = _cmul(r{digit}, i{digit}, wr{digit}, wi{digit})",
                ]
            )
    if radix == 16:
        body.extend(_emit_radix16_codelet_call("    ", direction))
    elif radix == 32:
        body.extend(_emit_natural_order_radix32_codelet_call("    ", direction))
    else:
        body.extend(_emit_natural_order_codelet_call("    ", radix, direction))
    body.append(f"    dst = batch * {n} + {radix} * k - {radix - 1} * j")
    if (_ix_backend_active() and _maca_knob("STOCKHAM_STORE_JOIN") == "1"
            and stage_span and stage_span < block and radix & (radix - 1) == 0):
        for component in ('r', 'i'):
            joined = _distributed_join_tree([f'{component}{digit}' for digit in range(radix)])
            body.append(f"    result_{component} = tl.reshape({joined}, ({block}, {radix}))")
        body += [f"    digit = tl.arange(0, {radix})",
                 "    offset = (dst[:, None] + digit[None, :] * span) * 2",
                 "    tl.store(out_ptr + offset, result_r, mask[:, None])",
                 "    tl.store(out_ptr + offset + 1, result_i, mask[:, None])"]
        return name, "\n".join(body) + "\n"
    for digit in range(radix):
        body.extend(
            [
                f"    tl.store(out_ptr + (dst + {digit} * span) * 2, r{digit}, mask)",
                f"    tl.store(out_ptr + (dst + {digit} * span) * 2 + 1, i{digit}, mask)",
            ]
        )
    return name, "\n".join(body) + "\n"


def _build_vector_stage(n: int, radix: int, direction: str, dtype: str, stage_span: int, block: int):
    """Pair both input and output digits to bound the NPU scheduling DAG.

    For complex inputs, sum/difference pairs share their cosine/sine terms.
    Four half-width accumulators then produce both k and radix-k outputs.
    The direction-specific N-point roots preserve the inverse sign convention.
    """
    name = f"stockham_paired_vector_{direction}_n{n}_r{radix}_s{stage_span}_b{block}_{_dtype_suffix(dtype)}"
    half = radix // 2
    width = 1 << half.bit_length()
    span_line = f"    span = {stage_span}\n" if stage_span else ""
    twiddle_lines = ""
    if stage_span != 1:
        twiddle_lines = f"""        tw = digit * j * ({n} // ({radix} * span)) * 2
        wr = tl.load(twiddle_ptr + tw, mask, 0)
        wi = tl.load(twiddle_ptr + tw + 1, mask, 0)
        xr, xi = _cmul(xr, xi, wr, wi)
        tw_pair = ({radix} - digit) * j * ({n} // ({radix} * span)) * 2
        vr = tl.load(twiddle_ptr + tw_pair, mask, 0)
        vi = tl.load(twiddle_ptr + tw_pair + 1, mask, 0)
        yr, yi = _cmul(yr, yi, vr, vi)
"""
    source = f'''@triton.jit
def {name}(in_ptr, out_ptr, twiddle_ptr, span, nbatch):
{span_line}    index = tl.program_id(0).to(tl.int64) * {block} + tl.arange(0, {block})
    mask = index < nbatch.to(tl.int64) * {n // radix}
    batch = index // {n // radix}
    k = index % {n // radix}
    j = k % span
    output = tl.arange(0, {width})
    src0 = (batch * {n} + k) * 2
    x0r = tl.load(in_ptr + src0, mask, 0)
    x0i = tl.load(in_ptr + src0 + 1, mask, 0)
    cosine_r = tl.full(({width}, {block}), 0, tl.float32) + x0r[None, :]
    cosine_i = tl.full(({width}, {block}), 0, tl.float32) + x0i[None, :]
    sine_r = tl.full(({width}, {block}), 0, tl.float32)
    sine_i = tl.full(({width}, {block}), 0, tl.float32)
    for digit in tl.static_range(1, {half + 1}):
        src = (batch * {n} + k + digit * {n // radix}) * 2
        src_pair = (batch * {n} + k + ({radix} - digit) * {n // radix}) * 2
        xr = tl.load(in_ptr + src, mask, 0)
        xi = tl.load(in_ptr + src + 1, mask, 0)
        yr = tl.load(in_ptr + src_pair, mask, 0)
        yi = tl.load(in_ptr + src_pair + 1, mask, 0)
{twiddle_lines}        pr, pi = xr + yr, xi + yi
        mr, mi = xr - yr, xi - yi
        root = ((output * digit) % {radix}) * {n // radix} * 2
        cr = tl.load(twiddle_ptr + root)
        ci = tl.load(twiddle_ptr + root + 1)
        cosine_r = cosine_r + pr[None, :] * cr[:, None]
        cosine_i = cosine_i + pi[None, :] * cr[:, None]
        sine_r = sine_r + mr[None, :] * ci[:, None]
        sine_i = sine_i + mi[None, :] * ci[:, None]
    base = batch[None, :] * {n} + {radix} * k[None, :] - {radix - 1} * j[None, :]
    dst = base + output[:, None] * span
    valid = mask[None, :] & (output[:, None] <= {half})
    tl.store(out_ptr + dst * 2, cosine_r - sine_i, valid)
    tl.store(out_ptr + dst * 2 + 1, cosine_i + sine_r, valid)
    mirror = base + ({radix} - output[:, None]) * span
    mirror_valid = valid & (output[:, None] > 0)
    tl.store(out_ptr + mirror * 2, cosine_r + sine_i, mirror_valid)
    tl.store(out_ptr + mirror * 2 + 1, cosine_i - sine_r, mirror_valid)
'''
    return name, source
