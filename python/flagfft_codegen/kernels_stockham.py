"""Global-memory Stockham mapping of the shared register radix codelets."""

from .kernels_common import _NATURAL_ORDER_CODELET_RADICES, _dtype_suffix
from .kernels_leaf import _emit_natural_order_codelet_call


def build_stockham_stage(n: int, radix: int, direction: str, dtype: str, stage_span: int = 0):
    if radix not in _NATURAL_ORDER_CODELET_RADICES or n % radix:
        raise ValueError(f"unsupported Stockham stage n={n}, radix={radix}")
    if stage_span < 0 or (stage_span and n % (radix * stage_span)):
        raise ValueError(f"invalid Stockham span {stage_span} for n={n}, radix={radix}")
    if radix in (13, 17, 19):
        return _build_vector_stage(n, radix, direction, dtype, stage_span)
    name = f"stockham_{direction}_n{n}_r{radix}_s{stage_span}_{_dtype_suffix(dtype)}"
    body = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, span, nbatch):",
        *([f"    span = {stage_span}"] if stage_span else []),
        "    index = tl.program_id(0).to(tl.int64) * 128 + tl.arange(0, 128)",
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
    body.extend(_emit_natural_order_codelet_call("    ", radix, direction))
    body.append(f"    dst = batch * {n} + {radix} * k - {radix - 1} * j")
    for digit in range(radix):
        body.extend(
            [
                f"    tl.store(out_ptr + (dst + {digit} * span) * 2, r{digit}, mask)",
                f"    tl.store(out_ptr + (dst + {digit} * span) * 2 + 1, i{digit}, mask)",
            ]
        )
    return name, "\n".join(body) + "\n"


def _build_vector_stage(n: int, radix: int, direction: str, dtype: str, stage_span: int):
    """Vectorize output digits while keeping a small compiler scheduling DAG.

    Reuse the direction-specific N-point table for both the stage twiddle
    and radix roots, vectorizing each input across all output digits.
    """
    name = f"stockham_vector_{direction}_n{n}_r{radix}_s{stage_span}_{_dtype_suffix(dtype)}"
    width = 1 << (radix - 1).bit_length()
    source = f'''@triton.jit
def {name}(in_ptr, out_ptr, twiddle_ptr, span, nbatch):
    index = tl.program_id(0).to(tl.int64) * 128 + tl.arange(0, 128)
    mask = index < nbatch.to(tl.int64) * {n // radix}
    batch = index // {n // radix}
    k = index % {n // radix}
    j = k % span
    output = tl.arange(0, {width})
    real = tl.full(({width}, 128), 0, tl.float32)
    imag = tl.full(({width}, 128), 0, tl.float32)
    for digit in tl.static_range({radix}):
        src = (batch * {n} + k + digit * {n // radix}) * 2
        xr = tl.load(in_ptr + src, mask, 0)
        xi = tl.load(in_ptr + src + 1, mask, 0)
        tw = digit * j * ({n} // ({radix} * span)) * 2
        wr = tl.load(twiddle_ptr + tw, mask, 0)
        wi = tl.load(twiddle_ptr + tw + 1, mask, 0)
        xr, xi = _cmul(xr, xi, wr, wi)
        root = ((output * digit) % {radix}) * {n // radix} * 2
        cr = tl.load(twiddle_ptr + root)
        ci = tl.load(twiddle_ptr + root + 1)
        real = real + xr[None, :] * cr[:, None] - xi[None, :] * ci[:, None]
        imag = imag + xi[None, :] * cr[:, None] + xr[None, :] * ci[:, None]
    dst = batch[None, :] * {n} + {radix} * k[None, :] - {radix - 1} * j[None, :] + output[:, None] * span
    valid = mask[None, :] & (output[:, None] < {radix})
    tl.store(out_ptr + dst * 2, real, valid)
    tl.store(out_ptr + dst * 2 + 1, imag, valid)
'''
    if stage_span:
        source = source.replace("    index =", f"    span = {stage_span}\n    index =", 1)
    if stage_span == 1:
        start = source.index("        tw =")
        end = source.index("        root =")
        source = source[:start] + source[end:]
    return name, source
