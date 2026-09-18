"""Global-memory Stockham mapping of the shared register radix codelets."""

from .kernels_common import _NATURAL_ORDER_CODELET_RADICES, _dtype_suffix
from .kernels_leaf import _emit_natural_order_codelet_call


def build_stockham_stage(n: int, radix: int, direction: str, dtype: str):
    if radix not in _NATURAL_ORDER_CODELET_RADICES or n % radix:
        raise ValueError(f"unsupported Stockham stage n={n}, radix={radix}")
    name = f"stockham_{direction}_n{n}_r{radix}_{_dtype_suffix(dtype)}"
    body = [
        "@triton.jit",
        f"def {name}(in_ptr, out_ptr, twiddle_ptr, span, nbatch):",
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
        if digit:
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
