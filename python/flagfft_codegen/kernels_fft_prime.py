"""Small prime codelets built from shared composite FFTs using Rader's identity."""

import cmath
import math


def rader_factors(radix):
    if radix not in {13, 17, 19, 23, 29, 31}:
        raise ValueError("unsupported generated Rader codelet")
    remaining, result = radix - 1, []
    for factor in (4, 2, 3, 5, 7, 11):
        while remaining % factor == 0:
            result.append(factor)
            remaining //= factor
    assert remaining == 1
    return tuple(result)


def _permutation(index, factors):
    source, stride = 0, math.prod(factors)
    for factor in factors:
        stride //= factor
        source += index % factor * stride
        index //= factor
    return source


def _constant(value):
    return repr(0.0 if abs(value) < 1e-14 else value)


def emit_rader_inner(radix):
    from .kernels_leaf import _emit_natural_order_codelet_call
    factors = rader_factors(radix)
    length = radix - 1
    name = f"_rader_fft_l{length}"
    arguments = [*(f"xr{i}" for i in range(length)), *(f"xi{i}" for i in range(length))]
    lines = ["@triton.jit", f"def {name}({', '.join(arguments)}):"]
    for index in range(length):
        source = _permutation(index, factors)
        lines += [f"    r{index}, i{index} = xr{source}, xi{source}"]
    previous = 1
    for factor in factors:
        span = previous * factor
        for base in range(0, length, span):
            for lane in range(previous):
                indices = [base + lane + digit * previous for digit in range(factor)]
                for digit, index in enumerate(indices[1:], 1):
                    phase = -2 * math.pi * digit * lane / span
                    if lane:
                        lines += [f"    r{index}, i{index} = _cmul(r{index}, i{index}, {_constant(math.cos(phase))}, {_constant(math.sin(phase))})"]
                lines += _emit_natural_order_codelet_call("    ", factor, "forward", indices)
        previous = span
    outputs = [*(f"r{i}" for i in range(length)), *(f"i{i}" for i in range(length))]
    lines += [f"    return {', '.join(outputs)}"]
    return "\n".join(lines) + "\n\n"


def emit_rader_call(radix, direction):
    length = radix - 1
    primitive = next(g for g in range(2, radix) if
        len({pow(g, k, radix) for k in range(length)}) == length)
    powers = [pow(primitive, k, radix) for k in range(length)]
    inverse_powers = [pow(primitive, (-k) % length, radix) for k in range(length)]
    roots = [cmath.exp(-2j * math.pi * k / radix) for k in powers]
    spectrum = [sum(value * cmath.exp(-2j * math.pi * j * k / length)
                    for j, value in enumerate(roots)) / length for k in range(length)]
    lines = []
    if direction == "inverse":
        lines += [f"    i{i} = -i{i}" for i in range(radix)]
    lines += ["    base_r, base_i = r0, i0"]
    inputs = [*(f"r{i}" for i in inverse_powers), *(f"i{i}" for i in inverse_powers)]
    fft_outputs = [*(f"ar{i}" for i in range(length)), *(f"ai{i}" for i in range(length))]
    lines += [f"    {', '.join(fft_outputs)} = _rader_fft_l{length}({', '.join(inputs)})",
        "    dc_r, dc_i = base_r + ar0, base_i + ai0"]
    for index, root in enumerate(spectrum):
        lines += [f"    ar{index}, ai{index} = _cmul(ar{index}, ai{index}, {_constant(root.real)}, {_constant(root.imag)})"]
    inputs = [*(f"ar{i}" for i in range(length)), *(f"-ai{i}" for i in range(length))]
    inverse_outputs = [*(f"cr{i}" for i in range(length)), *(f"ci{i}" for i in range(length))]
    lines += [f"    {', '.join(inverse_outputs)} = _rader_fft_l{length}({', '.join(inputs)})",
        "    r0, i0 = dc_r, dc_i"]
    for index, output in enumerate(powers):
        lines += [f"    r{output} = base_r + cr{index}",
            f"    i{output} = base_i - ci{index}"]
    if direction == "inverse":
        lines += [f"    i{i} = -i{i}" for i in range(radix)]
    return lines
