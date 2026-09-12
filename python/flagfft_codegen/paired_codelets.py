# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared cosine/sine accumulators for bounded odd-radix FP64 codelets."""

import math


def paired_codelet_source(radix: int) -> str:
    """Emit a forward DFT, sharing work between outputs k and radix-k.

    With P[j]=x[j]+x[n-j], M[j]=x[j]-x[n-j], compute
    C[k]=x[0]+sum(cos(2*pi*k*j/n)*P[j]) and
    S[k]=sum(sin(2*pi*k*j/n)*M[j]). Then X[k]=C[k]-i*S[k]
    and X[n-k]=C[k]+i*S[k]. Inverse callers retain their existing swap.
    Finishing one output pair at a time also shortens accumulator lifetimes.
    """
    if radix not in (11, 13, 17, 19):
        raise ValueError(f"unvalidated paired radix: {radix}")
    half = (radix + 1) // 2
    args = ", ".join(f"{c}{j}" for c in "ri" for j in range(radix))
    lines = ["@triton.jit", f"def _fwd_rad{radix}_b1({args}):"]
    for j in range(1, half):
        for c in "ri":
            lines.extend(
                (
                    f"    p{j}{c} = {c}{j} + {c}{radix-j}",
                    f"    m{j}{c} = {c}{j} - {c}{radix-j}",
                )
            )
    for c in "ri":
        lines.append(
            f"    y0{c} = {c}0 + " + " + ".join(f"p{j}{c}" for j in range(1, half))
        )
    for k in range(1, half):
        for c in "ri":
            lines.append(f"    c{k}{c} = {c}0")
            for j in range(1, half):
                angle = 2 * math.pi * ((k * j) % radix) / radix
                lines.append(f"    c{k}{c} = c{k}{c} + {math.cos(angle)!r} * p{j}{c}")
                previous = f"s{k}{c} + " if j > 1 else ""
                lines.append(f"    s{k}{c} = {previous}{math.sin(angle)!r} * m{j}{c}")
        lines.extend(
            (
                f"    y{k}r = c{k}r + s{k}i",
                f"    y{k}i = c{k}i - s{k}r",
                f"    y{radix-k}r = c{k}r - s{k}i",
                f"    y{radix-k}i = c{k}i + s{k}r",
            )
        )
    lines.append(
        "    return " + ", ".join(f"y{j}{c}" for c in "ri" for j in range(radix))
    )
    return "\n".join(lines) + "\n"
