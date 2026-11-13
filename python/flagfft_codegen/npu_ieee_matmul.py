# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bound FP32 DFT reduction depth without changing matrix layouts or precision."""


def ieee_matmul(reduction, rows, columns, lhs, rhs, *, index="k", result="y"):
    """Emit independent IEEE dot partials and a compensated FP32 sum.

    Long complex embeddings otherwise accumulate up to twice the FFT length
    in one dot. Keeping partials within 256 terms reduces rounding growth and
    the operand working set. TwoSum also gives each partial multiple uses,
    so dot/add combining cannot turn the partials back into one long reduction.
    """
    if reduction <= 256:
        return (f"    {index}=tl.arange(0,{reduction})\n"
                f"    _lhs=tl.load({lhs})\n"
                f"    _rhs=tl.load({rhs})\n"
                f'    {result}=tl.dot(_lhs,_rhs,input_precision="ieee")\n')
    code = ""
    for offset in range(0, reduction, 256):
        code += (f"    {index}=tl.arange(0,256)+{offset}\n"
                 f"    _lhs=tl.load({lhs})\n"
                 f"    _rhs=tl.load({rhs})\n"
                 '    _partial=tl.dot(_lhs,_rhs,input_precision="ieee")\n')
        if offset == 0:
            code += (f"    {result}=_partial\n"
                     f"    _error=tl.full(({rows},{columns}),0.,tl.float32)\n")
        else:
            code += (f"    _rounded={result}+_partial\n"
                     f"    _virtual=_rounded-{result}\n"
                     f"    _error=_error+(({result}-(_rounded-_virtual))+(_partial-_virtual))\n"
                     f"    {result}=_rounded\n")
    return code + f"    {result}={result}+_error\n"
