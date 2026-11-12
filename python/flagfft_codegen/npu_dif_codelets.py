# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP32 butterflies scoped to batch-grouped Ascend DIF kernels.

The thirteen-point butterfly reuses the cosine convolution's input sum for
DC. The fifteen-point butterfly applies the coprime axes in 5x3 order and
uses a private three-point helper that shares its pair sum with DC.
"""
from .npu_codelets import _radix13_source
from .npu_coprime import coprime_source

_RADIX3 = '''
@triton.jit
def _npu_dif_rad3(r0,r1,r2,i0,i1,i2):
    sr,si=r1+r2,i1+i2
    dr,di=r1-r2,i1-i2
    ar,ai=r0-0.5*sr,i0-0.5*si
    br,bi=0.8660254037844386*dr,0.8660254037844386*di
    return r0+sr,ar+bi,ar-bi,i0+si,ai-br,ai+br
'''


def dif_codelet_source(radices):
    parts=[]
    if 13 in radices:
        parts.append(_radix13_source(reuse_dc=True))
    if 15 in radices:
        parts.extend((_RADIX3,coprime_source(5,3).replace('_fwd_rad3_b1','_npu_dif_rad3')))
    return '\n'.join(parts)
