# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A 126-point natural prefix with bounded small-radix Vector suffixes."""
import math
from .metadata import _module_source
from .kernels_common import codelet_radices_for
from .kernels_npu_dif_cube import _batch_grouped_late_stage
from .kernels_npu_dif_aligned import _fused_finish
from .npu_coarse_dense import prefix_source
from .npu_radix9 import radix9_source


def build_natural_dif(factors,direction='forward',global_inverse=False):
    if (tuple(factors[:2])!=(7,18) or len(factors)<3 or
            any(r not in (2,3,4,5,6,7,8,9,10) for r in factors[2:])):
        raise ValueError('Natural prefix requires 7 x 18 and small suffix radices')
    if direction not in ('forward','inverse') or (global_inverse and direction!='inverse'):
        raise ValueError('Invalid natural-prefix direction')
    n=math.prod(factors);remaining=n//126;pn=remaining*128
    chunks=[prefix_source(factors,global_inverse,126)]
    specs=[dict(name='coarse_dense_prefix',phase='early',stage=0,r=126,m=remaining,
        prefix=1,physical_n=pn,grid=(math.ceil(remaining/64),),auto=True)]
    prefix,previous=126,7
    mode='forward' if global_inverse else direction
    for stage,r in enumerate(factors[2:],2):
        m=n//(prefix*r);pp=prefix//126*128
        spec=dict(name=f'natural_stage_{stage}',phase='late',stage=stage,r=r,m=m,
            prefix=prefix,previous_prefix=previous,physical_prefix=pp,initial_prefix=126,
            padded_prefix=128,physical_n=pn,deferred_twiddle=stage==2,auto=True)
        if stage+1==len(factors):
            body,spec=_fused_finish(spec,mode,conjugate_output=global_inverse)
        else:
            body=_batch_grouped_late_stage(spec,mode,4,128)
        chunks.append(body);specs.append(spec);previous=prefix;prefix*=r
    source=_module_source('\n'.join(chunks),tuple(sorted(codelet_radices_for(factors[2:]))))
    return source+'\n'+radix9_source(),specs
