# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep a bounded two-axis prefix in UB and write aligned late-stage rows."""
import math
from .metadata import _module_source
from .kernels_common import codelet_radices_for
from .kernels_leaf import _emit_natural_order_codelet_call
from .kernels_npu_dif_cube import _batch_grouped_late_stage,_grouped_late_stage
from .kernels_npu_dif_aligned import _fused_finish
from .npu_coprime import coprime_source
from .npu_radix9 import radix9_source


def build_coarse_dif(factors,direction='forward',global_inverse=False,single_batch=False):
    if global_inverse and direction!='inverse':raise ValueError('Global inverse requires inverse DIF')
    mode='forward' if global_inverse else direction
    factors=tuple(factors)
    if factors[:2] not in ((4,15),(8,15)) or any(r not in (2,3,4,5,7,8,9) for r in factors[2:]):
        raise ValueError('Coarse DIF requires a 4/8 x 15 prefix and small suffix codelets')
    if direction not in ('forward','inverse') or len(factors)<3:
        raise ValueError('Invalid coarse DIF plan')
    a,b=factors[:2];p0=a*b;pad=a*16;c=128 if a==4 else 64
    n=math.prod(factors);rest=n//p0;pn=rest*pad;ac=a*c;bc=16*c;state=pad*c
    ls=['@triton.jit','def coarse_prefix(X,TW,Y):',
        f'    base=tl.program_id(0)*{c}',
        '    batch=tl.program_id(1).to(tl.int64)',
        f'    word=base*2+tl.arange(0,{2*c})[None,:]',
        '    t=tl.arange(0,16)[:,None]']
    for d in range(a):
        ls += [f'    raw=tl.load(X+batch*{2*n}+(t+{d*b})*{2*rest}+word,(t<15)&(word<{2*rest}),0.)',
               f'    r{d},i{d}=tl.split(tl.reshape(raw,({bc},2)))']
        if direction=='inverse':ls += [f'    i{d}=-i{d}']
    ls += _emit_natural_order_codelet_call('    ',a,'forward')
    for d in range(a):
        ls += [f'    wr=tl.load(TW+{d*16}+tl.arange(0,16))[:,None]',
               f'    wi=tl.load(TW+{a*16+d*16}+tl.arange(0,16))[:,None]',
               f'    vr=tl.reshape(r{d},(16,{c}));vi=tl.reshape(i{d},(16,{c}))',
               f'    r{d}=tl.reshape(vr*wr-vi*wi,({bc},))',
               f'    i{d}=tl.reshape(vr*wi+vi*wr,({bc},))']
    ls += [f'    buf_r=tl.zeros(({state},),tl.float32)',f'    buf_i=tl.zeros(({state},),tl.float32)']
    for d in range(a):
        ls += [f'    buf_r=al.insert_slice(buf_r,r{d},[{d*bc}],[{bc}],[1])',
               f'    buf_i=al.insert_slice(buf_i,i{d},[{d*bc}],[{bc}],[1])']
    for d in range(b):
        ls += [f'    r{d}=tl.reshape(al.extract_slice(tl.reshape(buf_r,({a},{bc})),[0,{d*c}],[{a},{c}],[1,1]),({ac},))',
               f'    i{d}=tl.reshape(al.extract_slice(tl.reshape(buf_i,({a},{bc})),[0,{d*c}],[{a},{c}],[1,1]),({ac},))']
    ls += _emit_natural_order_codelet_call('    ',b,'forward')
    ls += [f'    re=tl.zeros(({state},),tl.float32)',f'    im=tl.zeros(({state},),tl.float32)']
    for d in range(b):
        ls += [f'    re=al.insert_slice(re,r{d},[{d*ac}],[{ac}],[1])',
               f'    im=al.insert_slice(im,i{d},[{d*ac}],[{ac}],[1])']
    if direction=='inverse' and not global_inverse:ls += ['    im=-im']
    ls += [f'    re=tl.trans(tl.reshape(re,({pad},{c})))',
           f'    im=tl.trans(tl.reshape(im,({pad},{c})))',
           f'    sample=base+tl.arange(0,{c})[:,None]',
           f'    freq=tl.arange(0,{pad})[None,:]',
           f'    out=sample*{pad}+freq',
           f'    tl.store(Y+batch*{2*pn}+out,re,sample<{rest})',
           f'    tl.store(Y+batch*{2*pn}+{pn}+out,im,sample<{rest})']
    chunks=['\n'.join(ls)+'\n'];specs=[dict(name='coarse_prefix',phase='early',stage=0,
        r=p0,m=rest,prefix=1,physical_n=pn,grid=(math.ceil(rest/c),),auto=True,local_prefix=factors[:2])]
    prefix=p0;previous_prefix=a
    for i,r in enumerate(factors[2:],2):
        m=n//(prefix*r);pp=prefix//p0*pad
        spec=dict(name=f'coarse_stage_{i}',phase='late',stage=i,r=r,m=m,prefix=prefix,
            previous_prefix=previous_prefix,physical_prefix=pp,initial_prefix=p0,
            padded_prefix=pad,physical_n=pn,deferred_twiddle=i==2,auto=True)
        if i+1==len(factors):body,spec=_fused_finish(spec,mode,conjugate_output=global_inverse)
        elif single_batch:
            columns=min(512,1<<(pp-1).bit_length())
            spec['grid']=(m*math.ceil(pp/columns),)
            body=_grouped_late_stage(spec,mode,columns)
        else:body=_batch_grouped_late_stage(spec,mode,4,128)
        chunks.append(body);specs.append(spec);previous_prefix=prefix;prefix*=r
    source=_module_source('\n'.join(chunks),tuple(sorted(codelet_radices_for(factors))))
    source+='\nimport triton.language.extra.cann.extension as al\n'+coprime_source(3,5)+'\n'+radix9_source()
    return source,specs
