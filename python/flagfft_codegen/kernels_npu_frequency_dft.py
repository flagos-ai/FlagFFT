# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Frequency-parallel short DFTs for batches that underfill FFT leaf cores."""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source

def emit_frequency_dft(n, direction, out_dir):
    if not 64 <= n <= 256 or direction not in ('forward','inverse'):
        raise ValueError('Frequency DFT requires length 64..256 and a valid direction')
    p=1<<(n-1).bit_length();tile=8
    name=f'npu_frequency_dft_n{n}_{direction}'
    body=f'''@triton.jit
def {name}(X,W,Y):
    b=tl.program_id(1).to(tl.int64)
    word=tl.arange(0,{2*p})
    raw=tl.load(X+b*{2*n}+word,word<{2*n},0.)
    re,im=tl.split(tl.reshape(raw,({p},2)))
    j=tl.arange(0,{p})[None,:]
    k=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    wr=tl.load(W+k*{p}+j,(k<{n})&(j<{n}),0.)
    wi=tl.load(W+{n*p}+k*{p}+j,(k<{n})&(j<{n}),0.)
    r=tl.sum(wr*re[None,:]-wi*im[None,:],1)
    i=tl.sum(wr*im[None,:]+wi*re[None,:],1)
    out=tl.reshape(tl.join(r,i),({2*tile},))
    word=tl.program_id(0)*{2*tile}+tl.arange(0,{2*tile})
    tl.store(Y+b*{2*n}+word,out,word<{2*n})
'''
    path=Path(out_dir)/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,_module_source(body))
    meta=dict(module_path=str(path),kernel_name=name,signature='*fp32:16,*fp32:16,*fp32:16',
        arg_names=['X','W','Y'],num_warps=4,num_stages=1,batch_per_block=1,
        kernel_type='leaf',length=n,dtype='complex64',direction=direction)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
