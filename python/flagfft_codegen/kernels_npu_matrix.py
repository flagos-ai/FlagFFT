# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Short batched transforms as one FP32 real matrix multiplication.

Complex inputs are embedded in two real blocks. Matrix output columns already
use the public interleaved/compact layout, avoiding vector operations in CBUF.
"""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source


def matrix_layout(n, mode):
    if n < 16 or n > 256 or mode not in (0,1,2):
        raise ValueError('Matrix FFT requires length 16..256 and a complex/real boundary')
    p=1<<(n-1).bit_length();half=n//2+1
    input_words=2*n if mode==0 else n if mode==1 else 2*half
    reduction=2*p if mode==0 else 1<<(input_words-1).bit_length()
    output_words=2*n if mode==0 else 2*half if mode==1 else n
    columns=1<<(output_words-1).bit_length()
    return p,input_words,reduction,output_words,columns,32 if mode==0 else 16,64


def build_matrix_source(n, mode, direction):
    p,ins,kin,outs,op,bm,bn=matrix_layout(n,mode)
    if direction not in ('forward','inverse') or (mode==1 and direction!='forward') or (mode==2 and direction!='inverse'):
        raise ValueError('Matrix FFT direction disagrees with its real boundary')
    prefix=f'npu_matrix_n{n}_m{mode}_{direction}'
    pack_name=prefix+'_pack';cube_name=prefix+'_cube';group=2
    body=f'''@triton.jit
def {pack_name}(X,P,input_distance,NB):
    b=tl.program_id(0).to(tl.int64)*{group}+tl.arange(0,{group})[:,None]
    k=tl.arange(0,{kin})[None,:]
    x=tl.load(X+b*input_distance+k,(b<NB)&(k<{ins}),0.)
'''
    if mode==0:
        body+=f'''    r,i=tl.split(tl.reshape(x,({group*p},2)))
    r=tl.reshape(r,({group},{p}));i=tl.reshape(i,({group},{p}))
    j=tl.arange(0,{p})[None,:]
    tl.store(P+b*{kin}+j,r,b<NB)
    tl.store(P+b*{kin}+{p}+j,i,b<NB)
'''
    else:body+=f'    tl.store(P+b*{kin}+k,x,b<NB)\n'
    body+=f'''@triton.jit
def {cube_name}(P,W,Y,input_distance,output_distance,NB):
    b=tl.program_id(0).to(tl.int64)*{bm}+tl.arange(0,{bm})
    j=tl.arange(0,{kin})
    k=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    x=tl.load(P+b[:,None]*input_distance+j[None,:],(b[:,None]<NB)&(j[None,:]<{kin if mode==0 else ins}),0.)
    w=tl.load(W+j[:,None]*{op}+k[None,:])
    y=tl.dot(x,w,input_precision="ieee")
    tl.store(Y+b[:,None]*output_distance+k[None,:],y,(b[:,None]<NB)&(k[None,:]<{outs}))
'''
    return pack_name,cube_name,_module_source(body)


def emit_matrix(n, mode, direction, phase, out_dir):
    if phase not in ('pack','cube'):raise ValueError('Invalid matrix FFT phase')
    pack,cube,source=build_matrix_source(n,mode,direction)
    name=pack if phase=='pack' else cube
    path=Path(out_dir)/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,source)
    args=['X','P','input_distance','NB'] if phase=='pack' else ['P','W','Y','input_distance','output_distance','NB']
    meta=dict(module_path=str(path),kernel_name=name,arg_names=args,
        signature='*fp32:16,*fp32:16,i64,i32' if phase=='pack' else '*fp32:16,*fp32:16,*fp32:16,i64,i64,i32',
        num_warps=4,num_stages=1,batch_per_block=2 if phase=='pack' else matrix_layout(n,mode)[5],
        kernel_type='npu_grouped_fft64',dtype='complex64',direction=direction,length=n)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
