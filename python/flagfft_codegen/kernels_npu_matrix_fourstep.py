# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP32 matrix DFT axes with tiled vector exchanges for mixed four-step FFTs."""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source


def build_matrix_fourstep(a, b, wide=False):
    if not 16 <= a <= 1024 or not 16 <= b <= 1024:
        raise ValueError('Matrix four-step axes require lengths 16..1024')
    tile=32 if wide else 16
    bm,bn=(64,128) if wide else (32,64)
    body = ''
    for stage, (length, width) in enumerate(((a, b), (b, a))):
        p = 1 << (length - 1).bit_length()
        kin = 2 * p
        op = 2 * p
        body += f'''@triton.jit
def pack{stage}(X,P,T):
    batch=tl.program_id(2).to(tl.int64)
    X=X+batch*{2 * a * b}
    P=P+batch*{width * kin}
    r=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    c=tl.program_id(1)*{tile}+tl.arange(0,{tile})[None,:]
    w=tl.program_id(1)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    v=tl.load(X+r*{2 * width}+w,(r<{length})&(w<{2 * width}),0.)
    vr,vi=tl.split(tl.reshape(v,({tile*tile},2)))
    vr=tl.reshape(vr,({tile},{tile}));vi=tl.reshape(vi,({tile},{tile}))
'''
        if stage:
            body += f'''    tr=tl.load(T+r*{width}+c,(r<{length})&(c<{width}),0.)
    ti=tl.load(T+{length * width}+r*{width}+c,(r<{length})&(c<{width}),0.)
    vr,vi=vr*tr-vi*ti,vr*ti+vi*tr
'''
        body += f'''    rr=tl.program_id(1)*{tile}+tl.arange(0,{tile})[:,None]
    cc=tl.program_id(0)*{tile}+tl.arange(0,{tile})[None,:]
    tl.store(P+rr*{kin}+cc,tl.trans(vr),(rr<{width})&(cc<{p}))
    tl.store(P+rr*{kin}+{p}+cc,tl.trans(vi),(rr<{width})&(cc<{p}))
'''
        if wide:
            body+=f'''@triton.jit
def cube{stage}(P,W,Y,NB):
    r=tl.program_id(0)*{bm}+tl.arange(0,{bm})
    j=tl.arange(0,{kin})
    c=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    x=tl.load(P+r[:,None]*{kin}+j[None,:],r[:,None]<NB*{width},0.)
    w=tl.load(W+j[:,None]*{op}+c[None,:],c[None,:]<{op},0.)
    y=tl.dot(x,w,input_precision="ieee")
    tl.store(Y+r[:,None]*{2*length}+c[None,:],y,(r[:,None]<NB*{width})&(c[None,:]<{2*length}))
'''
        else:
            body+=f'''@triton.jit
def cube{stage}(P,W,Y):
    batch=tl.program_id(2).to(tl.int64)
    P=P+batch*{width * kin}
    Y=Y+batch*{2 * a * b}
    r=tl.program_id(0)*32+tl.arange(0,32)
    j=tl.arange(0,{kin})
    c=tl.program_id(1)*64+tl.arange(0,64)
    x=tl.load(P+r[:,None]*{kin}+j[None,:],r[:,None]<{width},0.)
    w=tl.load(W+j[:,None]*{op}+c[None,:],c[None,:]<{op},0.)
    y=tl.dot(x,w,input_precision="ieee")
    tl.store(Y+r[:,None]*{2 * length}+c[None,:],y,(r[:,None]<{width})&(c[None,:]<{2 * length}))
'''
    body += f'''@triton.jit
def transpose(X,Y):
    batch=tl.program_id(2).to(tl.int64)
    X=X+batch*{2 * a * b}
    Y=Y+batch*{2 * a * b}
    r=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    w=tl.program_id(1)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    v=tl.load(X+r*{2 * b}+w,(r<{a})&(w<{2 * b}),0.)
    v=tl.reshape(v,({tile},{tile},2))
    v=tl.reshape(tl.permute(v,1,0,2),({tile},{2*tile}))
    rr=tl.program_id(1)*{tile}+tl.arange(0,{tile})[:,None]
    ww=tl.program_id(0)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    tl.store(Y+rr*{2 * a}+ww,v,(rr<{b})&(ww<{2 * a}))
'''
    return _module_source(body)



def emit_matrix_fourstep(a,b,phase,direction,out_dir,wide=False):
    names=('pack0','cube0','pack1','cube1','transpose')
    if phase not in names or direction not in ('forward','inverse'):
        raise ValueError('Invalid matrix four-step phase or direction')
    path=Path(out_dir)/(f'npu_matrix_fourstep_{a}_{b}_{phase}_{direction}'+('_wide' if wide else '')+'.py')
    path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,build_matrix_fourstep(a,b,wide=wide))
    args=['X','Y'] if phase=='transpose' else ['X','P','T'] if phase.startswith('pack') else ['P','W','Y']
    signature=','.join(['*fp32:16']*len(args))
    if wide and phase.startswith('cube'):
        args.append('NB');signature+=',i32'
    meta=dict(module_path=str(path),kernel_name=phase,arg_names=args,
        signature=signature,num_warps=4,num_stages=1,
        batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
