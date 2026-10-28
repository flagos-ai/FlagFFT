# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP32 matrix DFT axes with tiled vector exchanges for mixed four-step FFTs."""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source


def build_matrix_fourstep(a, b):
    if not 16 <= a <= 1024 or not 16 <= b <= 1024:
        raise ValueError('Matrix four-step axes require lengths 16..1024')
    body = ''
    for stage, (length, width) in enumerate(((a, b), (b, a))):
        p = 1 << (length - 1).bit_length()
        kin = 2 * p
        op = 2 * p
        body += f'@triton.jit\ndef pack{stage}(X,P,T):\n    batch=tl.program_id(2).to(tl.int64)\n    X=X+batch*{2 * a * b}\n    P=P+batch*{width * kin}\n    r=tl.program_id(0)*16+tl.arange(0,16)[:,None]\n    c=tl.program_id(1)*16+tl.arange(0,16)[None,:]\n    w=tl.program_id(1)*32+tl.arange(0,32)[None,:]\n    v=tl.load(X+r*{2 * width}+w,(r<{length})&(w<{2 * width}),0.)\n    vr,vi=tl.split(tl.reshape(v,(256,2)))\n    vr=tl.reshape(vr,(16,16));vi=tl.reshape(vi,(16,16))\n'
        if stage:
            body += f'    tr=tl.load(T+r*{width}+c,(r<{length})&(c<{width}),0.)\n    ti=tl.load(T+{length * width}+r*{width}+c,(r<{length})&(c<{width}),0.)\n    vr,vi=vr*tr-vi*ti,vr*ti+vi*tr\n'
        body += f'    rr=tl.program_id(1)*16+tl.arange(0,16)[:,None]\n    cc=tl.program_id(0)*16+tl.arange(0,16)[None,:]\n    tl.store(P+rr*{kin}+cc,tl.trans(vr),(rr<{width})&(cc<{p}))\n    tl.store(P+rr*{kin}+{p}+cc,tl.trans(vi),(rr<{width})&(cc<{p}))\n@triton.jit\ndef cube{stage}(P,W,Y):\n    batch=tl.program_id(2).to(tl.int64)\n    P=P+batch*{width * kin}\n    Y=Y+batch*{2 * a * b}\n    r=tl.program_id(0)*32+tl.arange(0,32)\n    j=tl.arange(0,{kin})\n    c=tl.program_id(1)*64+tl.arange(0,64)\n    x=tl.load(P+r[:,None]*{kin}+j[None,:],r[:,None]<{width},0.)\n    w=tl.load(W+j[:,None]*{op}+c[None,:])\n    y=tl.dot(x,w,input_precision="ieee")\n    tl.store(Y+r[:,None]*{2 * length}+c[None,:],y,(r[:,None]<{width})&(c[None,:]<{2 * length}))\n'
    body += f'@triton.jit\ndef transpose(X,Y):\n    batch=tl.program_id(2).to(tl.int64)\n    X=X+batch*{2 * a * b}\n    Y=Y+batch*{2 * a * b}\n    r=tl.program_id(0)*16+tl.arange(0,16)[:,None]\n    w=tl.program_id(1)*32+tl.arange(0,32)[None,:]\n    v=tl.load(X+r*{2 * b}+w,(r<{a})&(w<{2 * b}),0.)\n    v=tl.reshape(v,(16,16,2))\n    v=tl.reshape(tl.permute(v,1,0,2),(16,32))\n    rr=tl.program_id(1)*16+tl.arange(0,16)[:,None]\n    ww=tl.program_id(0)*32+tl.arange(0,32)[None,:]\n    tl.store(Y+rr*{2 * a}+ww,v,(rr<{b})&(ww<{2 * a}))\n'
    return _module_source(body)



def emit_matrix_fourstep(a,b,phase,direction,out_dir):
    names=('pack0','cube0','pack1','cube1','transpose')
    if phase not in names or direction not in ('forward','inverse'):
        raise ValueError('Invalid matrix four-step phase or direction')
    path=Path(out_dir)/f'npu_matrix_fourstep_{a}_{b}_{phase}_{direction}.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,build_matrix_fourstep(a,b))
    args=['X','Y'] if phase=='transpose' else ['X','P','T'] if phase.startswith('pack') else ['P','W','Y']
    meta=dict(module_path=str(path),kernel_name=phase,arg_names=args,
        signature=','.join(['*fp32:16']*len(args)),num_warps=4,num_stages=1,
        batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
