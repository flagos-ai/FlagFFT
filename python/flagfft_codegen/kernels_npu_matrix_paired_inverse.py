# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hermitian row pairs reduce both odd inverse matrix axes to half work."""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source

def build_paired_inverse(a,b):
    if not (16<=a<=256 and 16<=b<=256 and a%2==b%2==1):
        raise ValueError('Paired inverse requires odd matrix axes within 16..256')
    first_rows,last_rows,last_columns,pack_tile=128,64,64,64
    pa=1<<(a-1).bit_length();h=b//2+1;ph=1<<(h-1).bit_length();pb=1<<(b-1).bit_length();n=a*b;half=n//2+1;t=pack_tile;bm=first_rows;bn=128;lm=last_rows;ln=last_columns
    body=f'''
@triton.jit
def pack_inverse(X,P,ID):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*{t}+tl.arange(0,{t})[:,None]
    col=tl.arange(0,{t})[None,:]
    word=tl.arange(0,{2*t})[None,:]
    base=tl.program_id(1)*{t}
    forward=r*{2*b}+base*2+word
    backward=({n}-r*{b}-base-{t-1})*2+word
    ar=tl.load(X+batch*ID*2+forward,(r<={a//2})&(forward<{2*half}),0.)
    br=tl.load(X+batch*ID*2+backward,(r>{a//2})&(r<{a})&(backward>=0)&(backward<{2*half}),0.)
    take=tl.broadcast_to(({2*t-2}-2*(word//2)+word%2),({t},{2*t}))
    br=tl.gather(br,take,1)
    vr,vi=tl.split(tl.reshape(ar+br,({t*t},2)))
    vr=tl.reshape(vr,({t},{t}));vi=tl.reshape(vi,({t},{t}))
    vi=vi*tl.where(r<={a//2},1.,-1.)
    vi=tl.where((r==0)&(base+col==0),0.,vi)
    rr=base+tl.arange(0,{t})[:,None]
    cc=tl.program_id(0)*{t}+tl.arange(0,{t})[None,:]
    tl.store(P+batch*{h*2*pa}+rr*{2*pa}+cc,tl.trans(vr),(rr<{h})&(cc<{pa}))
    tl.store(P+batch*{h*2*pa}+rr*{2*pa}+{pa}+cc,tl.trans(vi),(rr<{h})&(cc<{pa}))

@triton.jit
def cube0_inverse_half_pack(P,W,T,Q):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*{bm}+tl.arange(0,{bm})
    k=tl.arange(0,{2*pa})
    word=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    x=tl.load(P+batch*{h*2*pa}+row[:,None]*{2*pa}+k[None,:],row[:,None]<{h},0.)
    w=tl.load(W+k[:,None]*{2*pa}+word[None,:],word[None,:]<{2*pa},0.)
    value=tl.dot(x,w,input_precision="ieee")
    vr,vi=tl.split(tl.reshape(value,({bm*bn//2},2)))
    vr=tl.reshape(vr,({bm},{bn//2}));vi=tl.reshape(vi,({bm},{bn//2}))
    freq=tl.program_id(1)*{bn//2}+tl.arange(0,{bn//2})[None,:]
    sample=row[:,None]
    valid=(sample<{h})&(freq<{a})
    co=tl.load(T+sample*{a}+freq,valid,0.)
    si=tl.load(T+{h*a}+sample*{a}+freq,valid,0.)
    re=vr*co-vi*si
    im=vr*si+vi*co
    rr=tl.program_id(1)*{bn//2}+tl.arange(0,{bn//2})[:,None]
    cc=row[None,:]
    tl.store(Q+batch*{a*2*ph}+rr*{2*ph}+cc,tl.trans(re),(rr<{a})&(cc<{ph}))
    tl.store(Q+batch*{a*2*ph}+rr*{2*ph}+{ph}+cc,tl.trans(im),(rr<{a})&(cc<{ph}))

@triton.jit
def cube1_inverse_half_real(Q,W,Y,OD):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*{lm}+tl.arange(0,{lm})
    k=tl.arange(0,{2*ph})
    col=tl.program_id(1)*{ln}+tl.arange(0,{ln})
    x=tl.load(Q+batch*{a*2*ph}+row[:,None]*{2*ph}+k[None,:],row[:,None]<{a},0.)
    w=tl.load(W+k[:,None]*{pb}+col[None,:],col[None,:]<{pb},0.)
    value=tl.dot(x,w,input_precision="ieee")
    tl.store(Y+batch*OD+col[:,None]*{a}+row[None,:],tl.trans(value),(col[:,None]<{b})&(row[None,:]<{a}))
'''
    return _module_source(body)


def emit_paired_inverse(a,b,phase,direction,out_dir):
    names=('pack_inverse','cube0_inverse_half_pack','cube1_inverse_half_real')
    if phase not in names or direction!='inverse':
        raise ValueError('Paired matrix phases require inverse direction')
    args=({'pack_inverse':['X','P','ID'],
           'cube0_inverse_half_pack':['P','W','T','Q'],
           'cube1_inverse_half_real':['Q','W','Y','OD']})[phase]
    signature=','.join('*fp32:16' if x not in ('ID','OD') else 'i64' for x in args)
    path=Path(out_dir)/f'npu_matrix_paired_{a}_{b}_{phase}.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,build_paired_inverse(a,b))
    meta=dict(module_path=str(path),kernel_name=phase,arg_names=args,signature=signature,
        num_warps=4,num_stages=1,batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,
        kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
