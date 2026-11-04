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

@triton.jit
def transpose_real(X,Y,D):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    w=tl.program_id(1)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    raw=tl.load(X+batch*{2*a*b}+r*{2*b}+w,(r<{a})&(w<{2*b}),0.)
    real,imag=tl.split(tl.reshape(raw,({tile*tile},2)))
    real=tl.trans(tl.reshape(real,({tile},{tile})))
    rr=tl.program_id(1)*{tile}+tl.arange(0,{tile})[:,None]
    cc=tl.program_id(0)*{tile}+tl.arange(0,{tile})[None,:]
    tl.store(Y+batch*D+rr*{a}+cc,real,(rr<{b})&(cc<{a}))
'''
    # Real input needs only the real half of the first matrix's K dimension.
    p = 1 << (a - 1).bit_length()
    body += f'''@triton.jit
def pack0_real(X,P,T):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    c=tl.program_id(1)*{tile}+tl.arange(0,{tile})[None,:]
    value=tl.load(X+batch*{a*b}+r*{b}+c,(r<{a})&(c<{b}),0.)
    rr=tl.program_id(1)*{tile}+tl.arange(0,{tile})[:,None]
    cc=tl.program_id(0)*{tile}+tl.arange(0,{tile})[None,:]
    tl.store(P+batch*{b*p}+rr*{p}+cc,tl.trans(value),(rr<{b})&(cc<{p}))
'''
    if wide:
        body += f'''@triton.jit
def cube0_real(P,W,Y,NB):
    r=tl.program_id(0)*{bm}+tl.arange(0,{bm})
    k=tl.arange(0,{p})
    c=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    value=tl.load(P+r[:,None]*{p}+k[None,:],r[:,None]<NB*{b},0.)
    weights=tl.load(W+k[:,None]*{2*p}+c[None,:],c[None,:]<{2*p},0.)
    result=tl.dot(value,weights,input_precision="ieee")
    tl.store(Y+r[:,None]*{2*a}+c[None,:],result,(r[:,None]<NB*{b})&(c[None,:]<{2*a}))
'''
    else:
        body += f'''@triton.jit
def cube0_real(P,W,Y):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*32+tl.arange(0,32)
    k=tl.arange(0,{p})
    c=tl.program_id(1)*64+tl.arange(0,64)
    value=tl.load(P+batch*{b*p}+r[:,None]*{p}+k[None,:],r[:,None]<{b},0.)
    weights=tl.load(W+k[:,None]*{2*p}+c[None,:],c[None,:]<{2*p},0.)
    result=tl.dot(value,weights,input_precision="ieee")
    tl.store(Y+batch*{2*a*b}+r[:,None]*{2*a}+c[None,:],result,(r[:,None]<{b})&(c[None,:]<{2*a}))
'''
    bp=1 << (b - 1).bit_length()
    compact_columns=2*(b//2+1)
    if wide:
        body += f'''@triton.jit
def cube1_half(P,W,Y,NB):
    r=tl.program_id(0)*{bm}+tl.arange(0,{bm})
    k=tl.arange(0,{2*bp})
    c=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    value=tl.load(P+r[:,None]*{2*bp}+k[None,:],r[:,None]<NB*{a},0.)
    weights=tl.load(W+k[:,None]*{2*bp}+c[None,:],c[None,:]<{2*bp},0.)
    result=tl.dot(value,weights,input_precision="ieee")
    tl.store(Y+r[:,None]*{2*b}+c[None,:],result,(r[:,None]<NB*{a})&(c[None,:]<{compact_columns}))
'''
    else:
        body += f'''@triton.jit
def cube1_half(P,W,Y):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*32+tl.arange(0,32)
    k=tl.arange(0,{2*bp})
    c=tl.program_id(1)*64+tl.arange(0,64)
    value=tl.load(P+batch*{a*2*bp}+r[:,None]*{2*bp}+k[None,:],r[:,None]<{a},0.)
    weights=tl.load(W+k[:,None]*{2*bp}+c[None,:],c[None,:]<{2*bp},0.)
    result=tl.dot(value,weights,input_precision="ieee")
    tl.store(Y+batch*{2*a*b}+r[:,None]*{2*b}+c[None,:],result,(r[:,None]<{a})&(c[None,:]<{compact_columns}))
'''
    half=a*b//2+1
    body += f'''@triton.jit
def transpose_half(X,Y,D):
    batch=tl.program_id(2).to(tl.int64)
    r=tl.program_id(0)*{tile}+tl.arange(0,{tile})[:,None]
    w=tl.program_id(1)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    value=tl.load(X+batch*{2*a*b}+r*{2*b}+w,(r<{a})&(w<{2*(b//2+1)}),0.)
    value=tl.reshape(tl.permute(tl.reshape(value,({tile},{tile},2)),1,0,2),({tile},{2*tile}))
    rr=tl.program_id(1)*{tile}+tl.arange(0,{tile})[:,None]
    ww=tl.program_id(0)*{2*tile}+tl.arange(0,{2*tile})[None,:]
    offset=rr*{2*a}+ww
    tl.store(Y+batch*D*2+offset,value,(rr<{b})&(ww<{2*a})&(offset<{2*half}))
'''
    if wide and a>=128 and b>=128:
        p=1<<(a-1).bit_length();bp=1<<(b-1).bit_length();bm,bn=128,128
        body+=f'''
@triton.jit
def cube0_pack1(P,W,T,Y):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*{bm}+tl.arange(0,{bm})
    k=tl.arange(0,{2*p})
    word=tl.program_id(1)*{bn}+tl.arange(0,{bn})
    x=tl.load(P+batch*{b*2*p}+row[:,None]*{2*p}+k[None,:],row[:,None]<{b},0.)
    w=tl.load(W+k[:,None]*{2*p}+word[None,:],word[None,:]<{2*p},0.)
    y=tl.dot(x,w,input_precision="ieee")
    vr,vi=tl.split(tl.reshape(y,({bm*bn//2},2)))
    vr=tl.reshape(vr,({bm},{bn//2}))
    vi=tl.reshape(vi,({bm},{bn//2}))
    freq=tl.program_id(1)*{bn//2}+tl.arange(0,{bn//2})
    off=row[:,None]*{a}+freq[None,:]
    good=(row[:,None]<{b})&(freq[None,:]<{a})
    tr=tl.load(T+off,good,0.)
    ti=tl.load(T+{a*b}+off,good,0.)
    re,im=vr*tr-vi*ti,vr*ti+vi*tr
    dst=batch*{a*2*bp}+freq[:,None]*{2*bp}+row[None,:]
    valid=(freq[:,None]<{a})&(row[None,:]<{bp})
    tl.store(Y+dst,tl.trans(re),valid)
    tl.store(Y+{bp}+dst,tl.trans(im),valid)
'''
    if wide and a>=128 and b>=128:
        # Real output uses only the even columns of the complex DFT matrix.
        # Keep the operands two-dimensional and transpose the real accumulator
        # directly into the API output, avoiding a complex intermediate.
        bp=1<<(b-1).bit_length()
        body+=f'''
@triton.jit
def cube1_finish_real(P,W,Y,D):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*64+tl.arange(0,64)
    k=tl.arange(0,{2*bp})
    freq=tl.program_id(1)*64+tl.arange(0,64)
    x=tl.load(P+batch*{a*2*bp}+row[:,None]*{2*bp}+k[None,:],row[:,None]<{a},0.)
    w=tl.load(W+k[:,None]*{bp}+freq[None,:],freq[None,:]<{bp},0.)
    value=tl.dot(x,w,input_precision="ieee")
    tl.store(Y+batch*D+freq[:,None]*{a}+row[None,:],tl.trans(value),(freq[:,None]<{b})&(row[None,:]<{a}))
'''
    if wide and a>=128 and b>=128:
        p=1<<(a-1).bit_length();pb=1<<(b-1).bit_length()
        body+=f'''
@triton.jit
def cube0_real_pack1(P,W,T,Y):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*128+tl.arange(0,128)
    k=tl.arange(0,{p})
    word=tl.program_id(1)*128+tl.arange(0,128)
    x=tl.load(P+batch*{b*p}+row[:,None]*{p}+k[None,:],row[:,None]<{b},0.)
    w=tl.load(W+k[:,None]*{2*p}+word[None,:],word[None,:]<{2*p},0.)
    y=tl.dot(x,w,input_precision="ieee")
    vr,vi=tl.split(tl.reshape(y,(8192,2)))
    vr=tl.reshape(vr,(128,64));vi=tl.reshape(vi,(128,64))
    freq=tl.program_id(1)*64+tl.arange(0,64)
    off=row[:,None]*{a}+freq[None,:]
    good=(row[:,None]<{b})&(freq[None,:]<{a})
    tr=tl.load(T+off,good,0.);ti=tl.load(T+{a*b}+off,good,0.)
    re,im=vr*tr-vi*ti,vr*ti+vi*tr
    dst=batch*{a*2*pb}+freq[:,None]*{2*pb}+row[None,:]
    valid=(freq[:,None]<{a})&(row[None,:]<{pb})
    tl.store(Y+dst,tl.trans(re),valid)
    tl.store(Y+{pb}+dst,tl.trans(im),valid)

'''
    return _module_source(body)



def emit_matrix_fourstep(a,b,phase,direction,out_dir,wide=False):
    names=('pack0','cube0','pack1','cube1','transpose','pack0_real','cube0_real','cube1_half','transpose_half','transpose_real','cube0_pack1','cube1_finish_real','cube0_real_pack1')
    if phase not in names or direction not in ('forward','inverse'):
        raise ValueError('Invalid matrix four-step phase or direction')
    if phase in ('cube0_pack1','cube1_finish_real','cube0_real_pack1') and not (wide and a>=128 and b>=128):
        raise ValueError('Fused matrix packing requires wide axes of at least 128')
    path=Path(out_dir)/(f'npu_matrix_fourstep_{a}_{b}_{phase}_{direction}'+('_wide' if wide else '')+'.py')
    path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,build_matrix_fourstep(a,b,wide=wide))
    args=['X','Y','D'] if phase in ('transpose_half','transpose_real') else ['X','Y'] if phase=='transpose' else ['X','P','T'] if phase.startswith('pack') else ['P','W','Y']
    if phase in ('cube0_pack1','cube0_real_pack1'): args=['P','W','T','Y']
    if phase=='cube1_finish_real': args=['P','W','Y','D']
    signature=','.join(['*fp32:16']*len(args))
    if phase in ('transpose_half','transpose_real'): signature='*fp32:16,*fp32:16,i64'
    if phase=='cube1_finish_real': signature='*fp32:16,*fp32:16,*fp32:16,i64'
    if wide and phase.startswith('cube') and phase not in ('cube0_pack1','cube1_finish_real','cube0_real_pack1'):
        args.append('NB');signature+=',i32'
    meta=dict(module_path=str(path),kernel_name=phase,arg_names=args,
        signature=signature,num_warps=4,num_stages=1,
        batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
