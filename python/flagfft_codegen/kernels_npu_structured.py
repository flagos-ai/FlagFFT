# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Structured Ascend FFTs validated with the fixed split/join transport.

The one-pass leaf uses a 32 x (N/32) decomposition. Larger four-step plans
reuse the two-pass row/column execution node with fused twiddle and stores.
"""
import json
from pathlib import Path
from .artifacts import write_text_atomic

FFT_HELPERS = """
@triton.jit
def split4(v, N: tl.constexpr, SPAN: tl.constexpr):
    G: tl.constexpr = N // SPAN
    Q: tl.constexpr = SPAN // 4
    p = tl.permute(tl.reshape(v, (G, 2, 2, Q)), 0, 3, 1, 2)
    # Flatten before split: keep both channels in compact one-dimensional
    # storage instead of creating a strided slice in a small middle axis.
    a, b = tl.split(tl.reshape(p,(N//2,2)))
    x0, x2 = tl.split(tl.reshape(a,(N//4,2)))
    x1, x3 = tl.split(tl.reshape(b,(N//4,2)))
    return (tl.reshape(x0, (N//4,)), tl.reshape(x1, (N//4,)),
            tl.reshape(x2, (N//4,)), tl.reshape(x3, (N//4,)))


@triton.jit
def merge4(a, b, c, d, N: tl.constexpr, SPAN: tl.constexpr):
    p = tl.join(tl.join(a, c), tl.join(b, d))
    p = tl.reshape(p, (N//SPAN, SPAN//4, 2, 2))
    return tl.reshape(tl.permute(p, 0, 2, 3, 1), (N,))


@triton.jit
def stage4(r, i, W, N: tl.constexpr, SPAN: tl.constexpr, S: tl.constexpr, SORT:tl.constexpr=False, SLICE:tl.constexpr=False, C:tl.constexpr=1, KEEP:tl.constexpr=False):
    if SLICE:
        x0r,x1r,x2r,x3r=split4_slice(r,N,SPAN,KEEP)
        x0i,x1i,x2i,x3i=split4_slice(i,N,SPAN,KEEP)
    else:
        x0r, x1r, x2r, x3r = split4(r, N, SPAN)
        x0i, x1i, x2i, x3i = split4(i, N, SPAN)
    t0r, t0i = x0r+x2r, x0i+x2i
    t1r, t1i = x0r-x2r, x0i-x2i
    t2r, t2i = x1r+x3r, x1i+x3i
    t3r, t3i = x1r-x3r, x1i-x3i
    g0r, g0i = t0r+t2r, t0i+t2i
    a1r, a1i = t1r+t3i, t1i-t3r
    a2r, a2i = t0r-t2r, t0i-t2i
    a3r, a3i = t1r-t3i, t1i+t3r
    if SPAN > 4*C:
        j = tl.arange(0, N//4)
        w1r = tl.load(W + S*6*(N//4) + j)
        w1i = tl.load(W + S*6*(N//4) + N//4 + j)
        w2r = tl.load(W + S*6*(N//4) + 2*(N//4) + j)
        w2i = tl.load(W + S*6*(N//4) + 3*(N//4) + j)
        w3r = tl.load(W + S*6*(N//4) + 4*(N//4) + j)
        w3i = tl.load(W + S*6*(N//4) + 5*(N//4) + j)
        if KEEP:
            w1r=tl.reshape(w1r,(N//SPAN,SPAN//4))
            w1i=tl.reshape(w1i,(N//SPAN,SPAN//4))
            w2r=tl.reshape(w2r,(N//SPAN,SPAN//4))
            w2i=tl.reshape(w2i,(N//SPAN,SPAN//4))
            w3r=tl.reshape(w3r,(N//SPAN,SPAN//4))
            w3i=tl.reshape(w3i,(N//SPAN,SPAN//4))
        g1r, g1i = a1r*w1r-a1i*w1i, a1r*w1i+a1i*w1r
        g2r, g2i = a2r*w2r-a2i*w2i, a2r*w2i+a2i*w2r
        g3r, g3i = a3r*w3r-a3i*w3i, a3r*w3i+a3i*w3r
    else:
        g1r, g1i, g2r, g2i, g3r, g3i = a1r, a1i, a2r, a2i, a3r, a3i
    if SORT:
        return pack4(g0r,g1r,g2r,g3r,N), pack4(g0i,g1i,g2i,g3i,N)
    return (merge4(g0r,g1r,g2r,g3r,N,SPAN),
            merge4(g0i,g1i,g2i,g3i,N,SPAN))


@triton.jit
def split4_slice(v,N:tl.constexpr,SPAN:tl.constexpr,KEEP:tl.constexpr):
    p=tl.reshape(v,(N//SPAN,SPAN))
    a=al.extract_slice(p,[0,0],[N//SPAN,SPAN//4],[1,1])
    b=al.extract_slice(p,[0,SPAN//4],[N//SPAN,SPAN//4],[1,1])
    c=al.extract_slice(p,[0,SPAN//2],[N//SPAN,SPAN//4],[1,1])
    d=al.extract_slice(p,[0,3*SPAN//4],[N//SPAN,SPAN//4],[1,1])
    if KEEP:
        return a,b,c,d
    else:
        return (tl.reshape(a,(N//4,)),tl.reshape(b,(N//4,)),
                tl.reshape(c,(N//4,)),tl.reshape(d,(N//4,)))


@triton.jit
def fft_axis_sliced(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,LOG:tl.constexpr,KEEP:tl.constexpr=False):
    for s in tl.static_range(0,LOG//2):
        r,i=stage4(r,i,W,N,L*C//4**s,s,True,True,C,KEEP)
    if LOG%2:
        p=tl.reshape(r,(N//(2*C),2*C))
        q=tl.reshape(i,(N//(2*C),2*C))
        ar=tl.reshape(al.extract_slice(p,[0,0],[N//(2*C),C],[1,1]),(N//2,))
        br=tl.reshape(al.extract_slice(p,[0,C],[N//(2*C),C],[1,1]),(N//2,))
        ai=tl.reshape(al.extract_slice(q,[0,0],[N//(2*C),C],[1,1]),(N//2,))
        bi=tl.reshape(al.extract_slice(q,[0,C],[N//(2*C),C],[1,1]),(N//2,))
        rr=tl.zeros((N,),tl.float32)
        ii=tl.zeros((N,),tl.float32)
        rr=al.insert_slice(rr,ar+br,[0],[N//2],[1])
        r=al.insert_slice(rr,ar-br,[N//2],[N//2],[1])
        ii=al.insert_slice(ii,ai+bi,[0],[N//2],[1])
        i=al.insert_slice(ii,ai-bi,[N//2],[N//2],[1])
    return r,i


@triton.jit
def pack4(a,b,c,d,N:tl.constexpr):
    a=tl.reshape(a,(N//4,));b=tl.reshape(b,(N//4,))
    c=tl.reshape(c,(N//4,));d=tl.reshape(d,(N//4,))
    v=tl.zeros((N,),tl.float32)
    v=al.insert_slice(v,a,[0],[N//4],[1])
    v=al.insert_slice(v,b,[N//4],[N//4],[1])
    v=al.insert_slice(v,c,[N//2],[N//4],[1])
    return al.insert_slice(v,d,[3*N//4],[N//4],[1])


"""

def emit_structured(*, length, direction, out_dir, first=None, n1=0, n2=0, tile=8):
    if direction not in ('forward','inverse'):
        raise ValueError('Invalid FFT direction')
    inverse=direction=='inverse'
    if first is None:
        if length not in (1024,2048):
            raise ValueError('Fused structured leaf supports 1024/2048')
        a,b=32,length//32
        name=f'npu_structured_fused_n{length}_{direction}'
        signature='*fp32:16,*fp32:16,*fp32:16,*fp32:16,*fp32:16,i32'
        args=['in_ptr','out_ptr','wa','wb','cross','nbatch']
        body=f"""@triton.jit
def {name}(in_ptr,out_ptr,wa,wb,cross,nbatch):
    pid=tl.program_id(0)
    k=tl.arange(0,{length})
    raw=tl.load(in_ptr+pid*{2*length}+tl.arange(0,{2*length}))
    r,i=tl.split(tl.reshape(raw,({length},2)))
    if {inverse}: i=-i
    r,i=fft_axis_sliced(r,i,wa,{length},{a},{b},{a.bit_length()-1})
    cr=tl.load(cross+k);ci=tl.load(cross+{length}+k)
    rr,ii=r*cr-i*ci,r*ci+i*cr
    rr=tl.reshape(tl.trans(tl.reshape(rr,({a},{b}))),({length},))
    ii=tl.reshape(tl.trans(tl.reshape(ii,({a},{b}))),({length},))
    r,i=fft_axis_sliced(rr,ii,wb,{length},{b},{a},{b.bit_length()-1})
    if {inverse}: i=-i
    out=tl.reshape(tl.join(r,i),({2*length},))
    tl.store(out_ptr+pid*{2*length}+tl.arange(0,{2*length}),out)
"""
        kind='leaf'
    else:
        if (any(v < 16 or v > 2048 or v & (v-1) for v in (n1,n2))
                or tile not in (1,2,4,8,16) or n1%tile or n2%tile
                or max(n1,n2)*tile > 4096):
            raise ValueError('Tiled structured FFT exceeds the power-of-two axis resource bounds')
        l,width=(n1,n2) if first else (n2,n1)
        name=f'npu_structured_axis_{l}_{width}_c{tile}_{int(first)}_{direction}'
        args=(['in_ptr','twiddle_ptr','out_ptr','roots_ptr','nbatch'] if first else
              ['in_ptr','out_ptr','roots_ptr','nbatch'])
        signature=','.join(['*fp32:16']*(len(args)-1)+['i32'])
        body=f"""@triton.jit
def {name}({','.join(args)}):
    col=tl.program_id(0)*{tile}+tl.arange(0,{tile})
    batch=tl.program_id(1)
    row=tl.arange(0,{l})
    word=tl.program_id(0)*{2*tile}+tl.arange(0,{2*tile})
    offsets=batch*{2*n1*n2}+row[:,None]*{2*width}+word[None,:]
    raw=tl.load(in_ptr+offsets)
    r,i=tl.split(tl.reshape(raw,({l*tile},2)))
    if {inverse}: i=-i
    r,i=fft_axis_sliced(r,i,roots_ptr,{l*tile},{l},{tile},{l.bit_length()-1})
    if {inverse}: i=-i
    r=tl.reshape(r,({l},{tile}));i=tl.reshape(i,({l},{tile}))
"""
        if first:
            body+=f"""    tw=row[:,None]*{width}+col[None,:]
    cr=tl.load(twiddle_ptr+tw);ci=tl.load(twiddle_ptr+{n1*n2}+tw)
    rr,ii=r*cr-i*ci,r*ci+i*cr
    rr=tl.reshape(tl.trans(rr),({l*tile},));ii=tl.reshape(tl.trans(ii),({l*tile},))
    raw=tl.reshape(tl.join(rr,ii),({tile},{2*l}))
    dest=batch*{2*n1*n2}+col[:,None]*{2*l}+tl.arange(0,{2*l})[None,:]
"""
        else:
            body+=f"""    raw=tl.reshape(tl.join(tl.reshape(r,({l*tile},)),tl.reshape(i,({l*tile},))),({l},{2*tile}))
    dest=offsets
"""
        body+='    tl.store(out_ptr+dest,raw)\n'
        kind='four_step_row' if first else 'four_step_col'
    source=('import torch\nimport torch_npu\nimport triton\nimport triton.language as tl\n'
            'import triton.language.extra.cann.extension as al\n\n'+FFT_HELPERS+body)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/(name+'.py');write_text_atomic(path,source)
    meta=dict(module_path=str(path),kernel_name=name,signature=signature,num_warps=4,
        num_stages=1,batch_per_block=1,arg_names=args,kernel_type=kind,length=length,
        inner_pack=tile if first is not None else 1,tle_fused_twiddle=first is not None,
        dtype='complex64',direction=direction)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
