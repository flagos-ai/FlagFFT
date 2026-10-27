"""Packed four-step FFT and compact coefficient tables for Ascend experiments."""
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from ascend_fourstep_probe import fft_axis_sliced, split4_slice, pack4


@triton.jit
def compact_stage(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,SPAN:tl.constexpr,S:tl.constexpr):
    x0r,x1r,x2r,x3r=split4_slice(r,N,SPAN,False)
    x0i,x1i,x2i,x3i=split4_slice(i,N,SPAN,False)
    t0r,t0i=x0r+x2r,x0i+x2i
    t1r,t1i=x0r-x2r,x0i-x2i
    t2r,t2i=x1r+x3r,x1i+x3i
    t3r,t3i=x1r-x3r,x1i-x3i
    g0r,g0i=t0r+t2r,t0i+t2i
    a1r,a1i=t1r+t3i,t1i-t3r
    a2r,a2i=t0r-t2r,t0i-t2i
    a3r,a3i=t1r-t3i,t1i+t3r
    if SPAN>4*C:
        K:tl.constexpr=SPAN//(4*C)
        PAD:tl.constexpr=max(8,L//4)
        j=tl.arange(0,K)
        w1r=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        w1i=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        w2r=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+2*PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        w2i=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+3*PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        w3r=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+4*PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        w3i=tl.reshape(tl.broadcast_to(tl.load(W+S*6*PAD+5*PAD+j)[None,:,None],(N//SPAN,K,C)),(N//4,))
        g1r,g1i=a1r*w1r-a1i*w1i,a1r*w1i+a1i*w1r
        g2r,g2i=a2r*w2r-a2i*w2i,a2r*w2i+a2i*w2r
        g3r,g3i=a3r*w3r-a3i*w3i,a3r*w3i+a3i*w3r
    else:
        g1r,g1i,g2r,g2i,g3r,g3i=a1r,a1i,a2r,a2i,a3r,a3i
    return pack4(g0r,g1r,g2r,g3r,N),pack4(g0i,g1i,g2i,g3i,N)


@triton.jit
def compact_axis(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,LOG:tl.constexpr):
    for s in tl.static_range(0,LOG//2):
        r,i=compact_stage(r,i,W,N,L,C,L*C//4**s,s)
    if LOG%2:
        p=tl.reshape(r,(N//(2*C),2*C));q=tl.reshape(i,(N//(2*C),2*C))
        ar=tl.reshape(al.extract_slice(p,[0,0],[N//(2*C),C],[1,1]),(N//2,))
        br=tl.reshape(al.extract_slice(p,[0,C],[N//(2*C),C],[1,1]),(N//2,))
        ai=tl.reshape(al.extract_slice(q,[0,0],[N//(2*C),C],[1,1]),(N//2,))
        bi=tl.reshape(al.extract_slice(q,[0,C],[N//(2*C),C],[1,1]),(N//2,))
        rr=tl.zeros((N,),tl.float32);ii=tl.zeros((N,),tl.float32)
        rr=al.insert_slice(rr,ar+br,[0],[N//2],[1]);r=al.insert_slice(rr,ar-br,[N//2],[N//2],[1])
        ii=al.insert_slice(ii,ai+bi,[0],[N//2],[1]);i=al.insert_slice(ii,ai-bi,[N//2],[N//2],[1])
    return r,i


@triton.jit
def packed_fft(X,Y,WA,WB,CROSS,NB:tl.constexpr,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,P:tl.constexpr,COMPACT:tl.constexpr):
    bid=tl.program_id(0)
    batch=bid*P+tl.arange(0,P)
    lane=tl.arange(0,2*N)
    raw=tl.load(X+batch[:,None]*2*N+lane[None,:],batch[:,None]<NB,0)
    r,i=tl.split(tl.reshape(raw,(P*N,2)))
    if COMPACT:
        r,i=compact_axis(r,i,WA,P*N,A,B,A.bit_length()-1)
    else:
        r,i=fft_axis_sliced(r,i,WA,P*N,A,B,A.bit_length()-1)
    k=tl.arange(0,N)
    c=tl.reshape(tl.load(CROSS+k),(A,1,B))
    d=tl.reshape(tl.load(CROSS+N+k),(A,1,B))
    r=tl.reshape(r,(A,P,B));i=tl.reshape(i,(A,P,B))
    rr,ii=r*c-i*d,r*d+i*c
    rr=tl.reshape(tl.permute(rr,2,1,0),(P*N,))
    ii=tl.reshape(tl.permute(ii,2,1,0),(P*N,))
    if COMPACT:
        r,i=compact_axis(rr,ii,WB,P*N,B,P*A,B.bit_length()-1)
    else:
        r,i=fft_axis_sliced(rr,ii,WB,P*N,B,P*A,B.bit_length()-1)
    r=tl.reshape(tl.permute(tl.reshape(r,(B,P,A)),1,0,2),(P,N))
    i=tl.reshape(tl.permute(tl.reshape(i,(B,P,A)),1,0,2),(P,N))
    for p in tl.static_range(0,P):
        ar=tl.reshape(al.extract_slice(r,[p,0],[1,N],[1,1]),(N,))
        ai=tl.reshape(al.extract_slice(i,[p,0],[1,N],[1,1]),(N,))
        out=tl.reshape(tl.join(ar,ai),(2*N,))
        tl.store(Y+(bid*P+p)*2*N+lane,out,bid*P+p<NB)


def compact_twiddles(length):
    pad=max(8,length//4)
    rows=[]
    for s in range((length.bit_length()-1)//2):
        span=length//4**s
        j=torch.arange(pad,dtype=torch.float64)
        for k in (1,2,3):
            angle=-2*math.pi*k*j/span
            rows.extend((angle.cos().float(),angle.sin().float()))
    return torch.cat(rows).npu()
