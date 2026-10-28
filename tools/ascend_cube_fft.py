"""Compensated half-precision Cube DFTs with FP32 external FFT values."""
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def dft_cube(V,WH,WL,L:tl.constexpr,C:tl.constexpr,TERMS:tl.constexpr):
    row=tl.arange(0,2*L)
    col=tl.arange(0,2*L)
    high_w=tl.load(WH+row[:,None]*(2*L)+col[None,:])
    low_w=tl.load(WL+row[:,None]*(2*L)+col[None,:])
    high_x=V.to(tl.float16)
    low_x=((V-high_x.to(tl.float32))*4096.0).to(tl.float16)
    correction=tl.dot(low_w,high_x)
    correction=tl.dot(high_w,low_x,correction)
    high=tl.dot(high_w,high_x)
    result=high+correction*0.000244140625
    if TERMS==4:
        ll=tl.dot(low_w,low_x)
        result=result+ll*0.000000059604644775390625
    return result


@triton.jit
def cube_fourstep(X,Y,AH,AL,BH,BL,CROSS,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,TERMS:tl.constexpr):
    bid=tl.program_id(0)
    k=tl.arange(0,N)
    raw=tl.load(X+bid*2*N+tl.arange(0,2*N))
    r,i=tl.split(tl.reshape(raw,(N,2)))
    v=tl.zeros((2*N,),tl.float32)
    v=al.insert_slice(v,r,[0],[N],[1]);v=al.insert_slice(v,i,[N],[N],[1])
    v=tl.reshape(v,(2*A,B))
    first=dft_cube(v,AH,AL,A,B,TERMS)
    r=tl.reshape(al.extract_slice(first,[0,0],[A,B],[1,1]),(N,))
    i=tl.reshape(al.extract_slice(first,[A,0],[A,B],[1,1]),(N,))
    c=tl.load(CROSS+k);d=tl.load(CROSS+N+k)
    rr,ii=r*c-i*d,r*d+i*c
    rr=tl.reshape(tl.trans(tl.reshape(rr,(A,B))),(N,))
    ii=tl.reshape(tl.trans(tl.reshape(ii,(A,B))),(N,))
    v=tl.zeros((2*N,),tl.float32)
    v=al.insert_slice(v,rr,[0],[N],[1]);v=al.insert_slice(v,ii,[N],[N],[1])
    second=dft_cube(tl.reshape(v,(2*B,A)),BH,BL,B,A,TERMS)
    r=tl.reshape(al.extract_slice(second,[0,0],[B,A],[1,1]),(N,))
    i=tl.reshape(al.extract_slice(second,[B,0],[B,A],[1,1]),(N,))
    out=tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)


def dft_weights(length):
    j=torch.arange(length,dtype=torch.float64)
    angle=-2*math.pi*j[:,None]*j[None,:]/length
    re,im=angle.cos().float(),angle.sin().float()
    w=torch.cat((torch.cat((re,-im),dim=1),torch.cat((im,re),dim=1)),dim=0)
    high=w.half()
    low=((w-high.float())*4096.0).half()
    return high.npu(),low.npu()
