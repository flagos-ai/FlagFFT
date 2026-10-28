"""Compensated half-precision Cube DFTs with FP32 external FFT values."""
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from ascend_fourstep_probe import fft_axis_sliced


@triton.jit
def dft_cube(V,WH,WL,L:tl.constexpr,C:tl.constexpr,TERMS:tl.constexpr):
    row=tl.arange(0,2*L)
    col=tl.arange(0,2*L)
    high_w=tl.load(WH+row[:,None]*(2*L)+col[None,:])
    if TERMS==0:
        return tl.dot(high_w,V,input_precision="ieee")
    else:
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
def cube_fourstep(X,Y,AH,AL,BH,BL,CROSS,WA,WB,NB:tl.constexpr,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,TERMS:tl.constexpr,CA:tl.constexpr,CB:tl.constexpr,LOGA:tl.constexpr,LOGB:tl.constexpr,P:tl.constexpr):
    bid=tl.program_id(0)
    lane=tl.arange(0,2*N)
    batch=bid*P+tl.arange(0,P)
    raw=tl.load(X+batch[:,None]*2*N+lane[None,:],batch[:,None]<NB,0)
    r,i=tl.split(tl.reshape(raw,(P*N,2)))
    if CA:
        r=tl.reshape(tl.permute(tl.reshape(r,(P,A,B)),1,0,2),(P*N,))
        i=tl.reshape(tl.permute(tl.reshape(i,(P,A,B)),1,0,2),(P*N,))
        v=tl.zeros((2*P*N,),tl.float32)
        v=al.insert_slice(v,r,[0],[P*N],[1]);v=al.insert_slice(v,i,[P*N],[P*N],[1])
        first=dft_cube(tl.reshape(v,(2*A,P*B)),AH,AL,A,P*B,TERMS)
        r=tl.reshape(al.extract_slice(first,[0,0],[A,P*B],[1,1]),(P*N,))
        i=tl.reshape(al.extract_slice(first,[A,0],[A,P*B],[1,1]),(P*N,))
    else:
        r,i=fft_axis_sliced(r,i,WA,P*N,A,B,LOGA)
    k=tl.arange(0,N)
    c=tl.reshape(tl.load(CROSS+k),(A,1,B));d=tl.reshape(tl.load(CROSS+N+k),(A,1,B))
    r=tl.reshape(r,(A,P,B));i=tl.reshape(i,(A,P,B))
    rr,ii=r*c-i*d,r*d+i*c
    rr=tl.reshape(tl.permute(rr,2,1,0),(P*N,))
    ii=tl.reshape(tl.permute(ii,2,1,0),(P*N,))
    if CB:
        v=tl.zeros((2*P*N,),tl.float32)
        v=al.insert_slice(v,rr,[0],[P*N],[1]);v=al.insert_slice(v,ii,[P*N],[P*N],[1])
        second=dft_cube(tl.reshape(v,(2*B,P*A)),BH,BL,B,P*A,TERMS)
        r=tl.reshape(al.extract_slice(second,[0,0],[B,P*A],[1,1]),(P*N,))
        i=tl.reshape(al.extract_slice(second,[B,0],[B,P*A],[1,1]),(P*N,))
    else:
        r,i=fft_axis_sliced(rr,ii,WB,P*N,B,P*A,LOGB)
    r=tl.reshape(tl.permute(tl.reshape(r,(B,P,A)),1,0,2),(P,N))
    i=tl.reshape(tl.permute(tl.reshape(i,(B,P,A)),1,0,2),(P,N))
    for p in tl.static_range(0,P):
        ar=tl.reshape(al.extract_slice(r,[p,0],[1,N],[1,1]),(N,))
        ai=tl.reshape(al.extract_slice(i,[p,0],[1,N],[1,1]),(N,))
        out=tl.reshape(tl.join(ar,ai),(2*N,))
        tl.store(Y+(bid*P+p)*2*N+lane,out,bid*P+p<NB)


def dft_weights(length,ieee=False):
    j=torch.arange(length,dtype=torch.float64)
    angle=-2*math.pi*j[:,None]*j[None,:]/length
    re,im=angle.cos().float(),angle.sin().float()
    w=torch.cat((torch.cat((re,-im),dim=1),torch.cat((im,re),dim=1)),dim=0)
    if ieee:
        return w.npu(),torch.zeros(1,device="npu")
    high=w.half()
    low=((w-high.float())*4096.0).half()
    return high.npu(),low.npu()
