"""Experimental packed-vector FFT, using the split/join compiler fix.

Run on Ascend with the isolated FlagTree/FLIR build. Output is natural order.
This probe does not alter planner defaults.
"""
import argparse
import hashlib
import os
import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


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
def stage4(r, i, W, N: tl.constexpr, SPAN: tl.constexpr, S: tl.constexpr, SORT:tl.constexpr=False, SLICE:tl.constexpr=False, C:tl.constexpr=1):
    if SLICE:
        x0r,x1r,x2r,x3r=split4_slice(r,N,SPAN)
        x0i,x1i,x2i,x3i=split4_slice(i,N,SPAN)
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
def split4_slice(v,N:tl.constexpr,SPAN:tl.constexpr):
    p=tl.reshape(v,(N//SPAN,SPAN))
    a=al.extract_slice(p,[0,0],[N//SPAN,SPAN//4],[1,1])
    b=al.extract_slice(p,[0,SPAN//4],[N//SPAN,SPAN//4],[1,1])
    c=al.extract_slice(p,[0,SPAN//2],[N//SPAN,SPAN//4],[1,1])
    d=al.extract_slice(p,[0,3*SPAN//4],[N//SPAN,SPAN//4],[1,1])
    return (tl.reshape(a,(N//4,)),tl.reshape(b,(N//4,)),
            tl.reshape(c,(N//4,)),tl.reshape(d,(N//4,)))


@triton.jit
def fft_axis_sliced(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,LOG:tl.constexpr):
    for s in tl.static_range(0,LOG//2):
        r,i=stage4(r,i,W,N,L*C//4**s,s,True,True,C)
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
def fourstep_fused(X,Y,WA,WB,CROSS,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,
                   LOGA:tl.constexpr,LOGB:tl.constexpr,NATIVE:tl.constexpr):
    bid=tl.program_id(0)
    v=tl.load(X+bid*2*N+tl.arange(0,2*N))
    r,i=tl.split(tl.reshape(v,(N,2)))
    r,i=fft_axis_sliced(r,i,WA,N,A,B,LOGA)
    k=tl.arange(0,N)
    c=tl.load(CROSS+k);d=tl.load(CROSS+N+k)
    rr,ii=r*c-i*d,r*d+i*c
    rr=tl.reshape(tl.trans(tl.reshape(rr,(A,B))),(N,))
    ii=tl.reshape(tl.trans(tl.reshape(ii,(A,B))),(N,))
    r,i=fft_axis_sliced(rr,ii,WB,N,B,A,LOGB)
    if NATIVE:
        out=tl.zeros((2*N,),tl.float32)
        out=al.insert_slice(out,r,[0],[N],[2])
        out=al.insert_slice(out,i,[1],[N],[2])
    else:
        out=tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)


@triton.jit
def fourstep_sliced_first(X,T,W,CROSS,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,LOG:tl.constexpr):
    bid=tl.program_id(0)
    v=tl.load(X+bid*2*N+tl.arange(0,2*N))
    r,i=tl.split(tl.reshape(v,(N,2)))
    r,i=fft_axis_sliced(r,i,W,N,A,B,LOG)
    k=tl.arange(0,N)
    c=tl.load(CROSS+k);d=tl.load(CROSS+N+k)
    rr,ii=r*c-i*d,r*d+i*c
    rr=tl.reshape(tl.trans(tl.reshape(rr,(A,B))),(N,))
    ii=tl.reshape(tl.trans(tl.reshape(ii,(A,B))),(N,))
    tl.store(T+bid*2*N+k,rr)
    tl.store(T+bid*2*N+N+k,ii)


@triton.jit
def fourstep_sliced_second(T,Y,W,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,LOG:tl.constexpr):
    bid=tl.program_id(0)
    k=tl.arange(0,N)
    r=tl.load(T+bid*2*N+k)
    i=tl.load(T+bid*2*N+N+k)
    r,i=fft_axis_sliced(r,i,W,N,B,A,LOG)
    out=tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)


@triton.jit
def pack4(a,b,c,d,N:tl.constexpr):
    v=tl.zeros((N,),tl.float32)
    v=al.insert_slice(v,a,[0],[N//4],[1])
    v=al.insert_slice(v,b,[N//4],[N//4],[1])
    v=al.insert_slice(v,c,[N//2],[N//4],[1])
    return al.insert_slice(v,d,[3*N//4],[N//4],[1])


@triton.jit
def fft_autosort(X,Y,W,N:tl.constexpr,LOGN:tl.constexpr):
    bid=tl.program_id(0)
    v=tl.load(X+bid*2*N+tl.arange(0,2*N))
    r,i=tl.split(tl.reshape(v,(N,2)))
    for s in tl.static_range(0,LOGN//2):
        r,i=stage4(r,i,W,N,N//4**s,s,True)
    if LOGN%2:
        ar,br=tl.split(tl.reshape(r,(N//2,2)))
        ai,bi=tl.split(tl.reshape(i,(N//2,2)))
        rr=tl.zeros((N,),tl.float32)
        ii=tl.zeros((N,),tl.float32)
        rr=al.insert_slice(rr,ar+br,[0],[N//2],[1])
        r=al.insert_slice(rr,ar-br,[N//2],[N//2],[1])
        ii=al.insert_slice(ii,ai+bi,[0],[N//2],[1])
        i=al.insert_slice(ii,ai-bi,[N//2],[N//2],[1])
    out=tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)


@triton.jit
def fft_planar(X, Y, W, N: tl.constexpr, LOGN: tl.constexpr, SHAPE:tl.constexpr, ORDER:tl.constexpr):
    bid = tl.program_id(0)
    v = tl.load(X + bid*2*N + tl.arange(0, 2*N))
    r, i = tl.split(tl.reshape(v, (N, 2)))
    for s in tl.static_range(0, (LOGN)//2):
        r, i = stage4(r, i, W, N, N//(4**s), s)
    if (LOGN) % 2:
        ar, br = tl.split(tl.reshape(r, (N//2, 2)))
        ai, bi = tl.split(tl.reshape(i, (N//2, 2)))
        r = tl.reshape(tl.join(ar+br, ar-br), (N,))
        i = tl.reshape(tl.join(ai+bi, ai-bi), (N,))
    r = tl.reshape(tl.permute(tl.reshape(r,SHAPE),ORDER),(N,))
    i = tl.reshape(tl.permute(tl.reshape(i,SHAPE),ORDER),(N,))
    out = tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)



@triton.jit
def digit_reverse(k, L: tl.constexpr, LOG: tl.constexpr):
    out = tl.full(k.shape, 0, tl.int32)
    for s in tl.static_range(0, LOG//2):
        out = out + ((k // (L//(4**(s+1)))) % 4) * (4**s)
    if LOG % 2:
        out = out + (k%2)*(L//2)
    return out


@triton.jit
def fft_axis(r, i, W, N: tl.constexpr, L: tl.constexpr,
             C: tl.constexpr, LOG: tl.constexpr):
    for s in tl.static_range(0, LOG//2):
        r, i = stage4(r, i, W, N, L*C//(4**s), s)
    if LOG % 2:
        p = tl.permute(tl.reshape(r, (N//(2*C),2,C)),0,2,1)
        q = tl.permute(tl.reshape(i, (N//(2*C),2,C)),0,2,1)
        ar, br = tl.split(tl.reshape(p,(N//2,2)))
        ai, bi = tl.split(tl.reshape(q,(N//2,2)))
        rr = tl.reshape(tl.join(ar+br, ar-br),(N//(2*C),C,2))
        ii = tl.reshape(tl.join(ai+bi, ai-bi),(N//(2*C),C,2))
        r = tl.reshape(tl.permute(rr,0,2,1),(N,))
        i = tl.reshape(tl.permute(ii,0,2,1),(N,))
    return r, i


@triton.jit
def fourstep_first(X,T,W,CROSS,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,LOGA:tl.constexpr,SHAPE:tl.constexpr,ORDER:tl.constexpr):
    bid=tl.program_id(0)
    v=tl.load(X+bid*2*N+tl.arange(0,2*N))
    r,i=tl.split(tl.reshape(v,(N,2)))
    r,i=fft_axis(r,i,W,N,A,B,LOGA)
    r=tl.reshape(tl.permute(tl.reshape(r,SHAPE),ORDER),(N,))
    i=tl.reshape(tl.permute(tl.reshape(i,SHAPE),ORDER),(N,))
    k=tl.arange(0,N)
    c=tl.load(CROSS+k)
    d=tl.load(CROSS+N+k)
    yr,yi=r*c-i*d,r*d+i*c
    dest=k
    tl.store(T+bid*2*N+dest,yr)
    tl.store(T+bid*2*N+N+dest,yi)


@triton.jit
def fourstep_second(T,Y,W,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr,LOGB:tl.constexpr,SHAPE:tl.constexpr,ORDER:tl.constexpr):
    bid=tl.program_id(0)
    k=tl.arange(0,N)
    r=tl.load(T+bid*2*N+k)
    i=tl.load(T+bid*2*N+N+k)
    r,i=fft_axis(r,i,W,N,B,1,LOGB)
    r=tl.reshape(tl.permute(tl.reshape(r,SHAPE),ORDER),(N,))
    i=tl.reshape(tl.permute(tl.reshape(i,SHAPE),ORDER),(N,))
    r=tl.reshape(tl.trans(tl.reshape(r,(A,B))),(N,))
    i=tl.reshape(tl.trans(tl.reshape(i,(A,B))),(N,))
    out=tl.reshape(tl.join(r,i),(2*N,))
    tl.store(Y+bid*2*N+tl.arange(0,2*N),out)


def axis_twiddles(n,l,c):
    stages=[]
    for s in range((l.bit_length()-1)//2):
        span=l//4**s
        j=(torch.arange(n//4,dtype=torch.int64)//c)%(span//4)
        for b in (1,2,3):
            angle=-2*math.pi*b*j.double()/span
            stages.extend((angle.cos().float(),angle.sin().float()))
    return torch.cat(stages).npu()


def reverse_cpu(k,l):
    log=l.bit_length()-1
    out=torch.zeros_like(k)
    for s in range(log//2):
        out+=((k//(l//4**(s+1)))%4)*4**s
    if log%2: out+=(k%2)*(l//2)
    return out


def twiddles(n):
    stages = []
    for s in range((n.bit_length()-1)//2):
        span=n//4**s
        j=torch.arange(n//4, dtype=torch.float64) % (span//4)
        for b in (1,2,3):
            angle=-2*math.pi*b*j/span
            stages.extend((angle.cos().float(), angle.sin().float()))
    return torch.cat(stages).npu()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=2048)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tuning',choices=['default','multi','vec2','singlebuffer','ubsave','flatten'],default='default')
    p.add_argument('--fused-fourstep',action='store_true')
    p.add_argument('--native-interleave',action='store_true')
    p.add_argument('--sliced-fourstep',action='store_true')
    p.add_argument('--autosort',action='store_true')
    p.add_argument('--factor',type=int,default=0,help='first FFT length; 0 uses one kernel')
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--input',choices=['random','impulse','constant','tone'],default='random')
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--rounds',type=int,default=5)
    args=p.parse_args()
    n,batch=args.n,args.batch
    torch.npu.set_device(0)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    cpu=torch.randn(batch,n,2)
    if args.input=='impulse':
        cpu.zero_();cpu[:,0,0]=1
    elif args.input=='constant':
        cpu.fill_(1)
    elif args.input=='tone':
        angle=2*math.pi*17*torch.arange(n,dtype=torch.float64)/n
        cpu=torch.stack((angle.cos(),angle.sin()),dim=-1).float().repeat(batch,1,1)
    tuning={'default':{},'multi':{'hfusion_enable_multiple_consumer_fusion':True},
            'vec2':{'enable_auto_vectorize_v2':True},'singlebuffer':{'multibuffer':False},
            'ubsave':{'enable_ubuf_saving':True},'flatten':{'enable_flatten':True}}[args.tuning]
    x=cpu.npu(); y=torch.full_like(x,float('nan')); w=twiddles(n)
    if args.fused_fourstep:
        a=args.factor or 64; b=n//a
        assert n%a==0 and a>=4 and b>=4
        wa,wb=axis_twiddles(n,a,b),axis_twiddles(n,b,a)
        k=torch.arange(n,dtype=torch.int64)
        angle=-2*math.pi*(k//b)*(k%b)/n
        cross=torch.cat((angle.cos(),angle.sin())).float().npu()
        ck=fourstep_fused[(batch,)](x,y,wa,wb,cross,n,a,b,a.bit_length()-1,b.bit_length()-1,args.native_interleave,**tuning)
        runner=ck[(batch,1,1)]
        def launch(): runner(x,y,wa,wb,cross,n,a,b,a.bit_length()-1,b.bit_length()-1,args.native_interleave)
    elif args.sliced_fourstep:
        a=args.factor or 64; b=n//a
        assert n%a==0 and a>=4 and b>=4
        wa,wb=axis_twiddles(n,a,b),axis_twiddles(n,b,a)
        k=torch.arange(n,dtype=torch.int64)
        angle=-2*math.pi*(k//b)*(k%b)/n
        cross=torch.cat((angle.cos(),angle.sin())).float().npu()
        tmp=torch.empty_like(x)
        ca=fourstep_sliced_first[(batch,)](x,tmp,wa,cross,n,a,b,a.bit_length()-1)
        ck=fourstep_sliced_second[(batch,)](tmp,y,wb,n,a,b,b.bit_length()-1)
        ra,rb=ca[(batch,1,1)],ck[(batch,1,1)]
        def launch():
            ra(x,tmp,wa,cross,n,a,b,a.bit_length()-1)
            rb(tmp,y,wb,n,a,b,b.bit_length()-1)
    elif args.autosort:
        ck=fft_autosort[(batch,)](x,y,w,n,n.bit_length()-1)
        runner=ck[(batch,1,1)]
        def launch(): runner(x,y,w,n,n.bit_length()-1)
    elif args.factor:
        a=args.factor; b=n//a
        assert n%a==0 and a>=4 and b>=4
        wa,wb=axis_twiddles(n,a,b),axis_twiddles(n,b,1)
        k=torch.arange(n,dtype=torch.int64)
        angle=-2*math.pi*(k//b)*(k%b)/n
        cross=torch.cat((angle.cos(),angle.sin())).float().npu()
        tmp=torch.empty_like(x)
        da=(4,)*((a.bit_length()-1)//2)+((2,) if (a.bit_length()-1)%2 else ())
        db=(4,)*((b.bit_length()-1)//2)+((2,) if (b.bit_length()-1)%2 else ())
        sa,oa=da+(b,),tuple(reversed(range(len(da))))+(len(da),)
        sb,ob=(a,)+db,(0,)+tuple(reversed(range(1,1+len(db))))
        ca=fourstep_first[(batch,)](x,tmp,wa,cross,n,a,b,a.bit_length()-1,sa,oa)
        ck=fourstep_second[(batch,)](tmp,y,wb,n,a,b,b.bit_length()-1,sb,ob)
        ra,rb=ca[(batch,1,1)],ck[(batch,1,1)]
        def launch():
            ra(x,tmp,wa,cross,n,a,b,a.bit_length()-1,sa,oa)
            rb(tmp,y,wb,n,a,b,b.bit_length()-1,sb,ob)
    else:
        shape=(4,)*((n.bit_length()-1)//2)+((2,) if (n.bit_length()-1)%2 else ())
        order=tuple(reversed(range(len(shape))))
        ck=fft_planar[(batch,)](x,y,w,N=n,LOGN=n.bit_length()-1,SHAPE=shape,ORDER=order)
        runner=ck[(batch,1,1)]
        def launch(): runner(x,y,w,n,n.bit_length()-1,shape,order)
    torch.npu.synchronize()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name,data in ck.asm.items():
        if isinstance(data,str):
            args.output.with_suffix('.'+name).write_text(data)
    ref=torch.fft.fft(torch.view_as_complex(cpu),dim=-1)
    def check():
        got=torch.view_as_complex(y.cpu())
        rel=((got-ref).abs().square().sum()/ref.abs().square().sum()).sqrt().item()
        error=(got-ref).abs().max().item()
        assert torch.allclose(got,ref,atol=3e-4,rtol=3e-5),(rel,error)
        return dict(rel_l2=rel,max_abs_err=error)
    result=check()
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(args.steps): launch()
    y.fill_(float('nan')); g.replay(); torch.npu.synchronize(); check()
    for _ in range(10): g.replay()
    torch.npu.synchronize()
    trials=[]
    for _ in range(args.rounds):
        a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        a.record()
        for _ in range(30): g.replay()
        b.record(); b.synchronize()
        trials.append(a.elapsed_time(b)*1000/30/args.steps/batch)
    result.update(source_sha256=source_sha,tuning=args.tuning,seed=args.seed,input=args.input,
                  all_blocks_parallel=os.environ.get('TRITON_ALL_BLOCKS_PARALLEL'),
                  n=n,batch=batch,factor=args.factor,sliced_fourstep=args.sliced_fourstep,fused_fourstep=args.fused_fourstep,native_interleave=args.native_interleave,autosort=args.autosort,us_per_transform=statistics.median(trials),
                  trials=trials,steps=args.steps,triton=triton.__file__,metadata=str(ck.metadata),
                  replay_pass=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
