"""Two-pass tiled four-step: FFT+twiddle+transpose, then FFT+natural store."""
import triton
import triton.language as tl
from ascend_fourstep_probe import fft_axis_sliced


@triton.jit
def tiled_axis(X,Y,W,CROSS,NB:tl.constexpr,N:tl.constexpr,L:tl.constexpr,
               WIDTH:tl.constexpr,C:tl.constexpr,FIRST:tl.constexpr,LOG:tl.constexpr):
    pid=tl.program_id(0)
    batch=pid//(WIDTH//C)
    tile=pid%(WIDTH//C)
    row=tl.arange(0,L)
    col=tile*C+tl.arange(0,C)
    pair=tl.arange(0,2)
    offsets=(batch*N+row[:,None]*WIDTH+col[None,:])*2
    raw=tl.load(X+offsets[:,:,None]+pair[None,None,:])
    r,i=tl.split(tl.reshape(raw,(L*C,2)))
    r,i=fft_axis_sliced(r,i,W,L*C,L,C,LOG)
    r=tl.reshape(r,(L,C));i=tl.reshape(i,(L,C))
    if FIRST:
        cr=tl.load(CROSS+row[:,None]*WIDTH+col[None,:])
        ci=tl.load(CROSS+N+row[:,None]*WIDTH+col[None,:])
        rr,ii=r*cr-i*ci,r*ci+i*cr
        rr=tl.reshape(tl.trans(rr),(L*C,));ii=tl.reshape(tl.trans(ii),(L*C,))
        raw=tl.reshape(tl.join(rr,ii),(C,L,2))
        dest=(batch*N+col[:,None]*L+row[None,:])*2
    else:
        raw=tl.reshape(tl.join(tl.reshape(r,(L*C,)),tl.reshape(i,(L*C,))),(L,C,2))
        dest=offsets
    tl.store(Y+dest[:,:,None]+pair[None,None,:],raw)
