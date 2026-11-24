"""Qualify the generated Cube leaf mapping using CPU matmuls and FFT reference."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import argparse
import torch
from ascend_local_fft_math_check import Pointer,load,store,extract,insert,axis


def weights(n):
    j=torch.arange(n,dtype=torch.float64)
    a=-2*math.pi*j[:,None]*j[None,:]/n
    r,i=a.cos().float(),a.sin().float()
    w=torch.cat((torch.cat((r,-i),1),torch.cat((i,r),1)),0)
    h=w.half();return h,((w-h.float())*4096).half()


def roots(l,c,steps):
    n=l*c;values=[]
    for s in range(steps):
        span=l//2**s;j=(torch.arange(n//2)//c)%(span//2)
        angle=-2*math.pi*j.double()/span
        values.extend((angle.cos().float(),angle.sin().float()))
    return torch.cat(values) if values else torch.zeros(1)


def dot(a,b,acc=None,**kwargs):
    result=a.float()@b.float()
    return result if acc is None else result+acc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tl=SimpleNamespace(constexpr=int,program_id=lambda axis:0,static_range=range,
        arange=torch.arange,reshape=lambda v,s:v.reshape(s),permute=lambda v,*s:v.permute(*s),
        split=lambda v:torch.unbind(v,dim=-1),join=lambda a,b:torch.stack((a,b),-1),
        zeros=lambda s,dtype:torch.zeros(s,dtype=dtype),float32=torch.float32,float16=torch.float16,
        dot=dot,load=load,store=store)
    source=Path(__file__).with_name('ascend_cube_fft.py').read_text().split('\ndef dft_weights')[0]
    code='\n'.join(s for s in source.splitlines() if not s.startswith(('import ','from ','@')))
    ns=dict(tl=tl,al=SimpleNamespace(extract_slice=extract,insert_slice=insert),fft_axis_sliced=axis)
    exec(compile(code,'cube-cpu-model','exec'),ns)
    rows=[]
    for p in (1,2):
      for terms in (3,4):
        n,a,b,leaf=2048,32,64,16
        x=torch.randn(p,n,2);y=torch.full_like(x,float('nan'))
        ah,al=weights(leaf);bh,bl=weights(leaf)
        ar,br=roots(a,p*b,1),roots(b,p*a,2)
        k=torch.arange(n);angle=-2*math.pi*(k//b)*(k%b)/n
        cross=torch.cat((angle.cos(),angle.sin()))
        ns['cube_fourstep'](Pointer(x),Pointer(y),Pointer(ah),Pointer(al),Pointer(bh),Pointer(bl),
            Pointer(cross),Pointer(torch.zeros(1)),Pointer(torch.zeros(1)),Pointer(ar),Pointer(br),
            p,n,a,b,terms,True,True,5,6,p,leaf,leaf,1,2)
        ref=torch.fft.fft(torch.view_as_complex(x),dim=-1);got=torch.view_as_complex(y)
        rel=((got-ref).abs().square().sum()/ref.abs().square().sum()).sqrt().item()
        assert torch.allclose(got,ref,atol=3e-4,rtol=3e-5),rel
        rows.append(dict(pack=p,terms=terms,rel_l2=rel))
    args.output.write_text(json.dumps(dict(status='pass',cases=rows),indent=2)+'\n')
    print('PASS: Cube leaf mapping, precision compensation and natural output order')


if __name__=='__main__':main()
