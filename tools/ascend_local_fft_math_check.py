"""Execute generated codelets on CPU tensors and compare with independent FFT."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from ascend_local_fft_codegen import generate


class Pointer:
    def __init__(self,data,offset=0):
        self.data,self.offset=data.reshape(-1),offset
    def __add__(self,offset):
        return Pointer(self.data,self.offset+offset)


def load(ptr,mask=True,other=0):
    index=torch.as_tensor(ptr.offset)
    mask=torch.broadcast_to(torch.as_tensor(mask),index.shape)
    return torch.where(mask,ptr.data[index.clamp(0,ptr.data.numel()-1)],other)


def store(ptr,value,mask=True):
    index=torch.as_tensor(ptr.offset)
    mask=torch.broadcast_to(torch.as_tensor(mask),index.shape)
    ptr.data[index[mask]]=torch.broadcast_to(value,index.shape)[mask]


def extract(v,offsets,sizes,strides):
    return v[tuple(slice(o,o+n*s,s) for o,n,s in zip(offsets,sizes,strides))]


def insert(v,x,offsets,sizes,strides):
    out=v.clone()
    out[tuple(slice(o,o+n*s,s) for o,n,s in zip(offsets,sizes,strides))]=x
    return out


def axis(r,i,w,n,l,c,log):
    v=torch.complex(r,i).reshape(-1,l,c)
    v=torch.fft.fft(v,dim=1).flatten()
    return v.real,v.imag


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import json,math
    rows=[]
    tl=SimpleNamespace(constexpr=int,program_id=lambda axis:0,arange=torch.arange,
        reshape=lambda v,s:v.reshape(s),permute=lambda v,*s:v.permute(*s),
        split=lambda v:torch.unbind(v,dim=-1),float32=torch.float32,
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),load=load,store=store)
    al=SimpleNamespace(extract_slice=extract,insert_slice=insert)
    for n,a in [(256,16),(2048,32)]:
      for p in [1,2,4]:
       for both in [False,True]:
        for radix in [2,4]:
            name,source=generate(n=n,factor=a,pack=p,both=both,radix=radix)
            code='\n'.join(s for s in source.splitlines() if not s.startswith(('import ','from ','@')))
            namespace=dict(tl=tl,al=al,fft_axis_sliced=axis)
            exec(compile(code,name,'exec'),namespace)
            x=torch.randn(p,n,2);y=torch.full_like(x,float('nan'))
            k=torch.arange(n);angle=-2*math.pi*(k//(n//a))*(k%(n//a))/n
            cross=torch.cat((angle.cos(),angle.sin()))
            namespace[name](Pointer(x),Pointer(y),Pointer(cross),Pointer(torch.zeros(1)),p)
            ref=torch.fft.fft(torch.view_as_complex(x),dim=-1)
            got=torch.view_as_complex(y)
            error=(got-ref).abs().max().item()
            rel=((got-ref).abs().square().sum()/ref.abs().square().sum()).sqrt().item()
            assert torch.allclose(got,ref,atol=3e-4,rtol=3e-5),(n,p,both,radix,rel,error)
            rows.append(dict(n=n,factor=a,pack=p,both=both,radix=radix,rel_l2=rel))
    args.output.write_text(json.dumps(dict(status='pass',cases=rows),indent=2)+'\n')
    print('PASS:',len(rows),'CPU codelet/layout comparisons')


if __name__=='__main__': main()
