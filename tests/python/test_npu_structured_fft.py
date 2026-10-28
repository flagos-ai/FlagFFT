"""CPU execution of the emitted FFT bodies, including tables, layouts and inverse."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_structured import emit_structured


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


def roots(length, columns):
    n = length * columns
    rows = []
    for stage in range((length.bit_length()-1)//2):
        span = length // 4**stage
        j = (torch.arange(n//4)//columns) % (span//4)
        for branch in (1, 2, 3):
            angle = -2*math.pi*branch*j.double()/span
            rows.extend((angle.cos().float(), angle.sin().float()))
    return torch.cat(rows)


def execute_module(path):
    pid = [0, 0]
    def load(ptr, mask=True, other=0):
        index=torch.as_tensor(ptr.offset)
        mask=torch.broadcast_to(torch.as_tensor(mask),index.shape)
        return torch.where(mask,ptr.data[index.clamp(0,ptr.data.numel()-1)],other)
    def store(ptr, value, mask=True):
        index=torch.as_tensor(ptr.offset)
        mask=torch.broadcast_to(torch.as_tensor(mask),index.shape)
        ptr.data[index[mask]]=torch.broadcast_to(value,index.shape)[mask]
    def extract(v, offsets, sizes, strides):
        return v[tuple(slice(o,o+n*s,s) for o,n,s in zip(offsets,sizes,strides))]
    def insert(v, x, offsets, sizes, strides):
        out = v.clone()
        out[tuple(slice(o,o+n*s,s) for o,n,s in zip(offsets,sizes,strides))] = x
        return out
    tl = SimpleNamespace(constexpr=int, program_id=lambda axis:torch.tensor(pid[axis]), static_range=range,
        arange=torch.arange,reshape=lambda v,s:v.reshape(s),permute=lambda v,*s:v.permute(*s),
        trans=lambda v:v.transpose(-1,-2),split=lambda v:torch.unbind(v,dim=-1),
        join=lambda a,b:torch.stack((a,b),-1),zeros=lambda s,dtype:torch.zeros(s,dtype=dtype),
        float32=torch.float32,int64=torch.int64,load=load,store=store,
        gather=lambda v,index,axis:torch.gather(v,axis,index.long()),
        minimum=lambda a,b:torch.minimum(torch.as_tensor(a),torch.as_tensor(b)),where=torch.where)
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            node.decorator_list = []
    ns = dict(tl=tl,al=SimpleNamespace(extract_slice=extract,insert_slice=insert))
    exec(compile(tree,str(path),'exec'),ns)
    return ns,pid


@pytest.mark.parametrize('n',[1024,2048])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_fused_fft_generated_body(tmp_path,n,direction):
    meta = emit_structured(length=n,direction=direction,out_dir=tmp_path)
    ns,pid = execute_module(meta['module_path'])
    torch.manual_seed(20261028)
    x = torch.randn(3,n,2)
    saved = x.clone()
    guarded = torch.full((x.numel()+128,),31415.25)
    y = Pointer(guarded,64)
    a,b = 32,n//32
    k = torch.arange(n)
    angle = -2*math.pi*(k//b)*(k%b)/n
    cross = torch.cat((angle.cos(),angle.sin()))
    for i in range(3):
        pid[0] = i
        ns[meta['kernel_name']](Pointer(x),y,Pointer(roots(a,b)),Pointer(roots(b,a)),Pointer(cross),3)
    got = torch.view_as_complex(guarded[64:-64].reshape(3,n,2))
    ref = torch.fft.fft(torch.view_as_complex(x)) if direction=='forward' else torch.fft.ifft(torch.view_as_complex(x))*n
    assert torch.allclose(got,ref,atol=3e-4,rtol=3e-5)
    assert torch.equal(x,saved)
    assert (guarded[:64]==31415.25).all() and (guarded[-64:]==31415.25).all()


@pytest.mark.parametrize('tile',[8,16])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_two_pass_tables_and_store_contract(tmp_path,tile,direction):
    torch.manual_seed(20261028)
    n,a,b,batch = 16384,128,128,2
    x = torch.randn(batch,n,2)
    saved = x.clone()
    temp,y = torch.empty_like(x),torch.full_like(x,float('nan'))
    # The private fused axis uses contiguous planar coefficients in k1,j2 order.
    k1,j2 = torch.arange(a)[:,None],torch.arange(b)[None,:]
    angle = (-1 if direction=='forward' else 1)*2*math.pi*j2.double()*k1.double()/n
    twiddle = torch.cat((angle.cos().flatten(),angle.sin().flatten())).float()
    for first in (True,False):
        meta = emit_structured(length=a,direction=direction,out_dir=tmp_path,first=first,n1=a,n2=b,tile=tile)
        ns,pid = execute_module(meta['module_path'])
        for transform in range(batch):
            pid[1] = transform
            for col in range(a//tile):
                pid[0] = col
                args = (Pointer(x),Pointer(twiddle),Pointer(temp),Pointer(roots(a,tile)),batch) if first else (Pointer(temp),Pointer(y),Pointer(roots(b,tile)),batch)
                ns[meta['kernel_name']](*args)
    got = torch.view_as_complex(y)
    ref = torch.fft.fft(torch.view_as_complex(x)) if direction=='forward' else torch.fft.ifft(torch.view_as_complex(x))*n
    assert torch.allclose(got,ref,atol=3e-4,rtol=3e-5)
    assert torch.equal(x,saved)
