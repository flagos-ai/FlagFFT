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

    def __getitem__(self, key):
        return Pointer(self.data, self.offset[key])

    def reshape(self, shape):
        return Pointer(self.data, self.offset.reshape(shape))


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
        float32=torch.float32,int64=torch.int64,int32=torch.int32,int1=torch.bool,
        full=lambda shape,value,dtype:torch.full(shape,value,dtype=dtype),
        zeros_like=torch.zeros_like,range=range,load=load,store=store,
        sin=torch.sin,cos=torch.cos,
        gather=lambda v,index,axis:torch.gather(v,axis,index.long()),
        minimum=lambda a,b:torch.minimum(torch.as_tensor(a),torch.as_tensor(b)),
        maximum=lambda a,b:torch.maximum(torch.as_tensor(a),torch.as_tensor(b)),where=torch.where)
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef)]
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
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref) < 1e-6
    assert (got-ref).abs().max()/ref.abs().max() < 2e-6
    assert torch.equal(x,saved)
    assert (guarded[:64]==31415.25).all() and (guarded[-64:]==31415.25).all()


@pytest.mark.parametrize('a,b,tile', [
    (512,512,(8,8)), (512,1024,(8,4)),
    (1024,512,(4,8)), (512,2048,(8,2)),
])
@pytest.mark.parametrize('direction', ['forward','inverse'])
def test_dense_second_fourstep(tmp_path, monkeypatch, a, b, tile, direction):
    monkeypatch.setenv('FLAGFFT_NPU_FOURSTEP_DENSE_SECOND','1')
    test_two_pass_tables_and_store_contract(tmp_path,a,b,tile,True,direction)


def axis_roots(length, columns, recursive):
    if not recursive or length < 256:
        return roots(length, columns)
    a = 16 if length == 256 else 64 if length == 2048 else 32
    b = length // a
    k = torch.arange(length * columns)
    phase = -2*math.pi*(k//(b*columns)).double()*((k//columns)%b)/length
    return torch.cat((roots(a, b*columns), roots(b, a*columns),
                      phase.cos().float(), phase.sin().float()))


@pytest.mark.parametrize('a,b,tile,recursive',[
    (128,128,8,False),(128,128,16,False),(128,64,(16,32),False),(16,32,4,False),
    (64,128,8,False),(128,256,8,False),(512,1024,2,False),
    (512,1024,4,True),(1024,512,4,True),(512,2048,2,True),(512,1024,(8,4),True)])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_two_pass_tables_and_store_contract(tmp_path,a,b,tile,recursive,direction):
    torch.manual_seed(20261028)
    n,batch = a*b,1
    x = torch.randn(batch,n,2)
    saved = x.clone()
    temp,y = torch.empty_like(x),torch.full_like(x,float('nan'))
    # The private fused axis uses contiguous planar coefficients in k1,j2 order.
    k1,j2 = torch.arange(a)[:,None],torch.arange(b)[None,:]
    angle = (-1 if direction=='forward' else 1)*2*math.pi*j2.double()*k1.double()/n
    twiddle = torch.cat((angle.cos().flatten(),angle.sin().flatten())).float()
    for first in (True,False):
        inner = tile[0 if first else 1] if isinstance(tile, tuple) else tile
        meta = emit_structured(length=a,direction=direction,out_dir=tmp_path,first=first,n1=a,n2=b,tile=inner,recursive=recursive)
        ns,pid = execute_module(meta['module_path'])
        for transform in range(batch):
            pid[1] = transform
            for col in range((b if first else a)//inner):
                pid[0] = col
                args = (Pointer(x),Pointer(twiddle),Pointer(temp),Pointer(axis_roots(a,inner,recursive)),batch) if first else (Pointer(temp),Pointer(y),Pointer(axis_roots(b,inner,recursive)),batch)
                ns[meta['kernel_name']](*args)
    got = torch.view_as_complex(y)
    ref = torch.fft.fft(torch.view_as_complex(x)) if direction=='forward' else torch.fft.ifft(torch.view_as_complex(x))*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref) < 1e-6
    assert (got-ref).abs().max()/ref.abs().max() < 2e-6
    assert torch.equal(x,saved)
