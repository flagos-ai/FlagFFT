"""Independent full-FFT checks for the bounded three-matrix layout."""
import math
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.npu_matrix_chain import build
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from test_npu_structured_fft import Pointer, execute_module


def weights(radix, direction, global_inverse):
    n = 255*26*radix
    physical = 256*26*radix
    sign = 1 if direction == 'inverse' and not global_inverse else -1
    first = []
    for tile in range(2):
        f = tile*64+torch.arange(64,dtype=torch.float64)[:,None]
        d = torch.arange(256)[None,:]
        phase = (1 if direction == 'inverse' else -1)*2*math.pi*f*d/255
        first.append(torch.cat((torch.where(d<255,phase.cos(),0).float(),
                                torch.where(d<255,phase.sin(),0).float())).flatten())
    sample = torch.arange(n//255,dtype=torch.float64)[:,None]
    f = torch.arange(256,dtype=torch.float64)[None,:]
    phase = sign*2*math.pi*sample*f/n
    first += [phase.cos().float().flatten(),phase.sin().float().flatten()]
    f = torch.arange(16,dtype=torch.float64)[:,None]
    d = torch.arange(32)[None,:]
    phase = sign*2*math.pi*f*d/26
    middle = torch.cat((torch.where(d<26,phase.cos(),0).float(),
                        torch.where(d<26,phase.sin(),0).float())).flatten()
    padded = 1 << (radix-1).bit_length()
    tiles = math.ceil(radix/64)
    f = torch.arange(tiles*64,dtype=torch.float64).reshape(tiles,64)[None,:,:,None]
    d = torch.arange(padded,dtype=torch.float64)[None,None,None,:]
    shift = torch.arange(26,dtype=torch.float64)[:,None,None,None]/26
    phase = sign*2*math.pi*(f+shift)*d/radix
    co = torch.where((f<radix)&(d<radix),phase.cos(),0).float()
    si = torch.where((f<radix)&(d<radix),phase.sin(),0).float()
    last = torch.cat((torch.cat((co,-si),3),torch.cat((si,co),3)),2).flatten()
    assert sum(x.numel() for x in first) == 65536+2*physical
    return [torch.cat(first),middle,last]


@pytest.mark.parametrize('columns',[128,256])
@pytest.mark.parametrize('radix,mode,batch',[
    (40,'forward',1),(50,'inverse',1),(64,'global_inverse',1),
    (80,'forward',1),(100,'forward',9),(100,'global_inverse',9),
    (112,'inverse',1),(128,'forward',1)])
def test_complete_fft(tmp_path,columns,radix,mode,batch):
    gi = mode == 'global_inverse'
    direction = 'inverse' if gi else mode
    source,specs = build((17,15,13,2,radix),direction,gi,columns)
    path = tmp_path/'chain.py'
    path.write_text(source)
    ns,pid = execute_module(path)
    ns['tl'].dot = lambda a,b,acc=None,**kw: (a.double()@b.double()+(0 if acc is None else acc.double())).float()
    n = 255*26*radix
    torch.manual_seed(20261112)
    x = torch.randn(batch,n,2)
    saved = x.clone()
    current = x
    tables = weights(radix,direction,gi)
    for index,s in enumerate(specs):
        final = index == 2
        guard = torch.full((2*batch*(n if final else s['physical_n'])+128,),21828.5)
        for gx in range(s['grid'][0]):
            for gy in range(math.ceil(batch/s.get('batch_pack',1))):
                pid[:] = [gx,gy,0]
                args = [Pointer(current),Pointer(tables[index]),Pointer(guard,64)]
                if s.get('batch_pack',1)>1:args.append(batch)
                ns[s['name']](*args)
        assert (guard[:64] == 21828.5).all() and (guard[-64:] == 21828.5).all()
        current = guard[64:-64]
    got = torch.view_as_complex(current.reshape_as(x)).cdouble()
    z = torch.view_as_complex(x).cdouble()
    ref = torch.fft.fft(z) if direction == 'forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert (torch.linalg.vector_norm(got-ref,dim=1)/torch.linalg.vector_norm(ref,dim=1)<5e-7).all()
    assert torch.equal(x,saved)


def test_emitter_and_unsupported_geometry(tmp_path):
    options = dict(aligned=True,cube_prefix=True,batch_pack=True,dense_prefix=True)
    for phase in range(3):
        old = emit_vector_dif((17,15,13,2,100),'forward',phase,tmp_path,matrix_chain_columns=128,**options)
        text = Path(old['module_path']).read_text()
        new = emit_vector_dif((17,15,13,2,100),'forward',phase,tmp_path,matrix_chain_columns=256,**options)
        assert old['module_path'] != new['module_path']
        assert Path(old['module_path']).read_text() == text
        assert new['signature'] == '*fp32:16,*fp32:16,*fp32:16'+(',i32' if phase else '')
        assert new['batch_per_block'] == (8 if phase else 1)
        assert new['num_stages'] == (1 if phase==2 else 2)
    for bad in ((17,15,13,2,32),(17,15,13,2,160),(17,15,13,7,100)):
        with pytest.raises(ValueError,match='Matrix chain requires'):build(bad)
    with pytest.raises(ValueError,match='batched dense Cube'):
        emit_vector_dif((17,15,13,2,100),'forward',0,tmp_path,matrix_chain_columns=128)
    with pytest.raises(ValueError,match='Invalid DIF phase'):
        emit_vector_dif((17,15,13,2,100),'forward',3,tmp_path,matrix_chain_columns=128,**options)
