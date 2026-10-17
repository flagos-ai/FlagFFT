"""Preserve complex arithmetic, n/m ABI, padding, masks and backend isolation."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.emit import _bluestein_kernel_source
from flagfft_codegen.kernels_npu_complex_io import build_npu_bluestein_source
from test_npu_grouped_fft64 import Language, Pointer, insert_slice
from test_npu_transpose2d import Index

KINDS=('bluestein_prepare','bluestein_pointwise','bluestein_finalize')
SETTINGS=[(n,m,b) for n,m in ((1,2),(17,45),(31,64),(127,256),(257,1024),
    (1009,2048),(4099,16384)) for b in (1,3)]+[(46189,131072,1)]

class Grid(Language):
    ids=(0,0)
    def program_id(self,axis):return np.asarray(self.ids[axis]).view(Index)

@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('n,m,batch',SETTINGS)
def test_vector_complex_transfers_preserve_original_bits(kind,n,m,batch):
    original=_bluestein_kernel_source(kind,'complex64')
    replacement=_bluestein_kernel_source(kind,'complex64',npu_vector_io=True)
    assert replacement==build_npu_bluestein_source(kind)
    assert original[2]==replacement[2]
    rng=np.random.default_rng(20261018)
    point=kind=='bluestein_pointwise';final=kind=='bluestein_finalize'
    input_size=batch*(m if point or final else n)*2
    table_size=(m if point else n)*2
    output_size=batch*(n if final else m)*2
    source=rng.standard_normal(input_size).astype(np.float32)
    table=rng.standard_normal(table_size).astype(np.float32)
    outputs=[]
    for name,body,arg_names in (original,replacement):
        tree=ast.parse(body);tree.body[0].decorator_list=[]
        language=Grid()
        namespace={'tl':language,'tle':SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)),
            '_cmul':lambda ar,ai,br,bi:(ar*br-ai*bi,ar*bi+ai*br)}
        exec(compile(tree,name,'exec'),namespace)
        output=np.full(output_size+128,31415.25,np.float32)
        args=[Pointer(source),Pointer(table),Pointer(output,64)]+([m,batch] if point else [n,m,batch])
        for y in range(batch):
            for x in range(((n if final else m)+255)//256+1):
                language.ids=(x,y);namespace[name](*args)
        assert np.all(output[:64]==31415.25) and np.all(output[-64:]==31415.25)
        outputs.append(output)
    np.testing.assert_array_equal(outputs[0].view(np.uint32),outputs[1].view(np.uint32))


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('backend,dtype', [('npu','complex64'),('cuda','complex64'),
    ('ix','complex64'),('maca','complex64'),('npu','complex128')])
def test_vector_complex_cli_is_scoped_to_npu_fp32(tmp_path,kind,backend,dtype):
    repo=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-m','flagfft_codegen.jit_source','--kernel',kind,
        '--bluestein-n','257','--bluestein-m','1024','--dtype',dtype,
        '--target',f'{backend}:source-contract:32','--npu-vector-complex-io','--out-dir',str(tmp_path)]
    result=subprocess.run(command,text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH=str(repo/'python'),PYTHONDONTWRITEBYTECODE='1'))
    if backend!='npu' or dtype!='complex64':
        assert result.returncode!=0 and 'NPU FP32 Bluestein' in result.stderr
        return
    assert result.returncode==0,result.stderr
    metadata=json.loads(result.stdout)
    assert metadata['npu_vector_complex_io'] and metadata['block']==256
    assert metadata['kernel_name'].endswith('_aos_kernel')
    assert metadata['signature'].endswith('i64,i32' if kind=='bluestein_pointwise' else 'i64,i64,i32')
    source=Path(metadata['module_path']).read_text()
    assert 'batch = tl.program_id(1).to(tl.int64)' in source
    assert 'tle.dsa.insert_slice' in source


def test_vector_complex_source_rejects_fp64_and_unknown_kind():
    with pytest.raises(ValueError,match='NPU FP32'):
        _bluestein_kernel_source('bluestein_prepare','complex128',npu_vector_io=True)
    with pytest.raises(ValueError,match='Unsupported'):
        build_npu_bluestein_source('leaf')
