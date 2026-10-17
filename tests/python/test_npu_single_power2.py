"""CPU arithmetic, table-layout, ABI and backend scope for single-row FFTs."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from flagfft_codegen.kernels_npu_power2 import build_npu_single_power2_source
from test_npu_grouped_fft64 import Language, Pointer, insert_slice

def tables(n, inverse=False):
    stages = n.bit_length() - 1
    grouped = n == 128
    a = 3 * n if grouped else n
    b = a + stages * n
    output = n if grouped else b + stages * n
    size = b + stages * n if grouped else output + 2 * n
    index = np.zeros(size, np.uint32)
    twiddle = np.zeros(2 * stages * n, np.float32)
    sign = 1 if inverse else -1
    for k in range(n):
        rev = int(f'{k:0{stages}b}'[::-1], 2)
        index[k] = rev * 2 * 4
        index[output + 2 * k] = k * 4
        index[output + 2 * k + 1] = (n + k) * 4
    for stage in range(stages):
        length = 1 << (stage + 1)
        half = length // 2
        for k in range(n):
            offset = k % length
            left = (k // length) * length + offset % half
            index[a + stage * n + k] = left * 4
            index[b + stage * n + k] = (left + half) * 4
            angle = sign * 2 * np.pi * (offset % half) / length
            negate = np.float32(-1 if offset >= half else 1)
            real = stage * n if grouped else stage * 2 * n
            imag = (stages + stage) * n if grouped else (stage * 2 + 1) * n
            twiddle[real + k] = negate * np.float32(np.cos(angle))
            twiddle[imag + k] = negate * np.float32(np.sin(angle))
    return index, twiddle


def extract_slice(value, offsets, sizes, strides):
    assert strides == [1]
    return value[offsets[0]:offsets[0]+sizes[0]]


@pytest.mark.parametrize("n",[128,1024,2048])
@pytest.mark.parametrize("inverse",[False,True])
@pytest.mark.parametrize("count",[1,3,17])
def test_single_power2_keeps_native_table_mapping(n,inverse,count):
    rng=np.random.default_rng(20261018)
    z=(rng.standard_normal((count,n))+1j*rng.standard_normal((count,n))).astype(np.complex64)
    reference=np.fft.ifft(z,axis=1)*n if inverse else np.fft.fft(z,axis=1)
    index,twiddle=tables(n,inverse)
    name,body=build_npu_single_power2_source(n)
    tree=ast.parse(body);tree.body[0].decorator_list=[]
    language=Language()
    ns=dict(tl=language,tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice,extract_slice=extract_slice)))
    exec(compile(tree,name,"exec"),ns)
    guard=64
    data=z.view(np.float32).reshape(-1)
    output=np.full(data.size+guard*2,31415.25,np.float32)
    for pid in range(count):
        language.pid=pid
        ns[name](Pointer(data),Pointer(output,guard),Pointer(index),Pointer(twiddle),count)
    value=output[guard:-guard].view(np.complex64).reshape(z.shape)
    error=value.astype(np.complex128)-reference
    assert np.linalg.norm(error)/np.linalg.norm(reference)<2e-6
    assert np.max(np.abs(error))/np.max(np.abs(reference))<2e-6
    assert np.all(output[:guard]==31415.25) and np.all(output[-guard:]==31415.25)


@pytest.mark.parametrize("n",[0,64,256,512,4096])
def test_single_power2_rejects_unqualified_layouts(n):
    with pytest.raises(ValueError):
        build_npu_single_power2_source(n)


@pytest.mark.parametrize("backend,dtype",[("npu","complex64"),("npu","complex128"),("cuda","complex64"),("ix","complex64"),("maca","complex64"),("musa","complex64"),("hcu","complex64"),("ppu","complex64")])
@pytest.mark.parametrize("n",[128,1024,2048])
def test_single_power2_cli_scope_and_abi(tmp_path,backend,dtype,n):
    repo=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,"-m","flagfft_codegen.jit_source","--kernel","npu_single_power2",
        "--length",str(n),"--direction","forward","--target",f"{backend}:source-contract:32",
        "--dtype",dtype,"--out-dir",str(tmp_path)],text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH=str(repo/"python"),PYTHONDONTWRITEBYTECODE="1"))
    if backend!="npu" or dtype!="complex64":
        assert result.returncode!=0 and "NPU FP32" in result.stderr
        return
    assert result.returncode==0,result.stderr
    metadata=json.loads(result.stdout)
    assert metadata["signature"]=="*fp32:16,*fp32:16,*i32:16,*fp32:16,i32"
    assert metadata["length"]==n and metadata["batch_per_block"]==1
    assert metadata["num_warps"]==4 and metadata["num_stages"]==1
    source=Path(metadata["module_path"]).read_text()
    assert "tle.dsa.insert_slice" in source
    assert ("tle.dsa.extract_slice" in source)==(n==128)
