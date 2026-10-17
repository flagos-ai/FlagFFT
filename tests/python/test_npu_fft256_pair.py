"""CPU table ABI, arithmetic, guards and backend scope for the 256-point port."""
import ast
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from flagfft_codegen.kernels_npu_fft256 import build_npu_fft256_pair_source, emit_npu_fft256_pair
from flagfft_codegen.kernels_npu_radix4 import build_npu_radix4_pair_source
from test_npu_grouped_fft64 import Language, Pointer, Index, insert_slice

class GridLanguage(Language):
    programs=1
    def num_programs(self,axis):
        assert axis==0
        return self.programs

@pytest.fixture(scope="module")
def native_table_dump(tmp_path_factory):
    compiler=shutil.which("g++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable")
    repo=Path(__file__).resolve().parents[2]
    code=(repo/"src/codegen/compiler.cpp").read_text()
    start=code.index("  void build_npu_aiv_fft256_pair_group8_tables(")
    end=code.index("  void build_npu_aiv_fft64_group_tables(",start)
    prelude="""#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
struct FFTRequest { std::string direction; };
enum class NpuAivFFT256Mode { Complex=0, RealForward=1, RealInverse=2 };
"""
    main="""int main(int argc,char**argv) {
  if(argc!=4)return 2;
  FFTRequest request{std::string(argv[3])=="1" ? "inverse":"forward"};
  std::vector<uint32_t> index;std::vector<float> twiddle;
  if(std::string(argv[1])=="store")
    build_npu_aiv_fft256_pair_radix4_group8_tables(request,index,twiddle);
  else
    build_npu_aiv_fft256_pair_group8_tables(request,index,twiddle,false,
        static_cast<NpuAivFFT256Mode>(std::atoi(argv[2])));
  uint64_t sizes[2]={index.size(),twiddle.size()};
  std::cout.write(reinterpret_cast<const char*>(sizes),sizeof(sizes));
  std::cout.write(reinterpret_cast<const char*>(index.data()),index.size()*4);
  std::cout.write(reinterpret_cast<const char*>(twiddle.data()),twiddle.size()*4);
}
"""
    folder=tmp_path_factory.mktemp("fft256_original_tables")
    source,exe=folder/"tables.cpp",folder/"tables"
    source.write_text(prelude+code[start:end]+main)
    subprocess.run([compiler,"-std=c++17","-O2",str(source),"-o",str(exe)],check=True,capture_output=True,text=True)
    return exe

@pytest.mark.parametrize("mode,inverse",[(0,False),(0,True),(1,False),(2,True)])
@pytest.mark.parametrize("count,grid",[(8,1),(328,1),(328,32),(4096,32)])
def test_pair_keeps_native_tables_and_real_grid(native_table_dump,mode,inverse,count,grid):
    if mode==0:grid=count//8
    raw=subprocess.check_output([str(native_table_dump),"pair",str(mode),str(int(inverse))])
    ni,nt=struct.unpack("<QQ",raw[:16])
    index=np.frombuffer(raw,np.uint32,ni,16).copy()
    twiddle=np.frombuffer(raw,np.float32,nt,16+ni*4).copy()
    rng=np.random.default_rng(20261018)
    real=rng.standard_normal((count,256)).astype(np.float32)
    if mode==1:data=real;reference=np.fft.rfft(real,axis=1).astype(np.complex64).view(np.float32)
    elif mode==2:
        z=np.fft.rfft(real,axis=1).astype(np.complex64)
        data=z.view(np.float32);reference=(np.fft.irfft(z,n=256,axis=1)*256).astype(np.float32)
    else:
        z=(real+1j*rng.standard_normal(real.shape)).astype(np.complex64)
        data=z.view(np.float32)
        reference=(np.fft.ifft(z,axis=1)*256 if inverse else np.fft.fft(z,axis=1)).astype(np.complex64).view(np.float32)
    name,body=build_npu_fft256_pair_source(mode)
    tree=ast.parse(body);tree.body[0].decorator_list=[]
    language=GridLanguage();language.programs=grid
    ns=dict(tl=language,tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)))
    exec(compile(tree,name,"exec"),ns)
    output=np.full(reference.size+128,31415.25,np.float32)
    for pid in range(grid):
        language.pid=pid
        ns[name](Pointer(data.reshape(-1)),Pointer(output,64),Pointer(index),Pointer(twiddle),count)
    actual=output[64:-64].reshape(reference.shape).astype(np.float64)
    expected=reference.astype(np.float64)
    assert np.isfinite(actual).all()
    assert np.max(np.linalg.norm(actual-expected,axis=1)/np.maximum(np.linalg.norm(expected,axis=1),1e-30))<1e-6
    assert np.all(output[:64]==31415.25) and np.all(output[-64:]==31415.25)

@pytest.mark.parametrize("inverse",[False,True])
@pytest.mark.parametrize("stride,offset,count",[(16,0,8),(24,8,320),(256,248,328),(65536,65528,16)])
def test_radix4_store_retains_global_row_stride(native_table_dump,inverse,stride,offset,count):
    raw=subprocess.check_output([str(native_table_dump),"store","0",str(int(inverse))])
    ni,nt=struct.unpack("<QQ",raw[:16])
    index=np.frombuffer(raw,np.uint32,ni,16).copy()
    twiddle=np.frombuffer(raw,np.float32,nt,16+ni*4).copy()
    rng=np.random.default_rng(20261018)
    z=(rng.standard_normal((count,256))+1j*rng.standard_normal((count,256))).astype(np.complex64)
    ref=np.fft.ifft(z,axis=1)*256 if inverse else np.fft.fft(z,axis=1)
    name,body=build_npu_radix4_pair_source(length=256,group=8,mode=0,transposed_store=True)
    tree=ast.parse(body);tree.body[0].decorator_list=[]
    language=Language()
    ns=dict(tl=language,tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)))
    exec(compile(tree,name,"exec"),ns)
    size=((offset+count+stride-1)//stride)*stride*512
    output=np.full(size+128,31415.25,np.float32)
    for pid in range(count//8):
        language.pid=pid
        ns[name](Pointer(z.view(np.float32).reshape(-1)),Pointer(output,64),Pointer(index),Pointer(twiddle),count,
            np.asarray(stride).view(Index),np.asarray(offset).view(Index))
    value=output[64:-64].view(np.complex64).reshape(-1,256,stride)
    for row in range(count):
        b,c=divmod(offset+row,stride)
        assert np.linalg.norm(value[b,:,c]-ref[row])/np.linalg.norm(ref[row])<1e-6
    assert np.all(output[:64]==31415.25) and np.all(output[-64:]==31415.25)

@pytest.mark.parametrize("backend,dtype",[("npu","complex64"),("npu","complex128"),("cuda","complex64"),("ix","complex64"),("maca","complex64"),("musa","complex64"),("hcu","complex64"),("ppu","complex64")])
@pytest.mark.parametrize("mode,inverse,store",[(0,False,False),(0,True,True),(1,False,False),(2,True,False)])
def test_fft256_cli_scope_and_abi(tmp_path,backend,dtype,mode,inverse,store):
    repo=Path(__file__).resolve().parents[2]
    command=[sys.executable,"-m","flagfft_codegen.jit_source","--kernel","npu_fft256_pair",
        "--target",f"{backend}:source-contract:32","--dtype",dtype,"--direction","inverse" if inverse else "forward",
        "--npu-fft-mode",str(mode),"--out-dir",str(tmp_path)]
    if store:command+=["--npu-fft-radix4-pair"]
    result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(repo/"python"),PYTHONDONTWRITEBYTECODE="1"))
    if backend!="npu" or dtype!="complex64":
        assert result.returncode!=0 and "NPU FP32" in result.stderr
        return
    assert result.returncode==0,result.stderr
    meta=json.loads(result.stdout)
    assert meta["signature"]=="*fp32:16,*fp32:16,*u32:16,*fp32:16,i32"+(",i32,i32" if store else "")
    assert meta["batch_per_block"]==8 and meta["radix4_store"]==store
    body=Path(meta["module_path"]).read_text()
    assert "tle.dsa.insert_slice" in body
    assert ("tl.num_programs(0)" in body)==bool(mode)

@pytest.mark.parametrize("mode,direction,store",[(3,"forward",False),(1,"inverse",False),(2,"forward",False),(1,"forward",True),(0,"invalid",False)])
def test_fft256_rejects_unqualified_modes(tmp_path,mode,direction,store):
    with pytest.raises(ValueError):
        emit_npu_fft256_pair(mode=mode,radix4_store=store,direction=direction,out_dir=tmp_path)
