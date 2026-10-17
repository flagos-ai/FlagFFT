"""Bounded fused Stockham: original-order maps, math, tail groups and scope."""
import ast,json,math,os,shutil,struct,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from flagfft_codegen.kernels_npu_fused_stockham import (
    fused_stockham_layout,build_npu_fused_stockham_source,emit_npu_fused_stockham)
from flagfft_codegen.metadata import _module_source
from test_npu_grouped_fft64 import Language,Pointer,Index,insert_slice

SETTINGS=[(12,(3,4)),(15,(5,3)),(16,(16,)),(19,(19,)),(20,(5,4)),(24,(8,3)),
    (32,(8,4)),(48,(12,2,2)),(60,(5,12)),(64,(8,8)),(96,(12,8)),
    (128,(8,4,4)),(209,(19,11)),(221,(17,13)),(256,(16,16))]
def group_size(n):
    group=1
    while group<32 and n*group*2<=2048:group*=2
    return group

class Grid(Language):
    programs=1
    minimum=staticmethod(np.minimum)
    maximum=staticmethod(np.maximum)
    where=staticmethod(np.where)
    def num_programs(self,axis):
        assert axis==0
        return self.programs

def local_range(*args):
    return (np.asarray(k).view(Index) for k in range(*args))

@pytest.fixture(scope="module")
def host_tables(tmp_path_factory):
    compiler=shutil.which("g++")
    if not compiler:pytest.skip("Host C++ compiler unavailable")
    repo=Path(__file__).resolve().parents[2]
    source=(repo/"src/codegen/compiler.cpp").read_text()
    start=source.index("  void build_npu_fused_stockham_tables(")
    end=source.index("  void build_npu_aiv_fft64_group_tables(",start)
    header="""#include <algorithm>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
"""
    main="""int main(int argc,char**argv) {
  if(argc!=5)return 2;
  std::vector<int64_t> factors;
  std::string text(argv[4]);std::size_t start=0;
  do {
    auto end=text.find(',',start);
    factors.push_back(std::stoll(text.substr(start,end-start)));
    if(end==std::string::npos)break;
    start=end+1;
  }while(true);
  std::vector<uint32_t> index;std::vector<float> twiddle;
  build_npu_fused_stockham_tables(std::stoll(argv[1]),factors,
      std::stoll(argv[2]),std::string(argv[3])=="1",index,twiddle);
  uint64_t sizes[2]={index.size(),twiddle.size()};
  std::cout.write(reinterpret_cast<const char*>(sizes),sizeof(sizes));
  std::cout.write(reinterpret_cast<const char*>(index.data()),index.size()*4);
  std::cout.write(reinterpret_cast<const char*>(twiddle.data()),twiddle.size()*4);
}
"""
    folder=tmp_path_factory.mktemp("npu_fused_stockham_host_tables")
    cpp,exe=folder/"tables.cpp",folder/"tables"
    cpp.write_text(header+source[start:end]+main)
    subprocess.run([compiler,"-std=c++17","-O2",str(cpp),"-o",str(exe)],check=True,capture_output=True,text=True)
    return exe

@pytest.mark.parametrize("n,factors",SETTINGS)
@pytest.mark.parametrize("inverse",[False,True])
@pytest.mark.parametrize("mode",["tail","stride"])
def test_fused_body_and_host_tables(host_tables,n,factors,inverse,mode):
    group=group_size(n);p=fused_stockham_layout(n,factors,group)
    raw=subprocess.check_output([str(host_tables),str(n),str(group),str(int(inverse)),','.join(map(str,factors))])
    ni,nt=struct.unpack("<QQ",raw[:16])
    index=np.frombuffer(raw,np.uint32,ni,16).copy()
    twiddle=np.frombuffer(raw,np.float32,nt,16+ni*4).copy()
    np.testing.assert_array_equal(index,np.asarray(p["indices"],np.uint32))
    expected_tw=[];span=1;sign=1 if inverse else -1
    for r in factors:
        if span>1:
            for d in range(r):
                for k in range(span):
                    a=sign*2*math.pi*d*k/(r*span)
                    expected_tw.extend((math.cos(a),math.sin(a)))
        span*=r
    np.testing.assert_array_equal(twiddle,np.asarray(expected_tw or [0],np.float32))
    count,grid=(3,1) if mode=="tail" else (group*3+5,2)
    direction="inverse" if inverse else "forward"
    name,body=build_npu_fused_stockham_source(n,factors,group,direction)
    deps=tuple(sorted(set((2,3,4,5,7,8,9,11,*factors))))
    tree=ast.parse(_module_source(body,deps))
    tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
    for node in tree.body:node.decorator_list=[]
    language=Grid();language.programs=grid
    ns=dict(tl=language,tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)),range=local_range)
    exec(compile(tree,name,"exec"),ns)
    rng=np.random.default_rng(20261018)
    z=(rng.standard_normal((count,n))+1j*rng.standard_normal((count,n))).astype(np.complex64)
    reference=np.fft.ifft(z,axis=1)*n if inverse else np.fft.fft(z,axis=1)
    data=z.view(np.float32).reshape(-1)
    output=np.full(data.size+128,31415.25,np.float32)
    for pid in range(grid):
        language.pid=pid
        ns[name](Pointer(data),Pointer(output,64),Pointer(index),Pointer(twiddle),count)
    value=output[64:-64].view(np.complex64).reshape(z.shape)
    error=value.astype(np.complex128)-reference
    assert np.isfinite(value).all()
    assert np.max(np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(reference,axis=1),1e-30))<2e-6
    assert np.all(output[:64]==31415.25) and np.all(output[-64:]==31415.25)

@pytest.mark.parametrize("backend,dtype",[("npu","complex64"),("npu","complex128"),("cuda","complex64"),("ix","complex64"),("maca","complex64"),("hcu","complex64"),("musa","complex64"),("ppu","complex64")])
def test_fused_cli_scope(tmp_path,backend,dtype):
    repo=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,"-m","flagfft_codegen.jit_source",
        "--kernel","npu_fused_stockham","--length","48","--factors","12,2,2",
        "--npu-fft-group","32","--direction","forward","--target",f"{backend}:source-contract:32",
        "--dtype",dtype,"--out-dir",str(tmp_path)],text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH=str(repo/"python"),PYTHONDONTWRITEBYTECODE="1"))
    if backend!="npu" or dtype!="complex64":
        assert result.returncode!=0 and "NPU FP32" in result.stderr
        return
    assert result.returncode==0,result.stderr
    m=json.loads(result.stdout)
    assert m["signature"]=="*fp32:16,*fp32:16,*u32:16,*fp32:16,i32"
    assert m["batch_per_block"]==32
    assert "tle.dsa.insert_slice" in Path(m["module_path"]).read_text()

@pytest.mark.parametrize("n,factors,group",[(14,(14,),8),(18,(18,),8),(257,(257,),1),(48,(12,4,2),8),(256,(16,16),32),(48,(12,2,2),3)])
def test_rejects_unqualified_layouts(n,factors,group):
    with pytest.raises(ValueError):fused_stockham_layout(n,factors,group)
