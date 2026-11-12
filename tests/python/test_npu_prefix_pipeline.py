"""Keep matrix-prefix pipelining scoped and provide a cache-safe opt-out."""
import json
from pathlib import Path
import pytest
from flagfft_codegen.kernels_npu_dif import emit_vector_dif


def emit(tmp_path, factors, direction='forward', phase=0, public=True):
    return emit_vector_dif(factors,direction,phase,tmp_path,aligned=True,
        cube_prefix=True,vector_first17=True,batch_pack=True,
        vector_second15=True,matrix_first17=True,prefix_pitch=32,
        codelet_reuse=True,vector_basis=public,global_inverse=direction=='inverse')


@pytest.mark.parametrize('factors,stages',[
    ((17,15,13,7,8),1),((17,15,13,10,10,2),2),
    ((17,15,13,8,8),1),((17,15,13,16,16),2)])
def test_wide_public_prefix(tmp_path,monkeypatch,factors,stages):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    monkeypatch.delenv('FLAGFFT_NPU_DIF_PREFIX_PIPELINE',raising=False)
    meta=emit(tmp_path,factors)
    assert meta['num_stages']==stages
    assert json.loads(Path(meta['module_path']).with_suffix('.json').read_text())==meta
    assert meta['signature']=='*fp32:16,*fp32:16,*fp32:16'
    assert meta['kernel_name']=='vector_stage_0'


@pytest.mark.parametrize('direction,phase,public',[
    ('inverse',0,True),('forward',1,True),('forward',0,False)])
def test_other_schedules(tmp_path,monkeypatch,direction,phase,public):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_PREFIX_PIPELINE','1')
    assert emit(tmp_path,(17,15,13,10,10,2),direction,phase,public)['num_stages']==1


def test_optout_preserves_source(tmp_path,monkeypatch):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_PREFIX_PIPELINE','0')
    before=emit(tmp_path/'off',(17,15,13,10,10,2))
    monkeypatch.setenv('FLAGFFT_NPU_DIF_PREFIX_PIPELINE','1')
    after=emit(tmp_path/'on',(17,15,13,10,10,2))
    assert before['num_stages']==1 and after['num_stages']==2
    assert Path(before['module_path']).read_text()==Path(after['module_path']).read_text()


def test_vector_component_optout(tmp_path,monkeypatch):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_PREFIX_PIPELINE','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COMPONENT17','0')
    assert emit(tmp_path,(17,15,13,10,10,2))['num_stages']==1
