"""Whole-FFT checks for fixed rotations through the existing power-axis ABI."""
import pytest
from test_npu_power_dif import test_power_dif_complete as run_fft

@pytest.mark.parametrize('factors',[(32,16,16),(32,32,16,16),(32,32,32,16)])
@pytest.mark.parametrize('direction',['forward','inverse','global_inverse'])
def test_factored_terminal_fft(tmp_path,factors,direction):
    run_fft(tmp_path,factors,direction,True,terminal8_factored=True)

def test_factored_terminal_metadata(tmp_path):
    from flagfft_codegen.kernels_npu_dif import emit_vector_dif
    from pathlib import Path
    meta=emit_vector_dif((32,32,32,16),'inverse',0,tmp_path,flat_power2=True,
        terminal8=True,terminal8_factored=True,global_inverse=True)
    assert 'T1r=(O1r+O1i)*' in Path(meta['module_path']).read_text()
    assert meta['arg_names']==['X','TW','Y']
