"""The provenance audit must reject backend reintroduction, not just pass today's tree."""
import importlib.util
from pathlib import Path

PATH=Path(__file__).resolve().parents[2]/'tools/audit_npu_backend.py'
spec=importlib.util.spec_from_file_location('npu_audit',PATH)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_reference_is_allowed_but_core_call_is_rejected(tmp_path):
    p=tmp_path/audit.REFERENCE_SOURCE;p.parent.mkdir(parents=True)
    p.write_text('AsdSip::asdFftExecC2C();')
    assert audit.scan_source(tmp_path)['passed']
    p=tmp_path/'src/exec/impl.cpp';p.parent.mkdir(parents=True)
    p.write_text('AsdSip::asdFftExecC2C();')
    assert not audit.scan_source(tmp_path)['passed']


def test_ascendc_kernel_and_build_are_rejected(tmp_path):
    p=tmp_path/'src/backend/device.cpp';p.parent.mkdir(parents=True)
    p.write_text('__aicore__ void fft() { AscendC::DataCopy(); }')
    assert not audit.scan_source(tmp_path)['passed']
    p.unlink()
    (tmp_path/'CMakeLists.txt').write_text('ascendc_library(fft STATIC device.cpp)')
    assert not audit.scan_source(tmp_path)['passed']


def test_npu_slice_modules_use_the_installed_language_extension(monkeypatch):
    from flagfft_codegen import metadata
    monkeypatch.setattr(metadata, '_npu_backend_active', lambda:True)
    source=metadata._module_source('x = tle.dsa.insert_slice(v,a,[0],[64],[1])')
    assert 'import triton.language.extra.cann.extension as al' in source
    assert 'al.insert_slice(v,a,[0],[64],[1])' in ''.join(source.split())
    assert 'triton.experimental.tle' not in source
