"""Guard the source/build boundary of the generated Ascend implementation."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def test_no_hand_written_ascend_device_sources():
    backend = ROOT / "src/adaptor/backend/npu"
    assert not list(backend.glob("ascendc_*"))
    for source in backend.glob("*.*"):
        text = source.read_text()
        assert "__aicore__" not in text, source
        assert "kernel_operator.h" not in text, source


def test_no_native_kernel_build_or_dispatch():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert not re.search(r"ascendc_library\s*\(|ascendc_kernel_cmake|flagfft_npu_aiv", cmake)
    for name in ("src/codegen/compiler.cpp", "src/exec/raw_nodes.cpp",
                 "src/utils/include/flagfft/codegen.hpp"):
        text = (ROOT / name).read_text()
        assert "launch_ascendc_" not in text, name
        assert "CompiledRawNpuAiv" not in text, name
        assert "CompiledRawNpuPackedRealBoundary" not in text, name


def test_no_product_sip_executor():
    assert not list((ROOT / "src").rglob("sip_executor.*"))
    for source in (ROOT / "src/exec").glob("c_api*.*"):
        assert "asdsip" not in source.read_text().lower(), source
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "FLAGFFT_ENABLE_SIP_EXECUTION" not in cmake
    # The independent test reference remains present.
    assert "find_library(ASCEND_SIP_LIBRARY NAMES asdsip" in cmake
    assert "${ASCEND_SIP_LIBRARY}" in cmake
