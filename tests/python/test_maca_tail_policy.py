"""CPU-only checks of the precise native-request-scoped tail policy."""
from pathlib import Path
import os
import json
import sys

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.maca_tail_policy import (
    ENV_NAMES, eligible_kernel_mode, maca_tail_kernel_scope, resource_default,
    reset_maca_tail_mode, set_maca_tail_mode, variant_suffix,
)
from flagfft_codegen.target import (
    set_codegen_target, set_maca_1d_single_default, reset_maca_1d_single_default,
)


@pytest.fixture(autouse=True)
def maca(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    profile = BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
        "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536,
    }, "legacy")
    p = set_profile(profile)
    single = set_maca_1d_single_default(True)
    tail = set_maca_tail_mode("off")
    set_codegen_target("maca:80:64")
    yield
    reset_profile(p)
    reset_maca_1d_single_default(single)
    reset_maca_tail_mode(tail)
    set_codegen_target("")


def emit(path, kernel="four_step_col", dtype="complex128", length=1024, n1=1024, n2=1024):
    real = kernel.startswith("direct_dft")
    return emit_jit_kernel(
        kernel=kernel, length=length, factors=() if real else (16, 8, 8),
        lanes=1 if real else 64, num_warps=4 if real else 2,
        generic_radices=(), smem_size=0 if real else 1024,
        direction="inverse" if kernel == "direct_dft_c2r" else "forward",
        dtype=dtype, prime_n=0, four_step_n1=n1, four_step_n2=n2, out_dir=path,
    )


@pytest.mark.parametrize("kernel", ["four_step_row", "four_step_col"])
def test_plan_local_resources_match_measured_manual_configuration(tmp_path, monkeypatch, kernel):
    before = dict(os.environ)
    baseline = emit(tmp_path, kernel)
    set_maca_tail_mode("p4w4")
    candidate = emit(tmp_path, kernel)
    assert candidate["inner_pack"] == 4
    assert candidate["num_warps"] == 4
    assert dict(os.environ) == before
    assert resource_default("INNER_PACK") is None  # scope restored
    assert candidate["module_path"] != baseline["module_path"]
    candidate_source = Path(candidate["module_path"]).read_text()
    set_maca_tail_mode("off")
    for key, value in {"FP64_REGISTER_PACK": "1", "INNER_PACK": "4", "MAX_WARPS": "4"}.items():
        monkeypatch.setenv("FLAGFFT_MACA_" + key, value)
    manual = emit(tmp_path, kernel)
    assert manual["inner_pack"] == candidate["inner_pack"]
    assert manual["num_warps"] == candidate["num_warps"]
    assert Path(manual["module_path"]).read_text() == candidate_source


@pytest.mark.parametrize("kernel", ["leaf", "four_step_row_strided", "four_step_real_row",
    "four_step_hermitian_row", "four_step_r2c_col", "four_step_c2r_col",
    "bluestein_four_step_prepare_row", "bluestein_four_step_pointwise_row",
    "bluestein_four_step_finish_col"])
def test_unmeasured_kernels_cannot_inherit_resources(kernel):
    set_maca_tail_mode("p4w4")
    assert eligible_kernel_mode(kernel, 1024, "complex128", 1024, 1024) == "off"
    with maca_tail_kernel_scope(kernel, 1024, "complex128", 1024, 1024):
        assert resource_default("FP64_REGISTER_PACK") is None


@pytest.mark.parametrize("dtype,n1,n2,length", [("complex64", 1024, 1024, 1024),
    ("complex128", 512, 1024, 1024), ("complex128", 1024, 2048, 1024),
    ("complex128", 1024, 1024, 512)])
def test_unmeasured_dimensions_are_excluded(dtype, n1, n2, length):
    set_maca_tail_mode("p4w4")
    assert eligible_kernel_mode("four_step_col", length, dtype, n1, n2) == "off"


@pytest.mark.parametrize("kernel", ["direct_dft_r2c", "direct_dft_c2r"])
def test_real23_tree_and_legacy_toggle_are_isolated(tmp_path, monkeypatch, kernel):
    kw = dict(kernel=kernel, length=23, n1=0, n2=0)
    kahan = emit(tmp_path, **kw)
    kahan_source = Path(kahan["module_path"]).read_text()
    set_maca_tail_mode("real23")
    tree = emit(tmp_path, **kw)
    assert tree["kernel_name"].endswith("_tree")
    monkeypatch.setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "kahan")
    override = emit(tmp_path, **kw)
    assert not override["kernel_name"].endswith("_tree")
    assert Path(override["module_path"]).read_text() == kahan_source
    set_maca_tail_mode("off")
    monkeypatch.setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "tree")
    legacy = emit(tmp_path, **kw)
    assert legacy["kernel_name"].endswith("_tree")
    assert Path(legacy["module_path"]).read_text() == Path(tree["module_path"]).read_text()
    assert len({x["module_path"] for x in [kahan, tree, override, legacy]}) == 4
    assert Path(kahan["module_path"]).read_text() == kahan_source
    monkeypatch.delenv("FLAGFFT_MACA_REAL_DFT_REDUCTION")
    assert emit(tmp_path, **kw) == kahan


@pytest.mark.parametrize("n", [22, 24, 29, 32, 33, 128])
def test_real23_does_not_expand_lengths(tmp_path, n):
    set_maca_tail_mode("real23")
    meta = emit(tmp_path, kernel="direct_dft_r2c", length=n, n1=0, n2=0)
    assert not meta["kernel_name"].endswith("_tree")


def test_explicit_resource_override_precedes_policy(tmp_path, monkeypatch):
    set_maca_tail_mode("p4w4")
    monkeypatch.setenv("FLAGFFT_MACA_INNER_PACK", "2")
    monkeypatch.setenv("FLAGFFT_MACA_MAX_WARPS", "8")
    candidate = emit(tmp_path)
    assert candidate["inner_pack"] == 2
    monkeypatch.setenv("FLAGFFT_MACA_FP64_REGISTER_PACK", "0")
    monkeypatch.setenv("FLAGFFT_MACA_INNER_PACK", "4")
    assert emit(tmp_path)["inner_pack"] == 2


def test_cache_identity_and_backend_scope(monkeypatch):
    identities = set()
    for value in [None, "", "kahan", "tree"]:
        if value is None:
            monkeypatch.delenv(ENV_NAMES[0], raising=False)
        else:
            monkeypatch.setenv(ENV_NAMES[0], value)
        identities.add(variant_suffix(root=True))
    assert len(identities) == 4
    set_maca_tail_mode("p4w4")
    set_codegen_target("cuda:80:32")
    assert variant_suffix(root=True) == ""
    assert eligible_kernel_mode("four_step_col", 1024, "complex128", 1024, 1024) == "off"
    with pytest.raises(ValueError):
        set_maca_tail_mode("all")


def test_cli_same_process_switches_filesystem_and_module_identity(tmp_path, monkeypatch, capsys):
    from flagfft_codegen.cli import main
    profile = json.dumps({"backend": "maca", "device_arch": "102", "warp_size": 64,
        "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536})
    def invoke(mode):
        monkeypatch.setattr(sys, "argv", ["jit_source", "--kernel", "direct_dft_r2c",
            "--length", "23", "--dtype", "complex128", "--target", "maca:80:64",
            "--out-dir", str(tmp_path), "--device-profile", profile,
            "--maca-1d-single", "--maca-tail-mode", mode])
        main()
        return json.loads(capsys.readouterr().out)
    baseline = invoke("off")
    tree = invoke("real23")
    assert tree["kernel_name"].endswith("_tree")
    assert Path(tree["module_path"]).parent != Path(baseline["module_path"]).parent
    assert invoke("off")["module_path"] == baseline["module_path"]
    monkeypatch.setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "tree")
    legacy = invoke("off")
    assert legacy["kernel_name"].endswith("_tree")
    assert Path(legacy["module_path"]).parent != Path(baseline["module_path"]).parent
