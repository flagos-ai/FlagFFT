"""CPU-only checks of the precise native-request-scoped tail policy."""
from inspect import signature
from pathlib import Path
import os
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import _emit_r2c_pointwise_jit_kernel, _rader_kernel_source, emit_jit_kernel
from flagfft_codegen.maca_tail_policy import (
    ENV_NAMES, eligible_kernel_mode, maca_tail_kernel_scope, resource_default,
    reset_maca_tail_mode, set_maca_tail_mode, variant_suffix,
)
from flagfft_codegen.target import (
    reset_maca_1d_batch_default, set_maca_1d_batch_default,
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
    factors = () if real else ((8, 4, 4, 4) if length == 512 else (16, 8, 8))
    return emit_jit_kernel(
        kernel=kernel, length=length, factors=factors,
        lanes=1 if real else 64, num_warps=4 if real else 2,
        generic_radices=(), smem_size=0 if real else length,
        direction="inverse" if kernel == "direct_dft_c2r" else "forward",
        dtype=dtype, prime_n=0, four_step_n1=n1, four_step_n2=n2, out_dir=path,
    )


@pytest.mark.parametrize("length,dtype", [(185640, "complex128"), (340200, "complex64"), (524288, "complex64")])
@pytest.mark.parametrize("inverse", [False, True])
def test_paired_batch_real_boundary_keeps_distance_and_grid(tmp_path, monkeypatch, length, dtype, inverse):
    triton = ModuleType("triton")
    triton.language = ModuleType("triton.language")
    triton.jit = lambda function: SimpleNamespace(arg_names=list(signature(function).parameters))
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", triton.language)
    metadata = _emit_r2c_pointwise_jit_kernel(
        kernel="c2r_packed_preprocess" if inverse else "r2c_packed_postprocess",
        n=length, dtype=dtype, target="maca:102:64", out_dir=tmp_path,
        perm_form="maca_paired_real_boundary_w4",
    )
    distance = "input_distance" if inverse else "output_distance"
    assert metadata["arg_names"] == ["in_ptr", "twiddle_ptr", "out_ptr", distance, "nbatch"]
    assert metadata["num_warps"] == 4
    assert metadata["grid_x_override"] == (length // 4 + 256) // 256


@pytest.mark.parametrize("n1,n2,factors", [(64, 128, (4, 4, 4)), (128, 64, (8, 4, 4))])
def test_fused_c2r_row_binds_the_phase_after_output(tmp_path, n1, n2, factors):
    config = dict(kernel="four_step_row", length=n1, factors=factors, lanes=16,
                  num_warps=2, generic_radices=(), smem_size=n1, direction="inverse",
                  dtype="complex64", prime_n=0, four_step_n1=n1, four_step_n2=n2, out_dir=tmp_path)
    base = emit_jit_kernel(**config)
    fused = emit_jit_kernel(**config, perm_form="maca_packed_c2r_row")
    assert fused["arg_names"] == [*base["arg_names"][:2], "packed_twiddle_ptr", *base["arg_names"][2:]]
    assert fused["arg_names"][:3] == ["in_ptr", "out_ptr", "packed_twiddle_ptr"]
    assert (fused["inner_pack"], fused["num_warps"]) == (4, 2)
    assert fused["module_path"] != base["module_path"]
    source = Path(fused["module_path"]).read_text()
    assert "four_step_batch.to(tl.int64) * 8193" in source
    assert "tl.where(src_idx0 == 0, 0.0, ai0)" in source


def test_2048_fp64_d2z_gather_keeps_the_packed_leaf_signature(tmp_path):
    config = dict(kernel="leaf", length=1024, factors=(16, 8, 8), lanes=64,
                  num_warps=4, generic_radices=(), smem_size=1024, direction="forward",
                  dtype="complex128", prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    base = emit_jit_kernel(**config, perm_form="maca_fp64_packed_dif1024")
    gather = emit_jit_kernel(**config, perm_form="maca_fp64_packed_dif1024_gather")
    assert gather["arg_names"] == base["arg_names"]
    assert gather["signature"] == base["signature"]
    assert gather["num_warps"] == 4
    assert gather["module_path"] != base["module_path"]
    source = Path(gather["module_path"]).read_text()
    assert "tl.gather(real, q, 0)" in source
    assert "if indices.shape[0] == 1:" in source


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_32_real_plane_dif_keeps_the_boundary_arguments(tmp_path, direction):
    from flagfft_codegen.kernels_small_3d import emit_fused_32_real_plane_kernel

    config = dict(dtype="complex128", direction=direction, out_dir=tmp_path)
    base = emit_fused_32_real_plane_kernel(**config)
    fused = emit_fused_32_real_plane_kernel(**config, maca_dif=True)
    assert fused["kernel_name"] == f"maca_real32_plane_dif_{direction}"
    assert fused["arg_names"] == base["arg_names"]
    assert fused["signature"] == base["signature"]
    assert fused["num_warps"] == 4
    assert fused["module_path"] != base["module_path"]
    with pytest.raises(ValueError):
        emit_fused_32_real_plane_kernel(**{**config, "dtype": "complex64"}, maca_dif=True)


@pytest.mark.parametrize("kernel", ["four_step_row", "four_step_col"])
@pytest.mark.parametrize("mode,n1", [("p4w4", 1024), ("z2z524288-pack4", 512)])
def test_plan_local_resources_match_measured_manual_configuration(tmp_path, monkeypatch, kernel, mode, n1):
    before = dict(os.environ)
    request = dict(kernel=kernel, length=n1 if kernel == "four_step_row" else 1024, n1=n1)
    baseline = emit(tmp_path, **request)
    set_maca_tail_mode(mode)
    candidate = emit(tmp_path, **request)
    assert candidate["inner_pack"] == 4
    assert candidate["num_warps"] == 4
    assert dict(os.environ) == before
    assert resource_default("INNER_PACK") is None  # scope restored
    assert candidate["module_path"] != baseline["module_path"]
    candidate_source = Path(candidate["module_path"]).read_text()
    set_maca_tail_mode("off")
    for key, value in {"FP64_REGISTER_PACK": "1", "INNER_PACK": "4", "MAX_WARPS": "4"}.items():
        monkeypatch.setenv("FLAGFFT_MACA_" + key, value)
    manual = emit(tmp_path, **request)
    assert manual["inner_pack"] == candidate["inner_pack"]
    assert manual["num_warps"] == candidate["num_warps"]
    assert Path(manual["module_path"]).read_text() == candidate_source


@pytest.mark.parametrize("kernel", ["four_step_row", "four_step_col"])
@pytest.mark.parametrize("mode,dtype,direction,settings", [
    ("r2c663000", "complex64", "forward", {"RADER_CODELET": "1", "VEC_IO": "packed"}),
    ("batch-c2r663000", "complex64", "inverse", {"VEC_IO": "packed"}),
    ("batch-c2r663000", "complex128", "inverse", {"VEC_IO": "packed"}),
    ("batch-c2r663000-split", "complex64", "inverse", {"RADER_CODELET": "1", "VEC_IO": "packed",
                                                     "FOURSTEP_ROW_PACK": "4", "FOURSTEP_COL_PACK": "2"}),
])
def test_packed_663000_resources_match_manual_mode(tmp_path, monkeypatch, kernel, mode, dtype, direction, settings):
    row = kernel == "four_step_row"
    split = mode == "batch-c2r663000-split"
    n1, n2 = (340, 975) if split else (300, 1105)
    row_factors, col_factors = ((17, 5, 4), (13, 5, 5, 3)) if split else ((5, 5, 4, 3), (17, 13, 5))
    row_lanes, col_lanes = (1, 5) if split else (5, 1)
    length, factors, lanes = (n1, row_factors, row_lanes) if row else (n2, col_factors, col_lanes)
    request = dict(
        kernel=kernel, length=length, factors=factors,
        lanes=lanes, num_warps=4, generic_radices=tuple(radix for radix in (13, 17) if radix in factors),
        smem_size=(1 << (length - 1).bit_length()) if split else length,
        direction=direction, dtype=dtype, prime_n=0,
        four_step_n1=n1, four_step_n2=n2,
    )
    set_maca_tail_mode(mode)
    candidate = emit_jit_kernel(**request, out_dir=tmp_path / "scoped")
    set_maca_tail_mode("off")
    for name, value in settings.items():
        monkeypatch.setenv(f"FLAGFFT_MACA_{name}", value)
    manual = emit_jit_kernel(**request, out_dir=tmp_path / "manual")
    assert Path(candidate["module_path"]).read_text() == Path(manual["module_path"]).read_text()
    for name in ("inner_pack", "num_warps", "signature"):
        assert candidate[name] == manual[name]
    if split:
        assert candidate["num_warps"] == 8
        assert candidate["inner_pack"] == (4 if row else 2)


@pytest.mark.parametrize("mode,dtype,n1,n2,vec_io,max_warps", [
    ("c2c340200-split", "complex64", 567, 600, "packed", "4"),
    ("z2z340200-split", "complex128", 504, 675, "packed", "4"),
    ("batch-real340200-split", "complex64", 405, 420, "packed", "8"),
    ("batch-real185640-p2", "complex64", 420, 221, "packed", "4"),
    ("batch-real185640-p4", "complex128", 420, 221, "packed", "8"),
    ("z2z16384-split", "complex128", 256, 64, "1", "4"),
])
def test_batch_split_resources_are_local_to_the_fourstep_axes(mode, dtype, n1, n2, vec_io, max_warps):
    other_dtype = "complex128" if dtype == "complex64" else "complex64"
    set_maca_tail_mode(mode)
    for kernel, length in (("four_step_row", n1), ("four_step_col", n2)):
        with maca_tail_kernel_scope(kernel, length, dtype, n1, n2):
            assert resource_default("VEC_IO") == vec_io
            assert resource_default("MAX_WARPS") == max_warps
        assert eligible_kernel_mode(kernel, length, other_dtype, n1, n2) == "off"
        assert eligible_kernel_mode(kernel, length, dtype, 378, 900) == "off"
    assert resource_default("VEC_IO") is None


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_16384_packed_real_resources_match_the_measured_axes(tmp_path, dtype):
    set_maca_tail_mode("batch-real16384-split")
    for axis, length, factors, inner_pack in (("row", 128, (8, 4, 4), 4), ("col", 64, (4, 4, 4), 8)):
        metadata = emit_jit_kernel(
            kernel=f"four_step_{axis}", length=length, factors=factors,
            lanes=16, num_warps=2, generic_radices=(), smem_size=length,
            direction="inverse", dtype=dtype, prime_n=0,
            four_step_n1=128, four_step_n2=64, out_dir=tmp_path,
        )
        assert (metadata["num_warps"], metadata["inner_pack"]) == (2, inner_pack)
    assert eligible_kernel_mode("four_step_row", 64, dtype, 64, 128) == "off"


def test_524288_packed_real_row_pack_preserves_the_default_column(tmp_path):
    request = dict(length=512, factors=(8, 4, 4, 4), lanes=64, num_warps=2,
                   generic_radices=(), smem_size=512, direction="inverse",
                   dtype="complex64", prime_n=0, four_step_n1=512, four_step_n2=512,
                   out_dir=tmp_path)
    batch = set_maca_1d_batch_default(True)
    try:
        set_maca_tail_mode("batch-vecio")
        baseline = emit_jit_kernel(kernel="four_step_col", **request)
        set_maca_tail_mode("batch-real524288-row4")
        assert emit_jit_kernel(kernel="four_step_col", **request) == baseline
        row = emit_jit_kernel(kernel="four_step_row", **request)
        assert (row["num_warps"], row["inner_pack"]) == (8, 4)
        set_maca_tail_mode("batch-real524288-split")
        request.update(four_step_n1=256, four_step_n2=1024)
        for axis, length, factors, geometry in (
            ("row", 256, (4, 4, 4, 4), (4, 4)),
            ("col", 1024, (16, 8, 8), (8, 8)),
        ):
            request.update(length=length, factors=factors, smem_size=length)
            split = emit_jit_kernel(kernel=f"four_step_{axis}", **request)
            assert (split["num_warps"], split["inner_pack"]) == geometry
    finally:
        reset_maca_1d_batch_default(batch)


@pytest.mark.parametrize("kernel", ["leaf", "four_step_row_strided", "four_step_real_row",
    "four_step_hermitian_row", "four_step_r2c_col", "four_step_c2r_col",
    "bluestein_four_step_prepare_row", "bluestein_four_step_pointwise_row",
    "bluestein_four_step_finish_col"])
def test_unmeasured_kernels_cannot_inherit_resources(kernel):
    set_maca_tail_mode("p4w4")
    assert eligible_kernel_mode(kernel, 1024, "complex128", 1024, 1024) == "off"
    with maca_tail_kernel_scope(kernel, 1024, "complex128", 1024, 1024):
        assert resource_default("FP64_REGISTER_PACK") is None


def test_16_fp64_complex_pack_matches_the_measured_geometry(tmp_path):
    set_maca_tail_mode("z2z16-pack16")
    metadata = emit_jit_kernel(
        kernel="leaf", length=16, factors=(16,), lanes=1, num_warps=2, generic_radices=(),
        smem_size=0, direction="inverse", dtype="complex128", prime_n=0,
        four_step_n1=0, four_step_n2=0, out_dir=tmp_path,
    )
    assert (metadata["num_warps"], metadata["batch_per_block"]) == (2, 16)
    assert eligible_kernel_mode("leaf", 16, "complex64") == "off"
    assert eligible_kernel_mode("leaf_c2r", 16, "complex128") == "off"


@pytest.mark.parametrize("dtype,n1,n2,length", [("complex64", 1024, 1024, 1024),
    ("complex128", 512, 1024, 1024), ("complex128", 1024, 2048, 1024),
    ("complex128", 1024, 1024, 512)])
def test_unmeasured_dimensions_are_excluded(dtype, n1, n2, length):
    set_maca_tail_mode("p4w4")
    assert eligible_kernel_mode("four_step_col", length, dtype, n1, n2) == "off"


@pytest.mark.parametrize("kernel", ["direct_dft_r2c", "direct_dft_c2r"])
def test_real_direct_tree_and_legacy_toggle_are_isolated(tmp_path, monkeypatch, kernel):
    kw = dict(kernel=kernel, length=23, n1=0, n2=0)
    kahan = emit(tmp_path, **kw)
    kahan_source = Path(kahan["module_path"]).read_text()
    set_maca_tail_mode("real-direct")
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
def test_real_direct_tree_does_not_expand_lengths(tmp_path, n):
    set_maca_tail_mode("real-direct")
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


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("kernel", ["rader_prepare", "rader_finalize"])
def test_prime_rader_vector_io_policy_and_override(monkeypatch, dtype, kernel):
    def source():
        return _rader_kernel_source(kernel, 8191, 8190, dtype)[1]

    baseline = source()
    assert "tl.arange(0, 2)[None, :]" not in baseline
    set_maca_tail_mode("batch-prime-rader-vecio")
    candidate = source()
    assert "tl.arange(0, 2)[None, :]" in candidate
    monkeypatch.setenv("FLAGFFT_MACA_RADER_VEC_IO", "0")
    assert source() == baseline
    monkeypatch.setenv("FLAGFFT_MACA_RADER_VEC_IO", "1")
    assert source() == candidate


def test_cli_accepts_prime_rader_vector_io_mode(tmp_path, monkeypatch, capsys):
    from flagfft_codegen.cli import main
    profile = json.dumps({"backend": "maca", "device_arch": "102", "warp_size": 64,
        "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536})
    monkeypatch.setattr(sys, "argv", ["jit_source", "--kernel", "rader_prepare",
        "--rader-n", "8191", "--rader-m", "8190", "--dtype", "complex64",
        "--target", "maca:80:64", "--out-dir", str(tmp_path),
        "--device-profile", profile, "--maca-tail-mode", "batch-prime-rader-vecio"])
    main()
    metadata = json.loads(capsys.readouterr().out)
    assert "tl.arange(0, 2)[None, :]" in Path(metadata["module_path"]).read_text()


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
    tree = invoke("real-direct")
    assert tree["kernel_name"].endswith("_tree")
    assert Path(tree["module_path"]).parent != Path(baseline["module_path"]).parent
    assert invoke("off")["module_path"] == baseline["module_path"]
    monkeypatch.setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "tree")
    legacy = invoke("off")
    assert legacy["kernel_name"].endswith("_tree")
    assert Path(legacy["module_path"]).parent != Path(baseline["module_path"]).parent
