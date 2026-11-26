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


@pytest.mark.parametrize("axis,phase,n1,n2", [("row", "gather", 70, 117), ("col", "scatter", 117, 70)])
def test_maca_rader_compact_indices_preserve_launch_order(tmp_path, axis, phase, n1, n2):
    config = dict(kernel=f"four_step_{axis}", length=70, factors=(7, 10), lanes=1,
                  num_warps=2, generic_radices=(), smem_size=128, direction="forward",
                  dtype="complex64", prime_n=8191, four_step_n1=n1, four_step_n2=n2)
    prefix = f"maca_rader_four_step_{phase}"
    wide = emit_jit_kernel(**config, perm_form=prefix, out_dir=tmp_path)
    narrow = emit_jit_kernel(**config, perm_form=prefix + "_i16", out_dir=tmp_path)

    assert wide["module_path"] != narrow["module_path"]
    assert wide["arg_names"][0] == "idx_ptr"
    assert narrow["arg_names"] == ["idx16_ptr", *wide["arg_names"][1:]]
    assert wide["signature"].split(",")[0] == "*i32:16"
    assert narrow["signature"].split(",")[0] == "*i16:16"
    wide_source = Path(wide["module_path"]).read_text()
    narrow_source = Path(narrow["module_path"]).read_text()
    assert narrow_source.replace("idx16_ptr", "idx_ptr").replace(
        narrow["kernel_name"], wide["kernel_name"]
    ) == wide_source


def test_maca_rader_sorted_scatter_keeps_the_int32_table_abi(tmp_path):
    config = dict(kernel="four_step_col", length=180, factors=(5, 4, 3, 3), lanes=3,
                  num_warps=2, generic_radices=(), smem_size=256, direction="forward",
                  dtype="complex64", prime_n=16381, four_step_n1=91, four_step_n2=180)
    phase = "maca_rader_four_step_scatter"
    base = emit_jit_kernel(**config, perm_form=phase, out_dir=tmp_path)
    sorted_indices = emit_jit_kernel(**config, perm_form=phase + "_sorted", out_dir=tmp_path)

    assert sorted_indices["module_path"] != base["module_path"]
    assert sorted_indices["arg_names"] == base["arg_names"]
    assert sorted_indices["signature"] == base["signature"]
    source = Path(sorted_indices["module_path"]).read_text()
    assert source.count("tl.store(") == 1
    assert "tl.gather(scatter_bank, packed_map" in source
    assert "x0r + r0 / 16380" in source


def test_maca_rader_fused_spectrum_emits_the_table_free_middle_stage(tmp_path):
    config = dict(kernel="four_step_col", length=91, factors=(13, 7), lanes=1,
                  num_warps=2, generic_radices=(), smem_size=128, direction="forward",
                  dtype="complex64", prime_n=16381, four_step_n1=180, four_step_n2=91,
                  perm_form="maca_rader_four_step_spectral", out_dir=tmp_path)
    metadata = emit_jit_kernel(**config)
    assert metadata["arg_names"] == [
        "b_fft_ptr", "user_ptr", "dc_ptr", "zero_ptr", "in_ptr", "out_ptr", "nbatch",
    ]
    assert (metadata["inner_pack"], metadata["num_warps"]) == (8, 2)
    assert not metadata["tle_fused_twiddle"]
    with pytest.raises(ValueError, match="16381-point FourStep column"):
        emit_jit_kernel(**dict(config, prime_n=8191))


@pytest.mark.parametrize("prime,rows,cols,factors,lanes,pack,warps,narrow", [
    (8191, 117, 70, (7, 10), 1, 8, 2, False),
    (8191, 117, 70, (7, 10), 1, 8, 2, True),
    (16381, 91, 180, (5, 4, 3, 3), 3, 4, 4, False),
])
def test_maca_rader_good_thomas_scatter_has_no_twiddle_arguments(
    tmp_path, prime, rows, cols, factors, lanes, pack, warps, narrow,
):
    suffix = "_i16" if narrow else ""
    meta = emit_jit_kernel(kernel="four_step_col", length=cols, factors=factors,
        lanes=lanes, num_warps=2, generic_radices=(), smem_size=1 << (cols - 1).bit_length(),
        direction="forward", dtype="complex64", prime_n=prime, four_step_n1=rows,
        four_step_n2=cols, perm_form="maca_rader_four_step_scatter_pfa" + suffix, out_dir=tmp_path)
    assert meta["arg_names"] == [
        "idx16_ptr" if narrow else "idx_ptr", "zero_ptr", "in_ptr", "out_ptr", "nbatch",
    ]
    assert meta["signature"].split(",")[0] == ("*i16:16" if narrow else "*i32:16")
    assert (meta["inner_pack"], meta["num_warps"]) == (pack, warps)
    assert not meta["tle_fused_twiddle"]


@pytest.mark.parametrize("real_kind", ["r2c", "c2r"])
@pytest.mark.parametrize("prime,row,col", [(8191, (7, 10), (13, 9)), (16381, (5, 4, 3, 3), (13, 7))])
def test_maca_rader_direct_real_passes_accept_public_distances(tmp_path, real_kind, prime, row, col):
    from math import lcm, prod

    n1, n2 = prod(row), prod(col)
    for phase, kernel, factors, rows, columns, front, distance in (
        ("gather", "row", row, n1, n2, ["idx_ptr"], "input_distance"),
        ("product", "col", col, n1, n2, ["b_fft_ptr", "user_ptr", "zero_ptr"], "input_distance"),
        ("scatter", "col", row, n2, n1, ["idx_ptr", "zero_ptr"], "output_distance"),
    ):
        length = prod(factors)
        meta = emit_jit_kernel(kernel=f"four_step_{kernel}", length=length, factors=factors,
            lanes=length // lcm(*factors), num_warps=2, generic_radices=(),
            smem_size=1 << (length - 1).bit_length(), direction="forward", dtype="complex64",
            prime_n=prime, four_step_n1=rows, four_step_n2=columns,
            perm_form=f"maca_rader_four_step_{phase}_{real_kind}", out_dir=tmp_path)
        twiddles = [f"tw{s}_{part}_ptr" for s in range(1, len(factors)) for part in ("r", "i")]
        assert meta["arg_names"] == [*front, "in_ptr", "out_ptr", *twiddles, distance, "nbatch"]
        assert meta["signature"].split(",")[-2:] == ["i64", "i32"]
        source = Path(meta["module_path"]).read_text()
        if phase == "product":
            assert "tl.store(zero_ptr + zero_offset + 1, x0r + r0, mask=is_dc)" in source
            assert "dc_ptr" not in source
        if phase == "scatter":
            assert "dc = tl.load(x0 + 1)" in source


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


def test_2048_fp32_r2c_gather_uses_float_tables_and_runtime_distances(tmp_path):
    config = dict(kernel="leaf", length=1024, factors=(16, 8, 8), lanes=64,
                  num_warps=2, generic_radices=(), smem_size=1024, direction="forward",
                  dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path,
                  perm_form="maca_fp32_packed_dif1024_gather")
    metadata = emit_jit_kernel(**config)
    assert metadata["kernel_name"] == "maca_fp32_packed_dif1024_r2c_gather"
    assert metadata["arg_names"] == ["in_ptr", "out_ptr", "roots_r_ptr", "roots_i_ptr",
                                     "phase_r_ptr", "phase_i_ptr", "input_distance", "output_distance", "nbatch"]
    signature = [field.split(":")[0] for field in metadata["signature"].split(",")]
    assert signature == ["*fp32"] * 6 + ["i64", "i64", "i32"]
    assert (metadata["num_warps"], metadata["batch_per_block"]) == (4, 1)
    with pytest.raises(ValueError):
        emit_jit_kernel(**{**config, "direction": "inverse"})
    with pytest.raises(ValueError):
        emit_jit_kernel(**{**config, "dtype": "complex128"})


def test_fp64_c2r48_from_columns_uses_the_raw_leaf_signature(tmp_path):
    config = dict(kernel="leaf", length=48, factors=(4, 4, 3), lanes=4,
                  num_warps=2, generic_radices=(), smem_size=64,
                  direction="inverse", dtype="complex128", prime_n=0,
                  four_step_n1=0, four_step_n2=0, out_dir=tmp_path,
                  perm_form="maca_fp64_c2r48_from_columns")
    metadata = emit_jit_kernel(**config)
    assert metadata["arg_names"] == ["in_ptr", "out_ptr", "phase_r_ptr", "phase_i_ptr", "nbatch"]
    assert metadata["num_warps"] == 2
    assert metadata["batch_per_block"] == 16
    with pytest.raises(ValueError):
        emit_jit_kernel(**{**config, "direction": "forward"})
    with pytest.raises(ValueError):
        emit_jit_kernel(**{**config, "dtype": "complex64"})


def test_fp64_dif_terminal8_uses_the_existing_middle_axis_arguments(tmp_path):
    config = dict(kernel="leaf", length=2048, factors=(16, 8, 16), lanes=128,
                  num_warps=2, generic_radices=(), smem_size=2048, direction="inverse",
                  dtype="complex128", prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    base = emit_jit_kernel(**config, perm_form="maca_fp64_dif")
    candidate = emit_jit_kernel(**config, perm_form="maca_fp64_dif_terminal8")
    assert candidate["arg_names"] == base["arg_names"]
    assert candidate["signature"] == base["signature"]
    assert candidate["num_warps"] == 2
    assert candidate["batch_per_block"] == 1
    assert candidate["module_path"] != base["module_path"]
    source = Path(candidate["module_path"]).read_text()
    assert "for stage in tl.static_range(4)" in source
    assert "def _fwd_rad8_b1(" in source


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_fp32_210_pfa_has_no_external_twiddle_tables(tmp_path, direction):
    config = dict(kernel="leaf", length=210, factors=(7, 5, 3, 2), lanes=1,
                  num_warps=2, generic_radices=(), smem_size=256, direction=direction,
                  dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
                  out_dir=tmp_path, perm_form="maca_fp32_fft210_pfa")
    meta = emit_jit_kernel(**config)
    assert meta["kernel_name"] == f"maca_fp32_fft210_pfa_{direction}"
    assert meta["arg_names"] == ["in_ptr", "out_ptr", "nbatch"]
    assert meta["batch_per_block"] == 2
    assert meta["num_warps"] == 2
    with pytest.raises(ValueError):
        emit_jit_kernel(**{**config, "dtype": "complex128"})


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


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_16384_fp32_row_twiddle_preserves_the_fourstep_launch_abi(tmp_path, direction):
    set_maca_tail_mode("c2c16384-row-twiddle")
    for axis, length, pack, warps in (("row", 256, 4, 4), ("col", 64, 8, 2)):
        metadata = emit_jit_kernel(
            kernel=f"four_step_{axis}", length=length, factors=(4,) * (4 if axis == "row" else 3),
            lanes=length // 4, num_warps=2, generic_radices=(), smem_size=length,
            direction=direction, dtype="complex64", prime_n=0, four_step_n1=256,
            four_step_n2=64, out_dir=tmp_path,
        )
        assert metadata["tle_fused_twiddle"]
        assert (metadata["inner_pack"], metadata["num_warps"]) == (pack, warps)
        assert ("twiddle_ptr" in metadata["arg_names"]) == (axis == "row")
        expected = ["in_ptr", "twiddle_ptr", "out_ptr"] if axis == "row" else ["in_ptr", "out_ptr"]
        assert metadata["arg_names"][:len(expected)] == expected
        source = Path(metadata["module_path"]).read_text()
        assert ("tl.cos(outer_angle0)" in source) == (axis == "row")
        assert "sin.approx.f32" not in source
    assert eligible_kernel_mode("four_step_row", 128, "complex64", 128, 128) == "off"
    assert eligible_kernel_mode("four_step_row", 256, "complex128", 256, 64) == "off"


@pytest.mark.parametrize("axis,length,factors,lanes,pack", [
    ("row", 900, (10, 9, 10), 10, 2),
    ("col", 378, (7, 6, 9), 3, 4),
])
@pytest.mark.parametrize("dtype,mode", [("complex64", "c2c340200-single"), ("complex128", "z2z340200-single")])
def test_340200_single_split_matches_the_screened_axes(tmp_path, monkeypatch, axis, length, factors, lanes,
                                                      pack, dtype, mode):
    kernel = f"four_step_{axis}"
    request = dict(kernel=kernel, length=length, factors=factors, lanes=lanes,
                   num_warps=2, generic_radices=(),
                   smem_size=1 << (length - 1).bit_length(), direction="inverse",
                   dtype=dtype, prime_n=0, four_step_n1=900, four_step_n2=378)
    set_maca_tail_mode(mode)
    candidate = emit_jit_kernel(**request, out_dir=tmp_path / "scoped")
    assert (candidate["num_warps"], candidate["inner_pack"]) == (4, pack)
    other_dtype = "complex64" if dtype == "complex128" else "complex128"
    assert eligible_kernel_mode(kernel, length, other_dtype, 900, 378) == "off"
    assert eligible_kernel_mode(kernel, length, dtype, 378, 900) == "off"
    other_length = 378 if axis == "row" else 900
    assert eligible_kernel_mode(kernel, other_length, dtype, 900, 378) == "off"
    set_maca_tail_mode("off")
    monkeypatch.setenv("FLAGFFT_MACA_VEC_IO", "packed")
    monkeypatch.setenv("FLAGFFT_MACA_MAX_WARPS", "8")
    manual = emit_jit_kernel(**request, out_dir=tmp_path / "manual")
    assert Path(candidate["module_path"]).read_text() == Path(manual["module_path"]).read_text()


@pytest.mark.parametrize("axis,length,factors,lanes,pack", [
    ("row", 900, (10, 9, 10), 10, 1),
    ("col", 378, (7, 6, 9), 3, 2),
])
def test_340200_single_inverse_pack_matches_the_screen(tmp_path, monkeypatch, axis, length, factors, lanes, pack):
    request = dict(kernel=f"four_step_{axis}", length=length, factors=factors, lanes=lanes,
                   num_warps=2, generic_radices=(), smem_size=1 << (length - 1).bit_length(),
                   direction="inverse", dtype="complex64", prime_n=0, four_step_n1=900, four_step_n2=378)
    set_maca_tail_mode("c2c340200-inverse-pack")
    candidate = emit_jit_kernel(**request, out_dir=tmp_path / "scoped")
    assert (candidate["num_warps"], candidate["inner_pack"]) == (2, pack)
    set_maca_tail_mode("c2c340200-single")
    monkeypatch.setenv("FLAGFFT_MACA_INNER_PACK", str(pack))
    manual = emit_jit_kernel(**request, out_dir=tmp_path / "manual")
    assert Path(candidate["module_path"]).read_bytes() == Path(manual["module_path"]).read_bytes()


@pytest.mark.parametrize("direction,mode,row_pack,col_pack", [
    ("forward", "c2c340200-single-row-phase", 2, 4),
    ("inverse", "c2c340200-inverse-pack-row-phase", 1, 2),
])
def test_340200_single_row_phase_keeps_the_twiddle_abi(tmp_path, direction, mode, row_pack, col_pack):
    set_maca_tail_mode(mode)
    for axis, length, factors, lanes, pack in (
        ("row", 900, (10, 9, 10), 10, row_pack),
        ("col", 378, (7, 6, 9), 3, col_pack),
    ):
        metadata = emit_jit_kernel(kernel=f"four_step_{axis}", length=length, factors=factors,
            lanes=lanes, num_warps=2, generic_radices=(), smem_size=1 << (length - 1).bit_length(),
            direction=direction, dtype="complex64", prime_n=0, four_step_n1=900,
            four_step_n2=378, out_dir=tmp_path)
        assert metadata["inner_pack"] == pack
        assert metadata["tle_fused_twiddle"]
        assert ("twiddle_ptr" in metadata["arg_names"]) == (axis == "row")
        source = Path(metadata["module_path"]).read_text()
        assert ("step_angle = four_step_inner * 90" in source) == (axis == "row")
        assert ("tw_r9, tw_i9 = _cmul(tw_r8, tw_i8, step_r, step_i)" in source) == (axis == "row")


@pytest.mark.parametrize("axis,length,factors,lanes,pack,max_warps", [
    ("row", 900, (10, 10, 9), 10, 2, 4),
    ("col", 378, (9, 7, 6), 3, 4, 8),
])
def test_340200_batch_swapped_axes_match_the_screen(tmp_path, monkeypatch, axis, length, factors,
                                                 lanes, pack, max_warps):
    token = set_maca_1d_single_default(False)
    try:
        request = dict(kernel=f"four_step_{axis}", length=length, factors=factors, lanes=lanes,
                       num_warps=2, generic_radices=(), smem_size=1 << (length - 1).bit_length(),
                       direction="inverse", dtype="complex64", prime_n=0, four_step_n1=900, four_step_n2=378)
        set_maca_tail_mode("c2c340200-batch-900x378")
        candidate = emit_jit_kernel(**request, out_dir=tmp_path / "scoped")
        assert (candidate["num_warps"], candidate["inner_pack"]) == (4, pack)
        assert eligible_kernel_mode(request["kernel"], length, "complex128", 900, 378) == "off"
        set_maca_tail_mode("off")
        for key, value in {"VEC_IO": "packed", "EXCHANGE": "direct_all",
                           "INNER_PACK": str(pack), "MAX_WARPS": str(max_warps)}.items():
            monkeypatch.setenv(f"FLAGFFT_MACA_{key}", value)
        manual = emit_jit_kernel(**request, out_dir=tmp_path / "manual")
        assert Path(candidate["module_path"]).read_bytes() == Path(manual["module_path"]).read_bytes()
    finally:
        reset_maca_1d_single_default(token)


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
