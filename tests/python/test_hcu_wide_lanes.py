# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the emitted wide FFT arithmetic against an independent NumPy FFT."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_ix_warp import build_ix_warp_source
from flagfft_codegen.kernels_hcu_warp import hcu_dense_wave_lanes
from flagfft_codegen.kernels_common import LeafPlan


class Pointer:
    def __init__(self, data, offset=0):
        self.data = data.reshape(-1)
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class NumpyTL:
    pid = 0
    pid1 = 0
    float64 = np.float64
    float32 = np.float32
    uint32 = np.uint32
    uint64 = np.uint64
    int64 = np.int64
    pointer_type = staticmethod(lambda dtype: dtype)
    arange = staticmethod(np.arange)
    full = staticmethod(np.full)
    broadcast_to = staticmethod(np.broadcast_to)
    gather = staticmethod(np.take_along_axis)
    reshape = staticmethod(np.reshape)
    join = staticmethod(lambda a, b: np.stack((a, b), axis=-1))
    trans = staticmethod(np.transpose)
    minimum = staticmethod(np.minimum)

    @staticmethod
    def program_id(axis):
        return NumpyTL.pid if axis == 0 else NumpyTL.pid1

    @staticmethod
    def multiple_of(value, hint):
        return value

    max_contiguous = multiple_of

    @staticmethod
    def where(condition, a, b):
        # Triton's untyped floating literals start as FP32. Preserve array
        # precision when either arm already carries an explicit dtype.
        if isinstance(a, float) and isinstance(b, float):
            a, b = np.float32(a), np.float32(b)
        return np.where(condition, a, b)

    @staticmethod
    def cast(value, dtype, bitcast=False):
        if isinstance(value, Pointer):
            assert value.offset == 0
            return Pointer(value.data.view(dtype))
        value = np.asarray(value)
        return value.view(dtype) if bitcast else value.astype(dtype)

    @staticmethod
    def load(ptr, mask=True, other=0):
        index, valid = np.broadcast_arrays(ptr.offset, mask)
        result = np.full(index.shape, other, dtype=ptr.data.dtype)
        result[valid] = ptr.data[index[valid]]
        return result

    @staticmethod
    def store(ptr, value, mask=True):
        shape = np.shape(ptr.offset)
        assert np.broadcast_shapes(shape, np.shape(value), np.shape(mask)) == shape
        index, value, valid = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.data[index[valid]] = value[valid]


@pytest.mark.parametrize("axes", [(64, 128), (512, 1024), (1024, 512)])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_complete_wave_fourstep_pair_applies_cross_axis_twiddle_once(monkeypatch, axes, wide, inverse):
    n1, n2 = axes
    real_dtype = np.float64 if wide else np.float32
    complex_dtype = np.complex128 if wide else np.complex64
    rng = np.random.default_rng(177)
    data = (rng.normal(size=n1*n2) + 1j*rng.normal(size=n1*n2)).astype(complex_dtype)
    expected = np.fft.ifft(data)*len(data) if inverse else np.fft.fft(data)
    intermediate = np.full_like(data, np.nan)
    output = np.full_like(data, np.nan)
    sign = 1 if inverse else -1
    # The shared table ABI stores contiguous n1 coefficients per input column.
    cross = np.exp(sign*2j*np.pi*np.arange(n2)[:, None]*np.arange(n1)[None, :]/(n1*n2)).astype(complex_dtype)
    for n, other, mode, src, dst in ((n1, n2, "four_step_row", data, intermediate),
                                     (n2, n1, "four_step_col", intermediate, output)):
        pack = 4 if wide else 8
        name, source = build_ix_warp_source(n, pack, mode, inverse,
            wide=wide, table_roots=not wide, four_step_other=other,
            local_codelet=True, output_radix_chunk=1)
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            function.decorator_list = []
        roots = np.exp(sign*2j*np.pi*np.arange(n)/n)
        namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.astype(real_dtype)),
            "wave_tw_i_ptr": Pointer(roots.imag.astype(real_dtype)),
            "twiddle_ptr": Pointer(cross.ravel().view(real_dtype))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<wave-fourstep-pair>", "exec"), namespace)
        for pid in range((other+pack-1)//pack):
            monkeypatch.setattr(NumpyTL, "pid", pid)
            namespace[name](Pointer(src.view(real_dtype)), Pointer(dst.view(real_dtype)), 1, 1, n1*n2, n1*n2)
    np.testing.assert_allclose(output, expected,
        rtol=2e-12 if wide else 2e-4, atol=5e-10 if wide else 2e-3)
    relative = np.linalg.norm(output-expected)/np.linalg.norm(expected)
    assert relative < (4e-15 if wide else 4e-7)


@pytest.mark.parametrize("n", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("layout", ["wave", "compact"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("span", [1, 3, 8, 65])
def test_packed_transposed_spectrum_matches_numpy_with_partial_planes(monkeypatch, n, layout, wide, span):
    pack, batch = (8 if wide else 16), 2 * span + 1
    dtype = np.float64 if wide else np.float32
    rng = np.random.default_rng(1711)
    data = rng.normal(size=(batch, 2 * n)).astype(dtype)
    transformed = np.fft.rfft(data, axis=-1)
    matrices = (batch + span - 1) // span
    expected = np.full((matrices * span, n + 1), np.nan, dtype=transformed.dtype)
    expected[:batch] = transformed
    expected = expected.reshape(matrices, span, n + 1).transpose(0, 2, 1).ravel()
    output = np.full(expected.shape, np.nan, dtype=np.complex128 if wide else np.complex64)
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", layout)
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64))
    try:
        plan = LeafPlan(n, (16, n//16), 1, 16, 1, (), n, dtype="complex128" if wide else "complex64")
        lanes = hcu_dense_wave_lanes(plan, "packed_r2c")
    finally:
        reset_profile(token)
    name, source = build_ix_warp_source(n, pack, "packed_r2c", wide=wide, lane_width=lanes,
        table_roots=not wide, local_codelet=True, output_radix_chunk=1,
        packed_output_transpose=True)
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    roots = np.exp(-2j * np.pi * np.arange(n) / n)
    packed_roots = np.exp(-1j * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.astype(dtype)),
        "wave_tw_i_ptr": Pointer(roots.imag.astype(dtype)),
        "packed_twiddle_ptr": Pointer(packed_roots.astype(output.dtype).view(dtype))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<packed-transposed-fft>", "exec"), namespace)
    for pid in range((batch + pack - 1) // pack):
        monkeypatch.setattr(NumpyTL, "pid", pid)
        namespace[name](Pointer(data), Pointer(output.view(dtype)), span, batch, 2*n, n+1)
    np.testing.assert_allclose(output, expected,
        rtol=5e-13 if wide else 2e-5, atol=5e-11 if wide else 5e-5, equal_nan=True)


@pytest.mark.parametrize("n", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("wide", [False, True])
def test_compact_packed_c2r_matches_independent_numpy_inverse(monkeypatch, n, wide):
    lanes = min(64, max(16, n//(16 if wide else 32)))
    pack, batch = 4, 3
    real_dtype = np.float64 if wide else np.float32
    complex_dtype = np.complex128 if wide else np.complex64
    real = np.random.default_rng(22).normal(size=(batch, 2*n))
    data = np.fft.rfft(real, axis=-1).astype(complex_dtype)
    expected = np.fft.irfft(data, n=2*n, axis=-1)*2*n
    output = np.full((batch, 2*n), np.nan, dtype=real_dtype)
    name, source = build_ix_warp_source(n, pack, "packed_c2r", True, wide=wide,
        table_roots=not wide, lane_width=lanes, local_codelet=True)
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    roots = np.exp(2j*np.pi*np.arange(n)/n)
    packed_roots = np.exp(-1j*np.pi*np.arange(n)/n).astype(complex_dtype)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.astype(real_dtype)),
        "wave_tw_i_ptr": Pointer(roots.imag.astype(real_dtype)),
        "packed_twiddle_ptr": Pointer(packed_roots.view(real_dtype))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<compact-packed-inverse>", "exec"), namespace)
    monkeypatch.setattr(NumpyTL, "pid", 0)
    namespace[name](Pointer(data.view(real_dtype)), Pointer(output), 1, batch, n+1, n)
    error = output-expected
    relative = np.linalg.norm(error, axis=-1)/np.linalg.norm(expected, axis=-1)
    assert np.max(relative) < (4e-15 if wide else 4e-7)
    relative_max = np.max(np.abs(error), axis=-1)/np.max(np.abs(expected), axis=-1)
    assert np.max(relative_max) < (6e-15 if wide else 6e-7)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("layout", ["wave", "compact"])
def test_cli_transports_the_transposed_packed_real_layout(tmp_path, monkeypatch, capsys, dtype, layout):
    import json
    from flagfft_codegen import cli
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_REAL_STORE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", layout)
    device = {"backend": "hcu", "device_arch": "gfx936", "warp_size": 64,
              "max_threads_per_block": 1024, "max_dynamic_shared_memory": 65536}
    monkeypatch.setattr(sys, "argv", ["codegen", "--kernel", "leaf_packed_r2c",
        "--length", "128", "--factors", "8,4,4", "--lanes", "16", "--num-warps", "1",
        "--smem-size", "128", "--direction", "forward", "--dtype", dtype,
        "--perm-form", "outer_packed", "--target", "hcu:gfx936:64",
        "--device-profile", json.dumps(device), "--execution-policy", "native",
        "--out-dir", str(tmp_path)])
    token = set_profile(BackendProfile())
    try:
        cli.main()
    finally:
        reset_profile(token)
    metadata = json.loads(capsys.readouterr().out)
    source = Path(metadata["module_path"]).read_text()
    assert "_transpose" in metadata["kernel_name"]
    assert "outer_packed" in metadata["module_path"]
    assert "span=output_distance" in source
    assert ("_lanes16" in metadata["kernel_name"]) == (layout == "compact")


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [(128, (8, 4, 4), 16), (1024, (16, 8, 8), 64)])
def test_transposed_packed_leaf_preserves_span_abi_and_native_fallback_names(
        tmp_path, monkeypatch, dtype, n, factors, lanes):
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_REAL_STORE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        meta = emit_jit_kernel(kernel="leaf_packed_r2c", length=n, factors=factors,
            lanes=lanes, num_warps=2, generic_radices=(), smem_size=n,
            direction="forward", dtype=dtype, prime_n=0, four_step_n1=0, four_step_n2=0,
            perm_form="outer_packed", out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(meta["module_path"]).read_text()
    compile(source, "<hcu-packed-transpose>", "exec")
    assert "outer_packed" in Path(meta["module_path"]).name
    assert "_transpose" in meta["kernel_name"]
    assert "span=output_distance" in source
    assert meta["arg_names"][-3:] == ["input_distance", "output_distance", "nbatch"]
    assert 0 < meta["batch_per_block"] <= 16
    assert meta["num_warps"] == min(8, meta["batch_per_block"])


@pytest.mark.parametrize("mode,inverse", [("contiguous", False), ("contiguous", True),
                                          ("contiguous_r2c", False), ("c2r", True),
                                          ("packed_r2c", False), ("packed_c2r", True)])
@pytest.mark.parametrize("period", [0, 4, 8])
@pytest.mark.parametrize("codelet", [False, True])
def test_two_wave_generated_fft_matches_numpy_with_masked_tail(mode, inverse, period, codelet):
    packed = mode.startswith("packed_")
    n, batch, pack = (1024 if packed else 2048), 3, 4
    rng = np.random.default_rng(716)
    real = rng.normal(size=(batch, 2*n if packed else n))
    if mode in {"contiguous_r2c", "packed_r2c"}:
        data = real
        expected = np.fft.rfft(real, axis=-1)
    elif mode in {"c2r", "packed_c2r"}:
        data = np.fft.rfft(real, axis=-1)
        expected = real * real.shape[-1]
        if packed:
            expected = expected.view(np.complex128)
    else:
        data = real + 1j * rng.normal(size=(batch, n))
        expected = np.fft.ifft(data, axis=-1) * n if inverse else np.fft.fft(data, axis=-1)
    input_distance = data.shape[-1]
    output_distance = expected.shape[-1]
    output = np.full(expected.shape, np.nan, dtype=expected.dtype)
    name, source = build_ix_warp_source(n, pack, mode, inverse, wide=True,
                                      lane_width=64 if packed or period else 128,
                                      output_radix_chunk=4, root_period=period,
                                      local_codelet=codelet)
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    cpu_module = ast.Module(body=functions, type_ignores=[])
    roots = np.exp((1j if inverse else -1j) * 2 * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.copy()),
                 "wave_tw_i_ptr": Pointer(roots.imag.copy())}
    packed_roots = np.exp(-1j * np.pi * np.arange(n) / n)
    namespace["packed_twiddle_ptr"] = Pointer(packed_roots.view(np.float64))
    exec(compile(cpu_module, "<emitted-wide-fft>", "exec"), namespace)
    namespace[name](Pointer(data.view(np.float64)), Pointer(output.view(np.float64)),
                    1, batch, input_distance, output_distance)
    np.testing.assert_allclose(output, expected, rtol=5e-13, atol=5e-11)
    error = output - expected
    assert np.max(np.linalg.norm(error, axis=-1) / np.linalg.norm(expected, axis=-1)) < 4e-15
    assert np.max(np.max(np.abs(error), axis=-1) / np.max(np.abs(expected), axis=-1)) < 6e-15


@pytest.mark.parametrize("n", [256, 2048])
@pytest.mark.parametrize("mode,inverse", [("outer", False), ("outer", True),
                                          ("inner", False), ("inner", True), ("r2c", False)])
@pytest.mark.parametrize("period", [0, 4])
def test_wide_transposed_fft_matches_numpy_with_partial_tiles(monkeypatch, n, mode, inverse, period):
    pack = 4
    span = 4 if mode == "r2c" else 3
    batch = 10 if mode == "r2c" else 15 if mode == "inner" else 9
    rng = np.random.default_rng(20261116)
    data = rng.normal(size=(batch, n))
    if mode == "r2c":
        transformed = np.fft.rfft(data, axis=-1)
    else:
        data = data + 1j * rng.normal(size=(batch, n))
        transformed = np.fft.ifft(data, axis=-1) * n if inverse else np.fft.fft(data, axis=-1)
    if mode == "inner":
        expected = transformed.reshape(batch // span, span, n).transpose(2, 1, 0).ravel()
        programs = ((batch + pack * span - 1) // (pack * span)) * span
    else:
        rows = ((batch + span - 1) // span) * span
        padded = np.full((rows, transformed.shape[-1]), np.nan, dtype=np.complex128)
        padded[:batch] = transformed
        expected = padded.reshape(-1, span, transformed.shape[-1]).transpose(0, 2, 1).ravel()
        programs = (rows // span) * ((span + pack - 1) // pack)
    output = np.full(expected.shape, np.nan, dtype=np.complex128)
    lanes = 128 if n > 1024 and not period else 64
    name, source = build_ix_warp_source(n, pack, mode, inverse, wide=True,
                                      lane_width=lanes, output_radix_chunk=4, root_period=period)
    source = "\n".join(line for line in source.splitlines()
                       if not line.startswith(("import ", "@triton.jit")))
    roots = np.exp((1j if inverse else -1j) * 2 * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.copy()),
                 "wave_tw_i_ptr": Pointer(roots.imag.copy())}
    exec(compile(source, "<emitted-wide-permutation>", "exec"), namespace)
    for pid in range(programs):
        monkeypatch.setattr(NumpyTL, "pid", pid)
        namespace[name](Pointer(data.view(np.float64)), Pointer(output.view(np.float64)),
                        span, batch, n, transformed.shape[-1])
    np.testing.assert_allclose(output, expected, rtol=5e-13, atol=5e-11)


@pytest.mark.parametrize("kind,n,pack,lanes", [("leaf_permuted_store", 256, 16, 64),
    ("leaf_permuted_store", 1024, 8, 64),
    ("leaf_r2c_permuted_store", 256, 16, 64)])
def test_wide_permutation_preserves_double_roots_and_payload_bounds(tmp_path, monkeypatch, kind, n, pack, lanes):
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    factors = (16, 16, 8) if n == 2048 else (16, 8, 8) if n == 1024 else (16, 16)
    try:
        metadata = emit_jit_kernel(kernel=kind, length=n, factors=factors,
            lanes=16, num_warps=1, generic_radices=(), smem_size=n,
            direction="forward", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, perm_form="outer",
            permutation_span=128, permutation_batch=32768, out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-wide-permutation>", "exec")
    assert metadata["kernel_name"].startswith("hcu_warp")
    assert "tl.float32" not in source and "tl.uint64" not in source
    assert f"wave_tw_r_ptr=tw{len(factors)-1}_r_ptr+" in source
    assert metadata["batch_per_block"] == pack
    assert metadata["num_warps"] == min(8, pack * lanes // 64)
    assert n * pack * 2 <= metadata["num_warps"] * 64 * 32


def test_wide_permutation_keeps_native_fallback_above_one_wave_payload(monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_hcu_warp import hcu_warp_store_pack
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        plan = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048, dtype="complex128")
        assert hcu_warp_store_pack(plan, "permuted_store", "inner", 64, 32768) == 0
        plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype="complex128")
        assert hcu_warp_store_pack(plan, "permuted_store_r2c", "outer", 997, 997*16) == 0
    finally:
        reset_profile(token)


@pytest.mark.parametrize("kind,pack", [("leaf", 4), ("leaf_permuted_store", 4)])
def test_periodic_coefficients_keep_large_wide_fft_inside_one_wave(tmp_path, monkeypatch, kind, pack):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "leaves")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_PERIOD", "4")
    monkeypatch.setenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "64")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        metadata = emit_jit_kernel(kernel=kind, length=2048, factors=(16, 16, 8),
            lanes=128, num_warps=4, generic_radices=(), smem_size=2048,
            direction="inverse", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, perm_form="outer",
            permutation_span=1025 if kind.endswith("store") else 0,
            permutation_batch=4100 if kind.endswith("store") else 0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(metadata["module_path"]).read_text()
    assert "_period4_" in metadata["kernel_name"]
    assert "_lanes128" not in metadata["kernel_name"]
    assert metadata["batch_per_block"] == pack
    assert metadata["num_warps"] == pack
    assert "base_wr=wr" in source and "wr*base_wr-wi*base_wi" in source


def test_wide_logical_lanes_share_launch_and_memory_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "auto")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        metadata = emit_jit_kernel(kernel="leaf", length=2048, factors=(16, 16, 8),
            lanes=128, num_warps=4, generic_radices=(), smem_size=2048,
            direction="forward", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    assert "_lanes128_" in metadata["kernel_name"]
    assert metadata["num_warps"] == 8
    assert metadata["batch_per_block"] == 4
    assert 4 * 128 * 4 * 16 <= 32768
