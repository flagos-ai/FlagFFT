from __future__ import annotations

import pytest
import torch

import flagfft
from src.tuning import TuneConfig, _batch_bucket, _connect, _insert_measurement, _problem_fields

_flagfft_core = pytest.importorskip("_flagfft_core")


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _sample(dtype: torch.dtype, n: int = 16) -> torch.Tensor:
    torch.manual_seed(1234)
    if dtype.is_complex:
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        real = torch.randn(2, n, device="cuda", dtype=real_dtype)
        imag = torch.randn(2, n, device="cuda", dtype=real_dtype)
        return torch.complex(real, imag).to(dtype)
    return torch.randn(2, n, device="cuda", dtype=dtype)


def _tuned_fields(x: torch.Tensor) -> dict[str, object]:
    request = dict(_flagfft_core.debug_request(x))
    fps = dict(_flagfft_core.tune_fingerprints())
    return {
        "schema_version": int(request.get("schema_version", 2)),
        "device_arch": request["device_arch"],
        "fft_length": request["requested_n"],
        "batch_bucket": _batch_bucket(request["batch"]),
        "batch": request["batch"],
        "dtype": request["input_dtype"],
        "direction": request["direction"],
        "norm": request["norm"],
        "input_layout": request["input_layout"],
        "planner_fingerprint": fps["planner"],
        "codegen_fingerprint": fps["codegen"],
        "runtime_fingerprint": fps["runtime"],
        "benchmark_fingerprint": fps["benchmark"],
    }


def _write_tuned_winner(db, x: torch.Tensor, plan=None):
    plan = plan if plan is not None else _flagfft_core.enumerate_plan_candidates(x)[0]
    fields = _tuned_fields(x)
    fields["schema_version"] = plan["schema_version"]
    with _connect(db) as conn:
        _insert_measurement(conn, fields, plan, status="valid", rank=0)
        conn.commit()
    return plan


@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "atol", "rtol"),
    [
        (torch.float32, torch.complex64, 2e-5, 2e-5),
        (torch.complex64, torch.complex64, 2e-5, 2e-5),
        (torch.float64, torch.complex128, 1e-12, 1e-12),
        (torch.complex128, torch.complex128, 1e-12, 1e-12),
    ],
)
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft_matches_torch_for_supported_dtypes(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
    atol: float,
    rtol: float,
    norm: str | None,
) -> None:
    x = _sample(dtype)

    y = flagfft.fft(x, norm=norm)
    ref = torch.fft.fft(x, dim=-1, norm=norm)

    assert y.dtype is expected_dtype
    torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)


def test_out_returns_same_object_and_copies_result() -> None:
    x = _sample(torch.float32)
    out = torch.empty(x.shape, device=x.device, dtype=torch.complex64)

    y = flagfft.fft(x, out=out)

    assert y is out
    torch.testing.assert_close(out, torch.fft.fft(x, dim=-1), atol=2e-5, rtol=2e-5)


def test_request_schema_exists_before_unsupported_errors() -> None:
    x = _sample(torch.complex64, n=8)

    n_request = _flagfft_core.debug_request(x, 4, -1, None)
    assert n_request["requested_n"] == 4
    assert n_request["length"] == 8
    with pytest.raises(NotImplementedError, match="padding or trimming"):
        flagfft.fft(x, n=4)

    dim_request = _flagfft_core.debug_request(x, None, 0, None)
    assert dim_request["dim"] == 0
    with pytest.raises(NotImplementedError, match="last dimension"):
        flagfft.fft(x, dim=0)


def test_debug_keys_cover_semantic_and_layout_fields() -> None:
    x = _sample(torch.complex64, n=16)
    strided = torch.randn(2, 32, device="cuda", dtype=torch.complex64)[:, ::2]

    default_key = _flagfft_core.debug_keys(x)["problem"]["repr"]
    forward_key = _flagfft_core.debug_keys(x, None, -1, "forward")["problem"]["repr"]
    strided_keys = _flagfft_core.debug_keys(strided)
    strided_key = strided_keys["problem"]["repr"]

    assert default_key != forward_key
    assert default_key != strided_key
    assert strided_keys["problem"]["requires_contiguous_copy"] is True
    assert _flagfft_core.debug_keys(x)["plan"]["kind"] == "ct_leaf"
    assert len(_flagfft_core.debug_keys(x)["kernels"]) >= 1


def test_cpp_plan_cache_hits_on_repeated_call() -> None:
    x = _sample(torch.complex64, n=16)
    _flagfft_core.clear_plan_cache()

    flagfft.fft(x)
    after_first = _flagfft_core.cache_info()
    flagfft.fft(x)
    after_second = _flagfft_core.cache_info()

    assert after_first["problem_size"] == 1
    assert after_first["problem_misses"] == 1
    assert after_first["plan_size"] == 1
    assert after_second["problem_size"] == 1
    assert after_second["problem_hits"] == 1
    assert after_second["plan_size"] == 1


def test_plan_and_kernel_caches_reuse_same_route_for_distinct_problems() -> None:
    x = _sample(torch.complex64, n=16)
    y = torch.randn(3, 16, device="cuda", dtype=torch.complex64)
    _flagfft_core.clear_plan_cache()

    flagfft.fft(x)
    after_first = _flagfft_core.cache_info()
    flagfft.fft(y)
    after_second = _flagfft_core.cache_info()

    assert after_first["problem_size"] == 1
    assert after_second["problem_size"] == 2
    assert after_second["problem_misses"] == 2
    assert after_second["plan_size"] == after_first["plan_size"] == 1
    assert after_second["plan_hits"] == after_first["plan_hits"] + 1
    assert after_second["kernel_size"] == after_first["kernel_size"]
    assert after_second["kernel_hits"] > after_first["kernel_hits"]


@pytest.mark.parametrize("n", [8, 16, 105])
def test_cpp_aot_leaf_lengths_match_torch(n: int) -> None:
    x = _sample(torch.complex64, n=n)

    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)

    assert _flagfft_core.debug_plan(x)["root"]["kind"] == "ct_leaf"
    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("n", [4096, 8192, 16384])
def test_cpp_aot_four_step_matches_torch(n: int) -> None:
    x = _sample(torch.complex64, n=n)[:1]

    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)

    assert _flagfft_core.debug_plan(x)["root"]["kind"] == "four_step"
    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


def test_cpp_aot_four_step_leaf_children_use_fused_kernels() -> None:
    x = _sample(torch.complex64, n=16384)[:1]

    kernels = _flagfft_core.debug_keys(x)["kernels"]
    kernel_kinds = {kernel["kind"] for kernel in kernels}

    assert {"four_step_row", "four_step_col"} <= kernel_kinds
    assert "transpose" not in kernel_kinds
    assert "twiddle_transpose" not in kernel_kinds


def test_cpp_aot_four_step_accepts_float32_input() -> None:
    x = _sample(torch.float32, n=8192)[:1]

    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)

    assert y.dtype is torch.complex64
    assert _flagfft_core.debug_plan(x)["root"]["kind"] == "four_step"
    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("n", [331, 997])
def test_cpp_aot_bluestein_lengths_match_torch(n: int) -> None:
    x = _sample(torch.complex64, n=n)[:1]

    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)
    root = _flagfft_core.debug_plan(x)["root"]

    assert root["kind"] == "bluestein"
    assert root["conv_length"] >= 2 * n - 1
    torch.testing.assert_close(y, ref, atol=5e-4, rtol=5e-4)


def test_cpp_aot_large_bluestein_reuses_child_fft_kernels() -> None:
    _flagfft_core.clear_plan_cache()
    first = torch.randn(1, 65537, device="cuda", dtype=torch.complex64)
    second = torch.randn(1, 65539, device="cuda", dtype=torch.complex64)

    first_plan = _flagfft_core.debug_plan(first)["root"]
    second_plan = _flagfft_core.debug_plan(second)["root"]
    assert first_plan["kind"] == second_plan["kind"] == "bluestein"
    assert first_plan["conv_length"] == second_plan["conv_length"]

    torch.testing.assert_close(
        flagfft.fft(first), torch.fft.fft(first, dim=-1), atol=8e-4, rtol=8e-4
    )
    after_first = _flagfft_core.cache_info()
    torch.testing.assert_close(
        flagfft.fft(second), torch.fft.fft(second, dim=-1), atol=8e-4, rtol=8e-4
    )
    after_second = _flagfft_core.cache_info()

    assert after_second["kernel_hits"] >= after_first["kernel_hits"] + 2


@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_cpp_aot_four_step_norm_modes(norm: str | None) -> None:
    x = _sample(torch.complex64, n=8192)[:1]

    y = flagfft.fft(x, norm=norm)
    ref = torch.fft.fft(x, dim=-1, norm=norm)

    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


def test_cpp_aot_four_step_plan_cache_hits_on_repeated_call() -> None:
    x = _sample(torch.complex64, n=8192)[:1]
    _flagfft_core.clear_plan_cache()

    flagfft.fft(x)
    after_first = _flagfft_core.cache_info()
    flagfft.fft(x)
    after_second = _flagfft_core.cache_info()

    assert after_first["problem_size"] == 1
    assert after_first["problem_misses"] == 1
    assert after_first["plan_size"] == 1
    assert after_second["problem_size"] == 1
    assert after_second["problem_misses"] == 1
    assert after_second["problem_hits"] == 1
    assert after_second["plan_size"] == 1


def test_debug_plan_returns_cpp_built_tree() -> None:
    x = _sample(torch.complex64, n=16)

    plan = _flagfft_core.debug_plan(x)

    assert plan["source"] == "cpp_auto"
    assert plan["request"]["length"] == 16
    assert plan["plan_key"]["kind"] == plan["root"]["kind"]
    assert plan["root"]["kind"] in {"ct_leaf", "four_step", "direct_dft"}


def test_enumerate_plan_candidates_and_forced_plan_roundtrip() -> None:
    x = _sample(torch.complex64, n=64)

    candidates = list(_flagfft_core.enumerate_plan_candidates(x))

    assert candidates
    assert len(candidates) <= 32
    forced = _flagfft_core.debug_forced_plan(x, candidates[0])
    assert forced["source"] == "forced"
    assert forced["plan_key"]["repr"] == candidates[0]["plan_key"]["repr"]

    y = _flagfft_core.fft_with_plan(x, candidates[0])
    ref = torch.fft.fft(x, dim=-1)
    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


def test_tuned_db_winner_is_used_when_fingerprints_match(tmp_path, monkeypatch) -> None:
    x = _sample(torch.complex64, n=64)
    plan = _flagfft_core.enumerate_plan_candidates(x)[0]
    db = tmp_path / "tuned.sqlite"
    _write_tuned_winner(db, x, plan)

    monkeypatch.setenv("FLAGFFT_TUNE_DB", str(db))
    monkeypatch.delenv("FLAGFFT_TUNE_DISABLE", raising=False)
    _flagfft_core.clear_plan_cache()

    y = flagfft.fft(x)
    resolved = _flagfft_core.debug_resolved_plan(x)
    ref = torch.fft.fft(x, dim=-1)

    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)
    assert resolved["source"] == "sqlite_tuned"
    assert resolved["plan_key"]["repr"] == plan["plan_key"]["repr"]


def test_tuning_problem_fields_use_current_plan_schema() -> None:
    x = _sample(torch.complex64, n=64)

    fields = _problem_fields(_flagfft_core, x, TuneConfig(lengths=(64,)))

    assert fields["schema_version"] == _flagfft_core.debug_plan(x)["schema_version"]


def test_default_tuned_db_path_is_used_when_env_is_unset(tmp_path, monkeypatch) -> None:
    x = _sample(torch.complex64, n=64)
    plan = _write_tuned_winner(tmp_path / ".flagfft" / "tuned_plans.sqlite", x)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLAGFFT_TUNE_DB", raising=False)
    monkeypatch.delenv("FLAGFFT_TUNE_DISABLE", raising=False)
    _flagfft_core.clear_plan_cache()

    resolved = _flagfft_core.debug_resolved_plan(x)

    assert resolved["source"] == "sqlite_tuned"
    assert resolved["plan_key"]["repr"] == plan["plan_key"]["repr"]


def test_tuned_db_disable_skips_matching_winner(tmp_path, monkeypatch) -> None:
    x = _sample(torch.complex64, n=64)
    db = tmp_path / "tuned.sqlite"
    _write_tuned_winner(db, x)

    monkeypatch.setenv("FLAGFFT_TUNE_DB", str(db))
    monkeypatch.setenv("FLAGFFT_TUNE_DISABLE", "1")
    _flagfft_core.clear_plan_cache()

    resolved = _flagfft_core.debug_resolved_plan(x)

    assert resolved["source"] == "cpp_auto"


def test_warm_problem_cache_hit_does_not_recheck_tuned_db(tmp_path, monkeypatch) -> None:
    x = _sample(torch.complex64, n=64)
    db = tmp_path / "tuned.sqlite"
    _write_tuned_winner(db, x)

    monkeypatch.setenv("FLAGFFT_TUNE_DB", str(db))
    monkeypatch.delenv("FLAGFFT_TUNE_DISABLE", raising=False)
    _flagfft_core.clear_plan_cache()

    flagfft.fft(x)
    after_first = _flagfft_core.cache_info()
    monkeypatch.setenv("FLAGFFT_TUNE_DISABLE", "1")
    flagfft.fft(x)
    after_second = _flagfft_core.cache_info()

    assert after_first["problem_misses"] == 1
    assert after_first["tuned_db_lookups"] == 1
    assert after_second["problem_hits"] == after_first["problem_hits"] + 1
    assert after_second["tuned_db_lookups"] == after_first["tuned_db_lookups"]
