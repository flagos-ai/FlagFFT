from __future__ import annotations

import pytest
import torch

import flagfft

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


def test_cpp_aot_four_step_accepts_float32_input() -> None:
    x = _sample(torch.float32, n=8192)[:1]

    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)

    assert y.dtype is torch.complex64
    assert _flagfft_core.debug_plan(x)["root"]["kind"] == "four_step"
    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)


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
    import json
    import sqlite3

    x = _sample(torch.complex64, n=64)
    plan = _flagfft_core.enumerate_plan_candidates(x)[0]
    request = _flagfft_core.debug_request(x)
    fps = _flagfft_core.tune_fingerprints()
    db = tmp_path / "tuned.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE tuned_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version INTEGER NOT NULL,
            device_arch TEXT NOT NULL,
            fft_length INTEGER NOT NULL,
            batch_bucket TEXT NOT NULL,
            batch INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            direction TEXT NOT NULL,
            norm TEXT NOT NULL,
            input_layout TEXT NOT NULL,
            planner_fingerprint TEXT NOT NULL,
            codegen_fingerprint TEXT NOT NULL,
            runtime_fingerprint TEXT NOT NULL,
            benchmark_fingerprint TEXT NOT NULL,
            plan_key TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            rank INTEGER,
            compile_ms REAL,
            first_call_ms REAL,
            median_ms REAL,
            p90_ms REAL,
            max_abs_err REAL,
            rms_err REAL,
            failure_reason TEXT,
            measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tuned_measurements (
            schema_version, device_arch, fft_length, batch_bucket, batch, dtype, direction,
            norm, input_layout, planner_fingerprint, codegen_fingerprint, runtime_fingerprint,
            benchmark_fingerprint, plan_key, plan_json, status, rank
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'valid', 0)
        """,
        (
            1,
            request["device_arch"],
            request["requested_n"],
            "2-8",
            request["batch"],
            request["input_dtype"],
            request["direction"],
            request["norm"],
            request["input_layout"],
            fps["planner"],
            fps["codegen"],
            fps["runtime"],
            fps["benchmark"],
            plan["plan_key"]["repr"],
            json.dumps(plan, sort_keys=True),
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("FLAGFFT_TUNE_DB", str(db))
    _flagfft_core.clear_plan_cache()

    y = flagfft.fft(x)
    resolved = _flagfft_core.debug_resolved_plan(x)
    ref = torch.fft.fft(x, dim=-1)

    torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)
    assert resolved["source"] == "sqlite_tuned"
    assert resolved["plan_key"]["repr"] == plan["plan_key"]["repr"]
