from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import benchmark_fft_mixed_radix as bench


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--lengths", "16"], None),
        (["--lengths", "16", "--tune"], "tune"),
        (["--lengths", "16", "--tune", "tune"], "tune"),
        (["--lengths", "16", "--tune", "retune"], "retune"),
    ],
)
def test_tune_argument_modes(argv: list[str], expected: str | None) -> None:
    args = bench._parser().parse_args(argv)

    assert args.tune == expected


def test_api_argument_accepts_ifft() -> None:
    args = bench._parser().parse_args(["--api", "ifft", "--lengths", "16"])

    assert args.api == "ifft"


def test_run_benchmark_tunes_each_length_before_benchmark(monkeypatch, capsys) -> None:
    tune_calls: list[dict[str, object]] = []
    benchmark_calls: list[int] = []

    def fake_tune(api, **kwargs):
        tune_calls.append({"api": getattr(api, "__name__", "?"), **kwargs})

    def fake_benchmark_once(
        n: int, batch: int, warmup: int, iters: int, api: bench.ApiName = "fft"
    ) -> dict[str, object]:
        benchmark_calls.append(n)
        return {
            "backend": "cpp",
            "api": api,
            "plan_source": "cpp_tuned",
            "plan": "ct_leaf(n=16, factors=[4, 4], lanes=16)",
            "flagfft_median_ms": 1.0,
            "torch_median_ms": 2.0,
            "flagfft_first_call_ms": 3.0,
            "cpp_cache_after_warm": {
                "problem_size": 1,
                "problem_hits": 1,
                "problem_misses": 1,
                "plan_size": 1,
                "kernel_size": 1,
            },
        }

    monkeypatch.setattr(bench, "flagfft_tune", fake_tune)
    monkeypatch.setattr(bench, "benchmark_mixed_radix_once", fake_benchmark_once)

    bench.run_benchmark([16, 32], batch=3, warmup=4, iters=5, tune_mode="retune", api="ifft")

    assert tune_calls == [
        {"api": "ifft", "lengths": [16], "batch": 3, "retune": True, "warmup": 4, "iters": 5},
        {"api": "ifft", "lengths": [32], "batch": 3, "retune": True, "warmup": 4, "iters": 5},
    ]
    assert benchmark_calls == [16, 32]
    assert "[api=ifft tune=retune n=16 batch=3]" in capsys.readouterr().out
