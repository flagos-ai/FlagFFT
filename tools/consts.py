"""Shared constants for FlagFFT test runner."""

from __future__ import annotations

# Mapping from API type to precision label
DTYPE_MAP: dict[str, str] = {
    "c2c": "fp32",
    "z2z": "fp64",
    "r2c": "fp32",
    "c2r": "fp32",
    "d2z": "fp64",
    "z2d": "fp64",
}

# Default test timeout in seconds
DEFAULT_TIMEOUT = 600

# Default benchmark parameters
BENCH_WARMUP = 10
BENCH_ITERS = 100
