"""Benchmark conftest: loads the shared pytest plugin with benchmark defaults.

If pytest-xdist is installed, add -p no:xdist to force serial execution.
"""

from __future__ import annotations

# ── Set benchmark defaults before loading the shared plugin ─────
from benchmark.utils import pytest_plugin as _plugin

_plugin._DEFAULTS.update(
    {
        "warmup": 10,
        "iters": 100,
        "csv": None,  # auto-generate path under benchmark/results/
        "suite": "typical",
    }
)

pytest_plugins = ["benchmark.utils.pytest_plugin"]
