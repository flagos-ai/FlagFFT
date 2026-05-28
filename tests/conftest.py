"""Test conftest: loads the shared pytest plugin with test-verification defaults.

Tests use warm=1 iter=1 for quick CLI verification. No CSV output by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path before importing benchmark
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── Set test-verification defaults before loading the shared plugin ──
from benchmark.utils import pytest_plugin as _plugin  # noqa: E402

_plugin._DEFAULTS.update(
    {
        "warmup": 1,
        "iters": 1,
        "csv": "",  # disabled by default for verification tests
        "suite": "full",
    }
)

pytest_plugins = ["benchmark.utils.pytest_plugin"]
