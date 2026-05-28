# Benchmark Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split benchmark/ into a shared plugin + real benchmark pytest cases, and add tests/test_bench_cli.py for quick CLI verification, both producing console + CSV reports via pytest_sessionfinish.

**Architecture:** A shared pytest plugin (`benchmark/utils/pytest_plugin.py`) provides CLI invocation, result collection, and report generation. `benchmark/conftest.py` and `tests/conftest.py` both load this plugin but with different defaults (benchmark: full warmup+CSV; tests: warm=1,iter=1,no CSV). `pytest_generate_tests` filters suites at collection time. `-p no:xdist` is enforced for serial benchmark execution.

**Tech Stack:** Python 3.12, pytest 9.0, subprocess for CLI invocation, csv module for CSV output.

---

### Task 1: Register `bench` marker in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add [tool.pytest.ini_options] with bench marker**

```toml
[tool.pytest.ini_options]
markers = ["bench: performance benchmark tests (require serial execution, use -p no:xdist)"]
```

Append to the end of `pyproject.toml`.

- [ ] **Step 2: Verify marker is recognized**

Run: `pytest --markers 2>&1 | grep bench`
Expected: line containing `bench: performance benchmark tests`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: register pytest bench marker in pyproject.toml

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Create `benchmark/utils/` package and `__init__.py`

**Files:**
- Create: `benchmark/utils/__init__.py`

- [ ] **Step 1: Create benchmark/utils/__init__.py**

```python
"""Benchmark utilities: suites, reporting, result collection, pytest plugin."""
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "import benchmark.utils"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add benchmark/utils/__init__.py
git commit -m "feat: create benchmark/utils package

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Move and expand `benchmark/utils/suites.py`

**Files:**
- Create: `benchmark/utils/suites.py`
- Delete: `benchmark/suites.py`

- [ ] **Step 1: Create benchmark/utils/suites.py**

```python
"""Benchmark size suites for FlagFFT performance testing."""

# All sizes from ctest k1DSizesSmall + k1DSizesMedium + k1DSizesLarge
ALL_SIZES = [16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384]

# All batch values from ctest k1DBatchValues
ALL_BATCHES = [1, 4, 256]

# All APIs from ctest coverage
ALL_APIS = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]

# Direction support per API
API_DIRECTIONS = {
    "c2c": ["forward", "inverse"],
    "z2z": ["forward", "inverse"],
    "r2c": ["forward"],
    "d2z": ["forward"],
    "c2r": ["inverse"],
    "z2d": ["inverse"],
}

# ── Suite definitions ──────────────────────────────────────────

# Smoke (24 cases): 3 key sizes x batch=1 x all API/direction combos
SMOKE = {"sizes": [16, 256, 2048], "batches": [1]}

# Typical (136 cases): all sizes x batch=1 + extra multi-batch on key sizes
TYPICAL = {
    "sizes": ALL_SIZES,
    "batches": [1],
    "extra": [(256, 4), (256, 256), (2048, 4), (2048, 256)],
}

# Full (312 cases): all sizes x all batches x all API/direction combos
FULL = {"sizes": ALL_SIZES, "batches": ALL_BATCHES}

SUITES = {"smoke": SMOKE, "typical": TYPICAL, "full": FULL}


def get_suite(name: str) -> dict:
    """Return suite definition by name."""
    return SUITES[name]


def expand_params(suite: dict):
    """Yield (size, batch, api, direction) tuples for a suite definition.

    Handles the 'extra' key for TYPICAL suite: additional (size, batch)
    pairs beyond the base sizes x batches cross product.
    """
    apis = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]
    seen = set()

    # Base: sizes x batches x apis
    for size in suite["sizes"]:
        for batch in suite["batches"]:
            for api in apis:
                for direction in API_DIRECTIONS[api]:
                    key = (size, batch, api, direction)
                    if key not in seen:
                        seen.add(key)
                        yield key

    # Extra (size, batch) pairs applied to all API/direction combos
    for size, batch in suite.get("extra", []):
        for api in apis:
            for direction in API_DIRECTIONS[api]:
                key = (size, batch, api, direction)
                if key not in seen:
                    seen.add(key)
                    yield key
```

- [ ] **Step 2: Verify expand_params counts match expectations**

Run:
```bash
python -c "
from benchmark.utils.suites import SMOKE, TYPICAL, FULL, expand_params
print('Smoke:', len(list(expand_params(SMOKE))))
print('Typical:', len(list(expand_params(TYPICAL))))
print('Full:', len(list(expand_params(FULL))))
"
```
Expected:
```
Smoke: 24
Typical: 136
Full: 312
```

- [ ] **Step 3: Delete old benchmark/suites.py**

```bash
git rm benchmark/suites.py
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/utils/suites.py && git rm benchmark/suites.py
git commit -m "feat: move suites to benchmark/utils, expand to ctest coverage

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Move and expand `benchmark/utils/report.py`

**Files:**
- Create: `benchmark/utils/report.py`
- Delete: `benchmark/report.py`

- [ ] **Step 1: Create benchmark/utils/report.py**

```python
"""Benchmark report generation: console tables, CSV, JSON, and Markdown output."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any


def _build_table(records: list[dict[str, Any]]) -> str:
    """Build a markdown table from benchmark records."""
    if not records:
        return "_No benchmark results._\n"

    headers = [
        "Size",
        "API",
        "Direction",
        "Batch",
        "FlagFFT (ms)",
        "cuFFT (ms)",
        "Speedup",
        "Correctness",
    ]
    rows = [headers]

    for r in records:
        correctness = r.get("correctness", {})
        passed = correctness.get("passed", False)
        timing = r.get("timing", {})
        rows.append(
            [
                str(r.get("size", "")),
                str(r.get("api", "")),
                str(r.get("direction", "")),
                str(r.get("batch", "")),
                f"{timing.get('flagfft_median_ms', 0):.4f}",
                f"{timing.get('cufft_median_ms', 0):.4f}",
                f"{timing.get('speedup', 0):.2f}x",
                "PASS" if passed else "FAIL",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for ri, row in enumerate(rows):
        line = (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )
        lines.append(line)
        if ri == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    return "\n".join(lines) + "\n"


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))


def generate_console_table(
    records: list[dict[str, Any]],
    suite: str,
    warmup: int,
    iters: int,
) -> str:
    """Generate a console-printable benchmark report with table and summary."""
    passed = sum(1 for r in records if r.get("correctness", {}).get("passed", False))
    total = len(records)

    lines = [
        "=" * 84,
        f"FlagFFT Benchmark Report  |  Suite: {suite}  |  Warmup: {warmup}  |  Iters: {iters}  |  Passed: {passed}/{total}",
        "",
        _build_table(records),
    ]

    if records:
        timings = [r.get("timing", {}) for r in records]
        flagfft_ms = [t.get("flagfft_median_ms", 0) for t in timings if t.get("flagfft_median_ms", 0) > 0]
        speedups = [t.get("speedup", 0) for t in timings if t.get("speedup", 0) > 0]

        lines.append(f"Correctness: {passed}/{total} passed")
        if flagfft_ms:
            lines.append(f"FlagFFT median range: {min(flagfft_ms):.4f} - {max(flagfft_ms):.4f} ms")
            lines.append(f"Geometric mean: {_geometric_mean(flagfft_ms):.4f} ms")
        if speedups:
            lines.append(f"Overall speedup (geomean): {_geometric_mean(speedups):.2f}x")

    return "\n".join(lines) + "\n"


CSV_COLUMNS = [
    "size", "api", "direction", "batch", "backend",
    "flagfft_median_ms", "cufft_median_ms", "speedup",
    "correctness_passed", "max_abs", "rms",
    "warmup", "iters",
]


def generate_csv(records: list[dict[str, Any]]) -> str:
    """Generate a CSV string from benchmark records."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        correctness = r.get("correctness", {})
        timing = r.get("timing", {})
        writer.writerow({
            "size": r.get("size", ""),
            "api": r.get("api", ""),
            "direction": r.get("direction", ""),
            "batch": r.get("batch", ""),
            "backend": r.get("backend", ""),
            "flagfft_median_ms": timing.get("flagfft_median_ms", 0),
            "cufft_median_ms": timing.get("cufft_median_ms", 0),
            "speedup": timing.get("speedup", 0),
            "correctness_passed": correctness.get("passed", False),
            "max_abs": correctness.get("max_abs", ""),
            "rms": correctness.get("rms", ""),
            "warmup": r.get("warmup", ""),
            "iters": r.get("iters", ""),
        })
    return output.getvalue()


def generate_markdown(report: dict[str, Any]) -> str:
    """Generate a Markdown benchmark report from a CLI JSON report."""
    cases = report.get("cases", [])
    records = []
    for case in cases:
        records.append(
            {
                "size": case.get("shape", ""),
                "api": case.get("api", ""),
                "direction": case.get("direction", ""),
                "batch": case.get("batch", 1),
                "correctness": case.get("correctness", {}),
                "timing": case.get("timing", {}),
            }
        )

    lines = ["# FlagFFT Benchmark Report\n"]
    lines.append(_build_table(records))

    passed = sum(1 for r in records if r["correctness"].get("passed", False))
    total = len(records)
    lines.append(f"**Correctness:** {passed}/{total} passed")
    if records:
        medians = [
            r["timing"].get("flagfft_median_ms", 0)
            for r in records
            if r["timing"].get("flagfft_median_ms", 0) > 0
        ]
        if medians:
            lines.append(
                f"**FlagFFT median range:** {min(medians):.4f} - {max(medians):.4f} ms"
            )
            lines.append(f"**Geometric mean:** {_geometric_mean(medians):.4f} ms")

    return "\n".join(lines) + "\n"


def generate_json_report(report: dict[str, Any]) -> str:
    """Generate a JSON benchmark report string (pretty-printed)."""
    return json.dumps(report, indent=2)
```

- [ ] **Step 2: Verify generation functions work with sample data**

Run:
```bash
python -c "
from benchmark.utils.report import generate_console_table, generate_csv
records = [{
    'size': 256, 'api': 'c2c', 'direction': 'forward', 'batch': 1,
    'backend': 'cuda', 'warmup': 10, 'iters': 100,
    'correctness': {'passed': True, 'max_abs': 1.2e-6, 'rms': 3.4e-7},
    'timing': {'flagfft_median_ms': 0.123, 'cufft_median_ms': 0.200, 'speedup': 1.63},
}]
print(generate_console_table(records, 'smoke', 10, 100))
print('---CSV---')
print(generate_csv(records))
"
```
Expected: table output and CSV with header row

- [ ] **Step 3: Delete old benchmark/report.py**

```bash
git rm benchmark/report.py
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/utils/report.py && git rm benchmark/report.py
git commit -m "feat: move report to benchmark/utils, add CSV and console table generation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Create `benchmark/utils/collector.py`

**Files:**
- Create: `benchmark/utils/collector.py`

- [ ] **Step 1: Create benchmark/utils/collector.py**

```python
"""Thread-safe session-scoped benchmark result collector."""

from __future__ import annotations

import threading


class BenchmarkCollector:
    """Collects benchmark results across a pytest session.

    Not a process-level singleton — held as a session-scoped fixture.
    Thread-safe for safety, though benchmarks run serially (-p no:xdist).
    """

    def __init__(self):
        self._results: list[dict] = []
        self._lock = threading.Lock()

    def add_result(self, case: dict) -> None:
        with self._lock:
            self._results.append(case)

    def get_results(self) -> list[dict]:
        with self._lock:
            return list(self._results)

    def clear(self) -> None:
        with self._lock:
            self._results.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)
```

- [ ] **Step 2: Verify it works**

Run:
```bash
python -c "
from benchmark.utils.collector import BenchmarkCollector
c = BenchmarkCollector()
c.add_result({'a': 1})
c.add_result({'a': 2})
assert len(c) == 2
assert c.get_results() == [{'a': 1}, {'a': 2}]
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark/utils/collector.py
git commit -m "feat: add BenchmarkCollector for session-scoped result aggregation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Create `benchmark/utils/pytest_plugin.py`

**Files:**
- Create: `benchmark/utils/pytest_plugin.py`

- [ ] **Step 1: Create benchmark/utils/pytest_plugin.py**

```python
"""Shared pytest plugin for benchmark CLI invocation, collection, and reporting.

Loaded by both benchmark/conftest.py and tests/conftest.py via
``pytest_plugins = ["benchmark.utils.pytest_plugin"]``.

Override defaults by setting keys in ``_DEFAULTS`` *before* the
``pytest_plugins`` assignment in the loading conftest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ── Overridable defaults ───────────────────────────────────────
# conftest files set values in this dict before loading the plugin.

_DEFAULTS = {
    "warmup": 10,
    "iters": 100,
    "csv": None,       # None = auto-generate path; "" = disabled
    "suite": "typical",
}

# ── pytest options ──────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--flagfft-cli",
        default=None,
        help="Path to flagfft-cli. Defaults to FLAGFFT_CLI_EXE or build/flagfft-cli.",
    )
    parser.addoption(
        "--bench-warmup",
        type=int,
        default=None,
        help=f"Number of warmup iterations (default: {_DEFAULTS['warmup']})",
    )
    parser.addoption(
        "--bench-iters",
        type=int,
        default=None,
        help=f"Number of benchmark iterations (default: {_DEFAULTS['iters']})",
    )
    parser.addoption(
        "--bench-csv",
        default=None,
        help="CSV output path. Empty string disables CSV. "
             f"Default: benchmark/results/benchmark_<timestamp>.csv",
    )
    parser.addoption(
        "--bench-suite",
        default=None,
        choices=["smoke", "typical", "full"],
        help=f"Benchmark suite (default: {_DEFAULTS['suite']})",
    )


# ── xdist guard ─────────────────────────────────────────────────

def pytest_configure(config):
    """Warn and disable bench tests if xdist is active."""
    if config.pluginmanager.hasplugin("xdist"):
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if not worker:  # controller process
            config.option.markexpr = _and_markexpr(
                config.option.markexpr, "not bench"
            )
            cap = config.get_verbosity() or 0
            sys.stderr.write(
                "\n⚠ xdist detected — bench tests disabled. "
                "Run with '-p no:xdist' for benchmarks.\n\n"
            )


def _and_markexpr(current: str, condition: str) -> str:
    if not current:
        return condition
    return f"{current} and {condition}"


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def flagfft_cli(request) -> Path:
    configured = request.config.getoption("--flagfft-cli")
    path = Path(
        configured or os.environ.get("FLAGFFT_CLI_EXE", ROOT / "build" / "flagfft-cli")
    )
    if not path.exists():
        pytest.skip(f"flagfft-cli is not built: {path}")
    return path


@pytest.fixture
def invoke_cli(flagfft_cli):
    def invoke(*arguments: str, env: dict[str, str] | None = None, timeout: int = 240):
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        result = subprocess.run(
            [str(flagfft_cli), *arguments, "--json"],
            cwd=ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if not result.stdout.strip() and (
            result.returncode == 77 or "cuInit failed" in result.stderr
        ):
            pytest.skip(result.stderr.strip() or "CLI skipped")
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            pytest.fail(f"invalid CLI JSON: {result.stdout}\n{result.stderr}\n{error}")
        if report.get("status") == "skipped":
            pytest.skip(report.get("reason", "CLI skipped"))
        return result, report

    return invoke


@pytest.fixture
def run_benchmark(invoke_cli, bench_warmup, bench_iters):
    """Invoke the bench subcommand with configured warmup/iters."""

    def _run(size: int, api: str = "c2c", direction: str = "forward", batch: int = 1):
        return invoke_cli(
            "bench",
            "--api", api,
            "--direction", direction,
            "--shape", str(size),
            "--batch", str(batch),
            "--warmup", str(bench_warmup),
            "--iters", str(bench_iters),
            "--print-path",
        )

    return _run


@pytest.fixture(scope="session")
def bench_warmup(request) -> int:
    val = request.config.getoption("--bench-warmup")
    if val is not None:
        return val
    return _DEFAULTS["warmup"]


@pytest.fixture(scope="session")
def bench_iters(request) -> int:
    val = request.config.getoption("--bench-iters")
    if val is not None:
        return val
    return _DEFAULTS["iters"]


@pytest.fixture(scope="session")
def bench_suite(request) -> str:
    val = request.config.getoption("--bench-suite")
    if val is not None:
        return val
    return _DEFAULTS["suite"]


@pytest.fixture(scope="session")
def bench_csv(request) -> str | None:
    """Resolved CSV path. None = auto, '' = disabled, otherwise user path."""
    val = request.config.getoption("--bench-csv")
    if val is not None:
        return val if val != "" else ""
    return _DEFAULTS["csv"]


@pytest.fixture(scope="session")
def bench_collector(request):
    from benchmark.utils.collector import BenchmarkCollector
    collector = BenchmarkCollector()
    request.config._bench_collector = collector
    return collector


@pytest.fixture
def record_result(bench_collector):
    return bench_collector.add_result


# ── Report generation on session finish ─────────────────────────

def _resolve_csv_path(config) -> Path | None:
    csv_opt = config.getoption("--bench-csv")
    if csv_opt is not None and csv_opt == "":
        return None  # explicitly disabled
    if csv_opt is not None:
        return Path(csv_opt)
    # Use _DEFAULTS['csv']; if None use auto path
    default = _DEFAULTS.get("csv")
    if default is not None and default == "":
        return None
    results_dir = ROOT / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"benchmark_{ts}.csv"


def pytest_sessionfinish(session):
    from benchmark.utils.report import generate_console_table, generate_csv

    # Collector is stored on config by the bench_collector fixture
    collector = getattr(session.config, "_bench_collector", None)
    if collector is None or len(collector) == 0:
        return

    results = collector.get_results()
    suite = _DEFAULTS["suite"]
    warmup = _DEFAULTS["warmup"]
    iters = _DEFAULTS["iters"]

    # Console table
    table = generate_console_table(results, suite, warmup, iters)
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write("\n")
        terminal.write(table)

    # CSV
    csv_path = _resolve_csv_path(session.config)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(generate_csv(results))
        if terminal is not None:
            terminal.write(f"CSV report saved to: {csv_path}\n")


def pytest_collection_modifyitems(config, items):
    """Apply @pytest.mark.bench to all items in benchmark/ or tests/ dirs
    that use run_benchmark or invoke_cli for bench testing."""
    bench_dirs = ("benchmark", "tests")
    for item in items:
        path = str(item.fspath)
        if any(f"/{d}/" in path or path.startswith(f"{d}/") for d in bench_dirs):
            if "bench" not in item.keywords:
                item.add_marker(pytest.mark.bench)
```

- [ ] **Step 2: Verify the plugin imports without error**

Run: `python -c "import benchmark.utils.pytest_plugin; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark/utils/pytest_plugin.py
git commit -m "feat: add shared pytest plugin for benchmark CLI, collector, and reporting

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Rewrite `benchmark/conftest.py`

**Files:**
- Modify: `benchmark/conftest.py`

- [ ] **Step 1: Rewrite benchmark/conftest.py**

```python
"""Benchmark conftest: loads the shared pytest plugin with benchmark defaults.

Ensure bench tests run serially: use -p no:xdist.
"""

from __future__ import annotations

# ── Set benchmark defaults before loading the shared plugin ─────
from benchmark.utils import pytest_plugin as _plugin

_plugin._DEFAULTS.update({
    "warmup": 10,
    "iters": 100,
    "csv": None,   # auto-generate path under benchmark/results/
    "suite": "typical",
})

pytest_plugins = ["benchmark.utils.pytest_plugin"]
```

- [ ] **Step 2: Verify conftest imports**

Run: `python -c "import benchmark.conftest; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark/conftest.py
git commit -m "refactor: rewrite benchmark/conftest to load shared pytest plugin

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: Create `benchmark/test_bench.py`

**Files:**
- Create: `benchmark/test_bench.py`

- [ ] **Step 1: Create benchmark/test_bench.py**

```python
"""Real benchmark cases covering ctest parameter ranges.

Run serially:  pytest benchmark/test_bench.py -p no:xdist --flagfft-cli ./build/flagfft-cli
Suite selection: --bench-suite=smoke|typical|full (default: typical)
"""

from __future__ import annotations

import pytest

from benchmark.utils.suites import API_DIRECTIONS, get_suite


def pytest_generate_tests(metafunc):
    """Parametrize at collection time, filtered by --bench-suite."""
    suite_name = metafunc.config.getoption("--bench-suite", default="typical")
    suite = get_suite(suite_name)

    # Build the list of (size, batch, api, direction) for this suite
    from benchmark.utils.suites import expand_params
    params = list(expand_params(suite))

    ids = [
        f"{api}-{direction}-{size}-b{batch}"
        for size, batch, api, direction in params
    ]
    metafunc.parametrize(
        "size,batch,api,direction",
        params,
        ids=ids,
    )


@pytest.mark.bench
def test_bench(size, batch, api, direction, run_benchmark, record_result):
    """Run a single benchmark configuration, record the result."""
    result, report = run_benchmark(size, api, direction, batch)

    # Enrich the case record with metadata for reporting
    case = report["cases"][0]
    case["size"] = size
    case["batch"] = batch
    record_result(case)

    if result.returncode != 0:
        pytest.fail(
            f"Benchmark process failed for {api}/{direction} size={size} batch={batch}"
        )
```

- [ ] **Step 2: Verify collection works (no GPU needed)**

Run:
```bash
pytest benchmark/test_bench.py --collect-only -q 2>&1 | tail -5
```
Expected: lists collected test IDs (will skip if CLI not built, that's fine)

- [ ] **Step 3: Commit**

```bash
git add benchmark/test_bench.py
git commit -m "feat: add benchmark/test_bench.py with pytest_generate_tests suite filtering

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: Rewrite `tests/conftest.py`

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Check tests/cli/test_flagfft_cli.py for fixtures it uses**

Run: `grep -n "invoke_cli\|flagfft_cli" tests/cli/test_flagfft_cli.py`

- [ ] **Step 2: Rewrite tests/conftest.py to load the plugin with test defaults**

```python
"""Test conftest: loads the shared pytest plugin with test-verification defaults.

Tests warm=1 iter=1 for quick CLI verification. No CSV output by default.
"""

from __future__ import annotations

# ── Set test-verification defaults before loading the shared plugin ──
from benchmark.utils import pytest_plugin as _plugin

_plugin._DEFAULTS.update({
    "warmup": 1,
    "iters": 1,
    "csv": "",   # disabled by default for verification tests
    "suite": "full",
})

pytest_plugins = ["benchmark.utils.pytest_plugin"]
```

Wait — the existing `tests/cli/test_flagfft_cli.py` uses `invoke_cli` from the old `tests/conftest.py`. The new conftest loads the shared plugin which provides the same `invoke_cli` fixture with identical behavior. So existing tests should continue to work.

- [ ] **Step 3: Verify existing CLI tests still collect**

Run:
```bash
pytest tests/cli/test_flagfft_cli.py --collect-only -q 2>&1 | tail -10
```
Expected: tests collected (may skip if CLI not built)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "refactor: rewrite tests/conftest to load shared pytest plugin with warm=1,iter=1 defaults

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 10: Create `tests/test_bench_cli.py`

**Files:**
- Create: `tests/test_bench_cli.py`

- [ ] **Step 1: Create tests/test_bench_cli.py**

```python
"""Quick verification of the benchmark CLI tool.

Runs ALL size/batch/API/direction combinations with warm=1 iter=1.
Full assertions on exit code, correctness, and timing fields.
Generates console report + optional CSV on session finish.
"""

from __future__ import annotations

import pytest

from benchmark.utils.suites import ALL_BATCHES, ALL_SIZES, API_DIRECTIONS

ALL_APIS = list(API_DIRECTIONS.keys())


def pytest_generate_tests(metafunc):
    """Parametrize over the full ctest grid (no suite filtering)."""
    params = []
    for size in ALL_SIZES:
        for batch in ALL_BATCHES:
            for api in ALL_APIS:
                for direction in API_DIRECTIONS[api]:
                    params.append((size, batch, api, direction))

    ids = [
        f"{api}-{direction}-{size}-b{batch}"
        for size, batch, api, direction in params
    ]
    metafunc.parametrize("size,batch,api,direction", params, ids=ids)


@pytest.mark.bench
def test_bench_cli(size, batch, api, direction, invoke_cli, record_result):
    """Verify the bench subcommand works correctly (warm=1, iter=1)."""
    result, report = invoke_cli(
        "bench",
        "--api", api,
        "--direction", direction,
        "--shape", str(size),
        "--batch", str(batch),
        "--warmup", "1",
        "--iters", "1",
        "--print-path",
    )

    # Enrich and record for the session-finish report
    case = report["cases"][0]
    case["size"] = size
    case["batch"] = batch
    record_result(case)

    # Assertions: the CLI must work and produce valid output
    assert result.returncode == 0, (
        f"Bench CLI failed for {api}/{direction} size={size} batch={batch}"
    )
    assert case["correctness"]["passed"], (
        f"Correctness failed for {api}/{direction} size={size} batch={batch}: "
        f"max_abs={case['correctness'].get('max_abs', 'N/A')}, "
        f"rms={case['correctness'].get('rms', 'N/A')}"
    )
    assert "speedup" in case["timing"], (
        f"Missing speedup in timing for {api}/{direction} size={size} batch={batch}"
    )
    assert case["timing"]["flagfft_median_ms"] > 0, (
        f"Median time should be positive for {api}/{direction} size={size} batch={batch}"
    )
```

- [ ] **Step 2: Verify collection count**

Run:
```bash
pytest tests/test_bench_cli.py --collect-only -q 2>&1 | tail -3
```
Expected: 312 tests collected (will skip if CLI not built)

- [ ] **Step 3: Commit**

```bash
git add tests/test_bench_cli.py
git commit -m "feat: add tests/test_bench_cli.py — warm=1 iter=1 full verification of bench CLI

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 11: Delete old test files

**Files:**
- Delete: `benchmark/test_bench_smoke.py`
- Delete: `benchmark/test_bench_full.py`

- [ ] **Step 1: Delete the old files**

```bash
git rm benchmark/test_bench_smoke.py benchmark/test_bench_full.py
```

- [ ] **Step 2: Verify no remaining imports reference deleted files**

Run:
```bash
grep -r "test_bench_smoke\|test_bench_full\|benchmark.suites\|benchmark.report" benchmark/ tests/ --include="*.py" 2>/dev/null || echo "No stale imports found"
```
Expected: no matches (or only in __pycache__)

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove old test_bench_smoke.py and test_bench_full.py

Replaced by benchmark/test_bench.py and tests/test_bench_cli.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 12: Build verify and smoke test

**Files:**
- None (verification only)

- [ ] **Step 1: Full collection check for both new test files**

```bash
pytest benchmark/test_bench.py tests/test_bench_cli.py --collect-only -q 2>&1 | tail -10
```
Expected: both files' tests collected (may skip if CLI not built)

- [ ] **Step 2: Verify existing CLI tests still work**

```bash
pytest tests/cli/test_flagfft_cli.py --collect-only -q 2>&1
```
Expected: existing tests collected normally

- [ ] **Step 3: Run a quick unit check on the report functions**

```bash
python -c "
from benchmark.utils.suites import SMOKE, TYPICAL, FULL, expand_params, get_suite
from benchmark.utils.report import generate_csv, generate_console_table, generate_markdown, generate_json_report
from benchmark.utils.collector import BenchmarkCollector

# Verify suite counts
assert len(list(expand_params(SMOKE))) == 24, f'Smoke: {len(list(expand_params(SMOKE)))}'
assert len(list(expand_params(TYPICAL))) == 136, f'Typical: {len(list(expand_params(TYPICAL)))}'
assert len(list(expand_params(FULL))) == 312, f'Full: {len(list(expand_params(FULL)))}'

# Verify collector
c = BenchmarkCollector()
c.add_result({'test': 1})
assert len(c) == 1
assert c.get_results() == [{'test': 1}]

# Verify CSV
records = [{'size': 16, 'api': 'c2c', 'direction': 'forward', 'batch': 1, 'backend': 'cuda', 'correctness': {'passed': True, 'max_abs': '1e-6', 'rms': '1e-7'}, 'timing': {'flagfft_median_ms': 0.1, 'cufft_median_ms': 0.2, 'speedup': 2.0}, 'warmup': 10, 'iters': 100}]
csv = generate_csv(records)
assert 'size,api,direction,batch,backend' in csv
assert '16,c2c,forward,1,cuda' in csv

# Verify console table
table = generate_console_table(records, 'smoke', 10, 100)
assert 'FlagFFT Benchmark Report' in table

print('All checks passed')
"
```
Expected: `All checks passed`

- [ ] **Step 5: Commit (if any fixups needed)**

Skip if clean.
```

---

## Post-Implementation Verification

```bash
# 1. Verify benchmark collection (smoke suite)
pytest benchmark/test_bench.py --collect-only -q --bench-suite=smoke

# 2. Verify benchmark collection (full suite)
pytest benchmark/test_bench.py --collect-only -q --bench-suite=full

# 3. Verify CLI verification test collection
pytest tests/test_bench_cli.py --collect-only -q

# 4. Run a real smoke benchmark (requires GPU + built CLI)
pytest benchmark/test_bench.py -v --flagfft-cli ./build/flagfft-cli \
    --bench-suite=smoke -p no:xdist

# 5. Run CLI verification (requires GPU + built CLI)
pytest tests/test_bench_cli.py -v --flagfft-cli ./build/flagfft-cli -p no:xdist
```
