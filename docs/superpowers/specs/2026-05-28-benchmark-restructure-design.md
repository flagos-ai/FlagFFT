# Benchmark Restructure Design

## Overview

Split the current `benchmark/` directory into two concerns:

1. **`benchmark/`** — benchmark tool library + pytest test cases that run real performance benchmarks
2. **`tests/`** — pytest test that verifies the benchmark CLI tool works correctly (warm=1, iter=1)

Both produce console + CSV reports via session-scoped result collection and `pytest_sessionfinish` hooks.

## Directory Structure

```
benchmark/
├── utils/
│   ├── __init__.py
│   ├── suites.py            # ctest coverage: sizes, batches, APIs, directions
│   ├── report.py            # Markdown table, CSV, JSON report generation
│   ├── collector.py         # Thread-safe session-scoped result collector
│   └── pytest_plugin.py     # Shared pytest plugin (options, fixtures, sessionfinish)
├── conftest.py              # loads pytest_plugin, benchmark-specific overrides
└── test_bench.py            # Real benchmark cases (parametrized via pytest_generate_tests)

tests/
├── conftest.py              # loads pytest_plugin, test-specific defaults
└── test_bench_cli.py        # Quick verification test (warm=1, iter=1, all combinations)
```

Files to remove:
- `benchmark/test_bench_smoke.py` (absorbed)
- `benchmark/test_bench_full.py` (absorbed)

## Shared Pytest Plugin (`benchmark/utils/pytest_plugin.py`)

Both `benchmark/` and `tests/` load this plugin to avoid duplication:

```python
# benchmark/conftest.py
pytest_plugins = ["benchmark.utils.pytest_plugin"]

# tests/conftest.py
pytest_plugins = ["benchmark.utils.pytest_plugin"]
```

The plugin provides:

### pytest options
| Option | Default | Description |
|--------|---------|-------------|
| `--flagfft-cli` | `FLAGFFT_CLI_EXE` env or `build/flagfft-cli` | Path to CLI binary |
| `--bench-warmup` | `10` | Warmup iterations |
| `--bench-iters` | `100` | Measurement iterations |
| `--bench-csv` | `None` (auto-path) | CSV output path; `""` disables CSV |
| `--bench-suite` | `typical` | Suite: `smoke`, `typical`, `full` |

### Fixtures
- `flagfft_cli` (session) — resolves CLI binary path
- `bench_warmup` / `bench_iters` / `bench_suite` / `bench_csv` (session) — option values
- `invoke_cli` — wraps `subprocess.run` + JSON parse with skip/error handling
- `run_benchmark` — calls `invoke_cli bench --warmup ... --iters ...`
- `bench_collector` (session) — `BenchmarkCollector` instance (NOT a process-level singleton; a plain session-scoped fixture)
- `record_result` — convenience fixture that calls `bench_collector.add_result()`

### Report generation (`pytest_sessionfinish`)
1. Read all results from `bench_collector` (accessed via the session's fixture store)
2. Print console table via `report.generate_console_table()`
3. Write CSV to the resolved path (auto or user-specified)
4. Print CSV path to terminal

`tests/conftest.py` overrides the `--bench-csv` default to `""` (no CSV by default for verification tests) and sets `--bench-warmup=1 --bench-iters=1`.

## Serial Execution and xdist

Benchmark tests require serial execution for accurate timing. Strategy:

- **`pyproject.toml` registers `bench` marker:**
  ```toml
  [tool.pytest.ini_options]
  markers = ["bench: performance benchmark tests (require serial execution)"]
  ```
- **Plugin sets `@pytest.mark.bench` on all benchmark test functions** (via `pytest_collection_modifyitems` if not explicitly set).
- **Plugin's `pytest_configure` detects xdist and refuses:** if `xdist` is in `config.pluginmanager.list_plugin_distinfo()`, emit a warning and force `-p no:xdist` behavior, OR skip benchmark tests entirely with a message. Simpler: document that benchmarks must run with `-p no:xdist`.
- **`conftest.py` enforces at import time:** benchmark tests check `PYTEST_XDIST_WORKER` env var and skip with explanation if set.

Rationale: xdist worker processes each get their own `bench_collector` fixture instance, so `pytest_sessionfinish` on the controller sees an empty collector. Forcing `-p no:xdist` is the simplest correct solution.

## Suite Filtering at Collection Time

Use `pytest_generate_tests` to filter parametrize values **before** collection, avoiding runtime skip noise and empty test reports:

```python
# benchmark/test_bench.py
def pytest_generate_tests(metafunc):
    suite_name = metafunc.config.getoption("--bench-suite", default="typical")
    suite = get_suite(suite_name)  # SMOKE, TYPICAL, or FULL

    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", suite["sizes"])
    if "batch" in metafunc.fixturenames:
        metafunc.parametrize("batch", suite["batches"])
    # api + direction: expand from API_DIRECTIONS, then optionally filter extras
```

Test function signature uses plain fixtures, no `@pytest.mark.parametrize` decorators — all parametrization is dynamic via `pytest_generate_tests`.

For `TYPICAL` extra combinations `[(256,4), (256,256), (2048,4), (2048,256)]`, `pytest_generate_tests` merges them into the parametrize lists or generates a combined `(size, batch)` id list.

## Suites (`benchmark/utils/suites.py`)

```python
ALL_SIZES   = [16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384]
ALL_BATCHES = [1, 4, 256]
ALL_APIS    = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]

API_DIRECTIONS = {
    "c2c": ["forward", "inverse"],
    "z2z": ["forward", "inverse"],
    "r2c": ["forward"],
    "d2z": ["forward"],
    "c2r": ["inverse"],
    "z2d": ["inverse"],
}

SMOKE   = {"sizes": [16, 256, 2048], "batches": [1]}                              # 24 cases
TYPICAL = {"sizes": ALL_SIZES, "batches": [1],
           "extra": [(256,4), (256,256), (2048,4), (2048,256)]}                   # 136 cases
FULL    = {"sizes": ALL_SIZES, "batches": ALL_BATCHES}                            # 312 cases
```

Suite selected via `--bench-suite={smoke|typical|full}`, default `typical`.

## Result Collector (`benchmark/utils/collector.py`)

Plain class (NOT a process-level singleton). Held as a session-scoped fixture:

```python
class BenchmarkCollector:
    def __init__(self):
        self._results: list[dict] = []

    def add_result(self, case: dict) -> None:
        self._results.append(case)

    def get_results(self) -> list[dict]:
        return list(self._results)
```

No global state — the session fixture ensures single instance per pytest session. Thread-safe because benchmarks run serially (`-p no:xdist`).

## Report Output (`benchmark/utils/report.py`)

Retain existing `generate_markdown()` and `generate_json_report()`. Add:

- `generate_csv(records)` — CSV string. Columns: `size,api,direction,batch,backend,flagfft_median_ms,ref_median_ms,speedup,correctness_passed,max_abs,rms,warmup,iters`
- `generate_console_table(records, suite, warmup, iters)` — Markdown-formatted table + summary stats (geometric mean, optional pass rate, speedup)

The `backend` column captures the active GPU backend name from the CLI JSON report.

### CSV path logic
Default: `benchmark/results/benchmark_YYYYMMDD_HHMMSS.csv`
- Auto-create `results/` directory
- `--bench-csv <path>` overrides
- `--bench-csv ""` disables CSV output
- Session end prints: `CSV report saved to: benchmark/results/benchmark_20260528_143052.csv`

## Test Files

### `benchmark/test_bench.py`
- `@pytest.mark.bench` — marks for serial requirement
- Uses `pytest_generate_tests` for suite-filtered parametrization
- Calls `run_benchmark()` with configured warmup/iters
- `record_result()` appends to collector
- `pytest.fail` on process error; no correctness/timing assert so failures don't block remaining cases

### `tests/test_bench_cli.py`
- `@pytest.mark.bench` — marks for serial requirement
- Runs ALL 312 combinations with warm=1, iter=1 (hardcoded, ignores `--bench-warmup/--bench-iters`)
- `pytest_generate_tests` parametrizes over full `ALL_SIZES × ALL_BATCHES × expand_api_directions(ALL_APIS)`
- Full assertions: exit code 0 and timing fields present; bench-only JSON does not include `correctness`
- Default `--bench-csv ""` (no CSV); user can opt in with `--bench-csv path.csv`

## pyproject.toml Changes

```toml
[tool.pytest.ini_options]
markers = ["bench: performance benchmark tests (require serial execution, -p no:xdist)"]
```

## Usage

```bash
# Quick verification (tests/): warm=1 iter=1, all 312 cases, full assertions, no CSV by default
pytest tests/test_bench_cli.py -v --flagfft-cli ./build/flagfft-cli

# Real benchmark (benchmark/): smoke suite
pytest benchmark/test_bench.py -v --flagfft-cli ./build/flagfft-cli \
    --bench-suite=smoke -p no:xdist

# Full benchmark with custom CSV output
pytest benchmark/test_bench.py -v --flagfft-cli ./build/flagfft-cli \
    --bench-suite=full --bench-csv results/full_bench.csv -p no:xdist
```

## Migration

1. Register `bench` marker in `pyproject.toml`
2. Create `benchmark/utils/` package with `suites.py`, `report.py`, `collector.py`, `pytest_plugin.py`
3. Rewrite `benchmark/conftest.py` — loads plugin, sets benchmark-specific defaults
4. Create `benchmark/test_bench.py` — `pytest_generate_tests` + `@pytest.mark.bench`
5. Create `tests/conftest.py` — loads plugin, overrides defaults (csv="", warmup=1, iters=1)
6. Create `tests/test_bench_cli.py` — `pytest_generate_tests` + assertions
7. Delete `benchmark/test_bench_smoke.py` and `benchmark/test_bench_full.py`
