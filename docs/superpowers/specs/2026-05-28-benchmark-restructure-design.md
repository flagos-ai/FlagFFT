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
│   └── collector.py         # Thread-safe session-scoped result collector singleton
├── conftest.py              # fixtures, pytest options, pytest_sessionfinish hook
└── test_bench.py            # Real benchmark cases (parametrized, full warmup/iters)

tests/
├── conftest.py              # fixtures (invoke_cli, record_result), pytest_sessionfinish
└── test_bench_cli.py        # Quick verification test (warm=1, iter=1, all combinations)
```

Files to remove:
- `benchmark/test_bench_smoke.py` (absorbed by `test_bench.py` and `tests/test_bench_cli.py`)
- `benchmark/test_bench_full.py` (absorbed by `test_bench.py` and `tests/test_bench_cli.py`)

## Suites (`benchmark/utils/suites.py`)

Three-tier hierarchy matching ctest parameter ranges:

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

# Smoke (24 cases): key sizes x batch=1
SMOKE   = {"sizes": [16, 256, 2048], "batches": [1]}

# Typical (136 cases): all sizes x batch=1 + extra multi-batch on key sizes
TYPICAL = {"sizes": ALL_SIZES, "batches": [1],
           "extra": [(256,4), (256,256), (2048,4), (2048,256)]}

# Full (312 cases): all sizes x all batches
FULL    = {"sizes": ALL_SIZES, "batches": ALL_BATCHES}
```

Suite selected via `--bench-suite={smoke|typical|full}`, default `typical`.

## Result Collector (`benchmark/utils/collector.py`)

Thread-safe singleton that accumulates benchmark results across the session:

```python
class BenchmarkCollector:
    def add_result(self, case: dict) -> None
    def get_results(self) -> list[dict]
    def clear(self) -> None
```

## Conftest (`benchmark/conftest.py`)

### pytest options
| Option | Default | Description |
|--------|---------|-------------|
| `--flagfft-cli` | `FLAGFFT_CLI_EXE` env or `build/flagfft-cli` | Path to CLI binary |
| `--bench-warmup` | `10` | Warmup iterations |
| `--bench-iters` | `100` | Measurement iterations |
| `--bench-csv` | `None` (auto-path) | CSV output path; `""` disables CSV |
| `--bench-suite` | `typical` | Suite: `smoke`, `typical`, `full` |

### Session fixtures
- `flagfft_cli` (session) — resolves CLI binary path
- `bench_warmup` / `bench_iters` / `bench_suite` / `bench_csv` (session) — option values
- `invoke_cli` — wraps `subprocess.run` + JSON parse with skip/error handling
- `run_benchmark` — calls `invoke_cli bench --warmup ... --iters ...`
- `bench_collector` (session) — `BenchmarkCollector` singleton
- `record_result` — convenience fixture that calls `bench_collector.add_result()`

### Serial execution
Benchmark tests are marked `@pytest.mark.serial`. `pytest_collection_modifyitems` ensures they run sequentially. If xdist is detected, these tests are pinned to a single worker.

### Report generation (`pytest_sessionfinish`)
1. Read all results from `bench_collector`
2. Print console table via `report.generate_console_table()`
3. Write CSV to the resolved path (auto or user-specified)
4. Print CSV path to terminal

## Report Output (`benchmark/utils/report.py`)

Retain existing `generate_markdown()` and `generate_json_report()`. Add:

- `generate_csv(records)` — CSV string with columns: `size,api,direction,batch,flagfft_median_ms,cufft_median_ms,speedup,correctness_passed,max_abs,rms,warmup,iters`
- `generate_console_table(records, suite, warmup, iters)` — Markdown-formatted table + summary stats (geometric mean, pass rate, speedup)

### CSV path logic
Default: `benchmark/results/benchmark_YYYYMMDD_HHMMSS.csv`
- Auto-create `results/` directory
- `--bench-csv <path>` overrides
- `--bench-csv ""` disables CSV output
- Session end prints: `CSV report saved to: benchmark/results/benchmark_20260528_143052.csv`

## Test Files

### `benchmark/test_bench.py`
- `@pytest.mark.serial` — force sequential execution
- Triple parametrize: `(size, batch, api/direction)` filtered by active suite
- Calls `run_benchmark()` with configured warmup/iters
- `record_result()` appends to collector
- `pytest.fail` on process error (no correctness/timing assert so failures don't block remaining cases)

### `tests/test_bench_cli.py`
- Runs ALL 312 combinations with warm=1, iter=1
- `@pytest.mark.parametrize` over all size/batch/api/direction combos
- Full assertions: exit code 0, correctness passed, timing fields present
- Also uses collector + `pytest_sessionfinish` for report output

### `tests/conftest.py`
- Provides `--flagfft-cli` option, `invoke_cli` fixture, `record_result` fixture
- `pytest_sessionfinish` generates console table + CSV
- Independent of `benchmark/conftest.py` to avoid circular imports

## Usage

```bash
# Quick verification (tests/): warm=1 iter=1, all 312 cases, full assertions
pytest tests/test_bench_cli.py -v --flagfft-cli ./build/flagfft-cli

# Real benchmark (benchmark/): smoke suite, default warmup=10 iters=100
pytest benchmark/test_bench.py -v --flagfft-cli ./build/flagfft-cli --bench-suite=smoke

# Full benchmark with custom CSV output
pytest benchmark/test_bench.py -v --flagfft-cli ./build/flagfft-cli \
    --bench-suite=full --bench-csv results/full_bench.csv
```

## Migration

1. Move `suites.py` and `report.py` into `benchmark/utils/`
2. Create `benchmark/utils/collector.py`
3. Rewrite `benchmark/conftest.py` with new options, fixtures, collector, sessionfinish
4. Create `benchmark/test_bench.py`
5. Create `tests/conftest.py` and `tests/test_bench_cli.py`
6. Delete `benchmark/test_bench_smoke.py` and `benchmark/test_bench_full.py`
