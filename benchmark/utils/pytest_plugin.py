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
    "csv": None,  # None = auto-generate path; "" = disabled
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
        "Default: benchmark/results/benchmark_<timestamp>.csv",
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
            config.option.markexpr = _and_markexpr(config.option.markexpr, "not bench")
            sys.stderr.write(
                "\n⚠ xdist detected — bench tests disabled. "
                "Run with '-p no:xdist' for benchmarks.\n\n"
            )


def _and_markexpr(current: str, condition: str) -> str:
    if not current:
        return condition
    if condition in current:
        return current
    return f"{current} and {condition}"


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def flagfft_cli(request) -> Path:
    configured = request.config.getoption("flagfft_cli")
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
            "--api",
            api,
            "--direction",
            direction,
            "--shape",
            str(size),
            "--batch",
            str(batch),
            "--warmup",
            str(bench_warmup),
            "--iters",
            str(bench_iters),
            "--print-path",
        )

    return _run


@pytest.fixture(scope="session")
def bench_warmup(request) -> int:
    val = request.config.getoption("bench_warmup")
    if val is not None:
        return val
    return _DEFAULTS["warmup"]


@pytest.fixture(scope="session")
def bench_iters(request) -> int:
    val = request.config.getoption("bench_iters")
    if val is not None:
        return val
    return _DEFAULTS["iters"]


@pytest.fixture(scope="session")
def bench_suite(request) -> str:
    val = request.config.getoption("bench_suite")
    if val is not None:
        return val
    return _DEFAULTS["suite"]


@pytest.fixture(scope="session")
def bench_csv(request) -> str | None:
    """Resolved CSV path. None = auto, '' = disabled, otherwise user path."""
    val = request.config.getoption("bench_csv")
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
    csv_opt = config.getoption("bench_csv")
    if csv_opt is not None and csv_opt == "":
        return None  # explicitly disabled
    if csv_opt is not None:
        return Path(csv_opt)
    # Use _DEFAULTS['csv']; if explicit path use it, if "" disable, if None auto
    default = _DEFAULTS.get("csv")
    if default is not None and default != "":
        return Path(default)
    if default == "":
        return None
    results_dir = ROOT / "benchmark" / "results"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"benchmark_{ts}.csv"


def pytest_sessionfinish(session):
    from benchmark.utils.report import generate_console_table, generate_csv

    # Collector is stored on config by the bench_collector fixture
    collector = getattr(session.config, "_bench_collector", None)
    if collector is None or len(collector) == 0:
        return

    results = collector.get_results()

    # Resolve suite/warmup/iters from actual config options (not _DEFAULTS directly)
    suite = session.config.getoption("bench_suite") or _DEFAULTS["suite"]
    warmup = session.config.getoption("bench_warmup")
    if warmup is None:
        warmup = _DEFAULTS["warmup"]
    iters = session.config.getoption("bench_iters")
    if iters is None:
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
