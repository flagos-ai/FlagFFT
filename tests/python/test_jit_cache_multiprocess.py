"""Optional native regression: run simultaneous JIT users against one cache."""

import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import pytest


def test_native_jit_cache_shared_by_processes():
    executable = os.environ.get("FLAGFFT_TEST_CLI")
    if not executable:
        pytest.skip("set FLAGFFT_TEST_CLI to a built flagfft-cli for the native test")

    cli = Path(executable).resolve()
    requests = [("16", "forward")] * 4 + [("16", "inverse")] * 2

    def run(request):
        shape, direction = request
        completed = subprocess.run(
            [str(cli), "bench", "--rank", "1", "--shape", shape,
             "--api", "c2c", "--direction", direction,
             "--warmup", "1", "--iters", "1", "--json"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["status"] == "passed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        list(pool.map(run, requests))

    cache = cli.parent / ".flagfft" / "requests"
    modules = list(cache.rglob("*.py"))
    assert modules
    for source in modules:
        metadata = json.loads(source.with_suffix(".json").read_text())
        assert Path(metadata["module_path"]) == source
