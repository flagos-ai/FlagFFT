# CTest-Pytest 集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ctest 测试可执行文件对接到 pytest 中，每个可执行文件映射为一个 pytest 用例

**Architecture:** 在 `tests/ctest/` 下新增 conftest.py 提供 `--ctest-build-dir` 选项和 fixtures，test_correctness.py 使用 parametrize 对 9 个 ctest 目标生成用例，通过 subprocess 运行可执行文件检查返回码

**Tech Stack:** pytest, subprocess

---

### Task 1: 创建目录和包标记

**Files:**
- Create: `tests/ctest/__init__.py`

- [ ] **Step 1: 创建空文件**

```bash
mkdir -p tests/ctest
touch tests/ctest/__init__.py
```

- [ ] **Step 2: 提交**

```bash
git add tests/ctest/__init__.py
git commit -m "chore: add tests/ctest package"
```

---

### Task 2: 创建 conftest.py

**Files:**
- Create: `tests/ctest/conftest.py`

- [ ] **Step 1: 写入 conftest.py**

```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

def pytest_addoption(parser):
    parser.addoption(
        "--ctest-build-dir",
        default=None,
        help="Path to ctest build directory. Defaults to build/ctest.",
    )


@pytest.fixture(scope="session")
def ctest_build_dir(request) -> Path:
    configured = request.config.getoption("--ctest-build-dir")
    path = Path(configured or ROOT / "build" / "ctest")
    if not path.is_dir():
        pytest.skip(f"ctest build directory not found: {path}")
    return path


@pytest.fixture
def run_ctest(ctest_build_dir: Path):
    def run(target: str, *, timeout: int = 300) -> subprocess.CompletedProcess:
        exe = ctest_build_dir / target
        if not exe.is_file():
            pytest.skip(f"ctest executable not found: {exe}")
        env = os.environ.copy()
        result = subprocess.run(
            [str(exe)],
            cwd=ctest_build_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result

    return run
```

- [ ] **Step 2: 提交**

```bash
git add tests/ctest/conftest.py
git commit -m "feat: add ctest conftest with --ctest-build-dir option and fixtures"
```

---

### Task 3: 创建测试文件

**Files:**
- Create: `tests/ctest/test_correctness.py`

- [ ] **Step 1: 写入 test_correctness.py**

```python
from __future__ import annotations

import pytest

CTEST_TARGETS = [
    "test_plan",
    "test_exec_c2c",
    "test_exec_z2z",
    "test_exec_r2c",
    "test_exec_c2r",
    "test_exec_d2z",
    "test_exec_z2d",
    "test_exec_r2c_c2r",
    "test_exec_d2z_z2d",
]


@pytest.mark.parametrize("target", CTEST_TARGETS)
def test_ctest_correctness(target: str, run_ctest) -> None:
    result = run_ctest(target)
    assert result.returncode == 0, (
        f"ctest {target} failed (exit {result.returncode})\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
```

- [ ] **Step 2: 提交**

```bash
git add tests/ctest/test_correctness.py
git commit -m "feat: add ctest correctness tests parametrized by target"
```

---

### Task 4: 构建并验证

**前置:** 需要已构建的 `build/ctest/` 目录

- [ ] **Step 1: 确认构建产物存在**

```bash
ls build/ctest/test_plan build/ctest/test_exec_c2c
```
预期：列出两个可执行文件

- [ ] **Step 2: 运行 ctest pytest（完整）**

```bash
python -m pytest tests/ctest/ -v
```
预期：9 passed

- [ ] **Step 3: 运行 ctest pytest（smoke）**

```bash
FLAGFFT_TEST_PROFILE=smoke python -m pytest tests/ctest/ -v
```
预期：9 passed（smoke profile 仍通过所有 9 个可执行文件，profile 过滤在 C++ 侧生效）

- [ ] **Step 4: 验证不存在的构建目录被 skip**

```bash
python -m pytest tests/ctest/ -v --ctest-build-dir /nonexistent
```
预期：9 skipped，提示 "ctest build directory not found"

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "test: verify ctest-pytest integration"
```
