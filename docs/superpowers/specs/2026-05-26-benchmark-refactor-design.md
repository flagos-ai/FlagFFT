# Benchmark Refactor Design

## Overview

基于项目现有的 `adaptor`（lib adapter）和 `test_adaptor` 两层后端抽象模式，重构 C++ benchmark 代码和 Python benchmark 测试基础设施。

## Architecture: Two-Layer Adapter

### Layer 1: Lib Adapter (`src/adaptor/adaptor.h`)

FlagFFT 运行时自身的后端抽象。已存在，本次不动。

- `Memory` — 设备内存 RAII
- `Stream` — CUDA stream RAII
- `EventTimer` — GPU event-based 计时器
- Device 管理函数 (`ensure_device`, `device_count`, `synchronize`, ...)

### Layer 2: Test Adapter (`src/adaptor/test_adaptor.h`) — 新增

参考 FFT 实现的抽象接口，服务于 ctest 正确性验证和 benchmark 性能对比。多后端支持：CUDA (cuFFT) → 未来 rocFFT。

接口：

```
namespace flagfft::test_adaptor {

// --- Reference plan RAII ---
class RefPlanHandle;           // 参考 FFT plan 生命周期管理

// --- Plan creation ---
void ref_plan_1d(RefPlanHandle&, int nx, flagfftType, int batch);
void ref_plan_2d(RefPlanHandle&, int nx, int ny, flagfftType);
void ref_plan_3d(RefPlanHandle&, int nx, int ny, int nz, flagfftType);

// --- Plan execution ---
void ref_exec_c2c(RefPlanHandle&, flagfftComplex* in, flagfftComplex* out, int dir);
void ref_exec_z2z(RefPlanHandle&, flagfftDoubleComplex* in, flagfftDoubleComplex* out, int dir);
void ref_exec_r2c(RefPlanHandle&, flagfftReal* in, flagfftComplex* out);
void ref_exec_d2z(RefPlanHandle&, flagfftDoubleReal* in, flagfftDoubleComplex* out);
void ref_exec_c2r(RefPlanHandle&, flagfftComplex* in, flagfftReal* out);
void ref_exec_z2d(RefPlanHandle&, flagfftDoubleComplex* in, flagfftDoubleReal* out);

// --- Data generation ---
std::vector<float> random_complex_data(int64_t n, int64_t batch);
std::vector<double> random_double_complex_data(int64_t n, int64_t batch);

// --- Correctness comparison ---
struct ErrorMetric { double max_abs; double rms; };
ErrorMetric compute_error(const float* a, const float* b, int64_t n);
ErrorMetric compute_error(const double* a, const double* b, int64_t n);

// --- Backend metadata ---
std::string backend_name();

}
```

### Directory Structure

```
src/adaptor/
  adaptor.h                  ← lib adapter 接口 (已有)
  test_adaptor.h             ← test adapter 接口 (新增: 从 ctest/flagfft_test.h 提取)
  backend/
    cuda/
      adaptor.cpp            ← CUDA lib 实现 (已有)
      test_adaptor.cpp       ← CUDA test 实现 (新增: 从 ctest/backend/cuda/adaptor.cpp 迁移)
    rocm/                     ← 未来: ROCm backend，加两个文件即可
      adaptor.cpp
      test_adaptor.cpp
```

## C++ Benchmark 改造

### 现有代码清理

`src/cli_tools/tune/bench.cpp` 中直接调用 cuFFT 的代码改为通过 `test_adaptor` 接口：

- `verify_against_cufft()` → 使用 `test_adaptor::ref_plan_1d` + `ref_exec_c2c`
- `bench_candidate()` 中的 FlagFFT 调用保持不变（它测的就是 FlagFFT）
- `generate_random_input()` → 使用 `test_adaptor::random_complex_data`

### CMake 改造

- `test_adaptor.cpp` 作为 `flagfft` 或独立 OBJECT library 加入编译
- ctest 链接 `test_adaptor` 目标

## ctest 改造

- 删除 `ctest/flagfft_test.h` 中已迁移的宏和函数
- 删除 `ctest/backend/` 目录
- 改为 `#include "adaptor/test_adaptor.h"`
- 保留仅 ctest 内部使用的 Google Test 特定便利函数（如 `FAIL()` 包装）

## Python Benchmark 基础设施

### Directory

```
benchmark/
  __init__.py
  conftest.py               ← bench CLI fixture + invoke wrapper
  suites.py                  ← 小/全量测试集规模定义
  test_bench_smoke.py        ← @pytest.mark.smoke 快速验证
  test_bench_full.py         ← @pytest.mark.full 全量 benchmark
  report.py                  ← JSON→Markdown 报告生成
```

### conftest.py

参照 `tests/conftest.py` 模式：

```python
def pytest_addoption(parser):
    parser.addoption("--flagfft-cli", ...)
    parser.addoption("--bench-warmup", ...)
    parser.addoption("--bench-iters", ...)
    parser.addoption("--bench-report", ...)

@pytest.fixture(scope="session")
def flagfft_cli(request) -> Path: ...

@pytest.fixture
def invoke_bench(flagfft_cli):
    # 封装 subprocess 调用，解析 JSON，skip 处理
```

### suites.py

```python
@dataclass
class BenchCase:
    size: int
    factorization: str
    codepath: str
    rationale: str

SMOKE_SUITE = [BenchCase(16, "2^4", "DirectDFT", ...), ...]
FULL_SUITE  = [全部 13 个]
```

### 测试文件

- `test_bench_smoke.py` — `@pytest.mark.smoke`，API × 3 个 smoke 规模
- `test_bench_full.py` — `@pytest.mark.full`，API × 全量 13 规模

### report.py

- 输入：`invoke_bench` 返回的 JSON report
- 输出：Markdown 表格，包含 `size | codepath | flagfft_ms | cufft_ms | speedup | error`

## Build & CMake

```
# 将 test_adaptor 编译为 OBJECT library，供 flagfft-cli 和 ctest 共用
add_library(flagfft_test_adaptor OBJECT
    src/adaptor/backend/cuda/test_adaptor.cpp
)
target_link_libraries(flagfft-cli PRIVATE flagfft_test_adaptor)
target_link_libraries(ctest PRIVATE flagfft_test_adaptor)
```

## Test Plan

1. C++ ctest 全量通过（验证 test_adaptor 迁移正确）
2. `pytest benchmark/test_bench_smoke.py -m smoke` 通过
3. `pytest benchmark/test_bench_full.py -m full` 通过
4. Markdown 报告正确生成
5. `--bench-report markdown` CLI 输出格式正确
