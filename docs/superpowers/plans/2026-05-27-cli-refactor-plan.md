# CLI 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除 CLI 的 test 职能，重构 bench 使所有平台相关代码通过 adaptor/test_adaptor 封装，tune 变为占位符。

**Architecture:** 新文件组织 —— `common/` 保留 CaseSpec/工具/RAII，`bench/` 新增 runner + report，`tune/` 改为占位符，`main.cpp` 重写参数解析和命令分发。bench runner 不包含任何 cuFFT 头文件，所有原生库操作通过 `test_adaptor` 接口。

**Tech Stack:** C++20, CMake + Ninja, CUDA 13.2, cuFFT (via test_adaptor), nlohmann_json, SQLite3

---

## 文件结构

| 文件 | 动作 | 职责 |
|------|------|------|
| `src/cli_tools/common/case.hpp` | 修改 | 删除 cufft_type, unsupported_reason, Operation 枚举, PlanApi 枚举, CaseSpec.stream；新增 int rank |
| `src/cli_tools/common/case.cpp` | 修改 | 删除对应实现，调整 parse_shape/case_json |
| `src/cli_tools/common/cli_utils.hpp` | 修改 | 删除 cufft_result_name, check_cufft, PlanApi::PlanMany, PlanApi 枚举；新增加 rank 解析 |
| `src/cli_tools/common/cli_utils.cpp` | 修改 | 删除对应实现，添加 parse_rank/rank_name |
| `src/cli_tools/common/plan_handles.hpp` | 修改 | 删除 CufftPlanHandle，移除 cufft.h include |
| `src/cli_tools/common/plan_handles.cpp` | 修改 | 删除 CufftPlanHandle 实现，移除 cufft.h include |
| `src/cli_tools/common/runner.hpp` | 删除 | 逻辑移至 bench/runner.hpp |
| `src/cli_tools/common/runner.cpp` | 删除 | 逻辑移至 bench/runner.cpp |
| `src/cli_tools/common/runtime_raii.hpp` | 不变 | DeviceMemory/Stream/Timer 别名 |
| `src/cli_tools/bench/runner.hpp` | 新建 | BenchResult, TimingStats, run_benchmark() 声明 |
| `src/cli_tools/bench/runner.cpp` | 新建 | bench 执行逻辑，仅依赖 adaptor + test_adaptor |
| `src/cli_tools/bench/report.hpp` | 新建 | format_table(), format_json() 声明 |
| `src/cli_tools/bench/report.cpp` | 新建 | 人类可读表格 + JSON 输出 |
| `src/cli_tools/tune/tune.hpp` | 新建 | tune_placeholder() 声明 |
| `src/cli_tools/tune/tune.cpp` | 新建 | 输出错误信息并 exit(1) |
| `src/cli_tools/tune/driver.hpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/tune/driver.cpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/tune/bench.hpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/tune/bench.cpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/tune/sqlite.hpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/tune/sqlite.cpp` | 删除 | 旧 tune 代码 |
| `src/cli_tools/flagfft-cli/main.cpp` | 重写 | 新参数解析、bench/tune 命令分发 |
| `CMakeLists.txt` | 修改 | 更新源文件列表和链接依赖 |
| `tests/cli/test_flagfft_cli.py` | 修改 | 适配新 CLI 接口 |

---

### Task 1: 清理 case.hpp / case.cpp

**Files:**
- Modify: `src/cli_tools/common/case.hpp`
- Modify: `src/cli_tools/common/case.cpp`

- [ ] **Step 1: 重写 case.hpp**

```cpp
#pragma once

#include <string>
#include <vector>

#include "cli_utils.hpp"

namespace flagfft::cli {

struct CaseSpec {
  FftApi api = FftApi::C2C;
  std::vector<int> shape{16};
  int batch = 1;
  int direction = FLAGFFT_FORWARD;
  Placement placement = Placement::OutOfPlace;
  int rank = 1;
};

std::vector<int> parse_shape(const std::string& value);
json case_json(const CaseSpec& spec);
flagfftType flagfft_type(FftApi api);
bool is_complex_api(FftApi api);
bool is_double_api(FftApi api);
bool is_real_forward_api(FftApi api);
bool is_real_inverse_api(FftApi api);

}  // namespace flagfft::cli
```

删除内容：
- `Operation` 枚举
- `PlanApi` 枚举引用
- `CaseSpec::stream` 字段
- `CaseSpec::plan_api` → 替换为 `int rank`
- `unsupported_reason()` 声明
- `cufftType cufft_type(FftApi)` 声明

- [ ] **Step 2: 重写 case.cpp**

```cpp
#include "case.hpp"

#include <sstream>

namespace flagfft::cli {

std::vector<int> parse_shape(const std::string& value) {
  std::vector<int> shape;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t end = value.find_first_of("xX", start);
    const std::string part =
        value.substr(start, end == std::string::npos ? end : end - start);
    if (part.empty()) {
      throw AssertionFailure("invalid --shape: " + value);
    }
    try {
      std::size_t consumed = 0;
      const int n = std::stoi(part, &consumed);
      if (consumed != part.size()) {
        throw AssertionFailure("invalid --shape: " + value);
      }
      if (n <= 0) {
        throw AssertionFailure("shape dimensions must be positive");
      }
      shape.push_back(n);
    } catch (const std::invalid_argument&) {
      throw AssertionFailure("invalid --shape: " + value);
    } catch (const std::out_of_range&) {
      throw AssertionFailure("invalid --shape: " + value);
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (shape.empty() || shape.size() > 3) {
    throw AssertionFailure("--shape rank must be between 1 and 3");
  }
  return shape;
}

json case_json(const CaseSpec& spec) {
  return {
      {"api", fft_api_name(spec.api)},
      {"shape", spec.shape},
      {"rank", spec.rank},
      {"batch", spec.batch},
      {"direction", direction_name(spec.direction)},
      {"placement", placement_name(spec.placement)},
  };
}

bool is_complex_api(FftApi api) {
  return api == FftApi::C2C || api == FftApi::Z2Z;
}

bool is_double_api(FftApi api) {
  return api == FftApi::Z2Z || api == FftApi::D2Z || api == FftApi::Z2D;
}

bool is_real_forward_api(FftApi api) {
  return api == FftApi::R2C || api == FftApi::D2Z;
}

bool is_real_inverse_api(FftApi api) {
  return api == FftApi::C2R || api == FftApi::Z2D;
}

flagfftType flagfft_type(FftApi api) {
  switch (api) {
    case FftApi::C2C: return FLAGFFT_C2C;
    case FftApi::Z2Z: return FLAGFFT_Z2Z;
    case FftApi::R2C: return FLAGFFT_R2C;
    case FftApi::D2Z: return FLAGFFT_D2Z;
    case FftApi::C2R: return FLAGFFT_C2R;
    case FftApi::Z2D: return FLAGFFT_Z2D;
  }
  return FLAGFFT_C2C;
}

}  // namespace flagfft::cli
```

删除内容：`unsupported_reason()`, `cufft_type()`

- [ ] **Step 3: 提交**

```bash
git add src/cli_tools/common/case.hpp src/cli_tools/common/case.cpp
git commit -m "refactor(cli): remove test-specific and CUDA-specific code from case"
```

---

### Task 2: 清理 cli_utils.hpp / cli_utils.cpp

**Files:**
- Modify: `src/cli_tools/common/cli_utils.hpp`
- Modify: `src/cli_tools/common/cli_utils.cpp`

- [ ] **Step 1: 重写 cli_utils.hpp**

```cpp
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "flagfft.h"

namespace flagfft::cli {

using json = nlohmann::json;

inline constexpr int kExitPassed = 0;
inline constexpr int kExitFailed = 1;
inline constexpr int kExitRuntimeError = 2;
inline constexpr int kExitSkipped = 77;

enum class FftApi { C2C, Z2Z, R2C, D2Z, C2R, Z2D };
enum class Placement { InPlace, OutOfPlace };

FftApi parse_fft_api(const std::string& value);
std::string fft_api_name(FftApi api);
Placement parse_placement(const std::string& value);
std::string placement_name(Placement p);

int parse_rank(const std::string& value);
std::string rank_name(int rank);

class CliException : public std::runtime_error {
 public:
  CliException(std::string message, int exit_code);
  int exit_code() const noexcept;

 private:
  int exit_code_;
};

class AssertionFailure : public CliException {
 public:
  explicit AssertionFailure(std::string message);
};

std::string flagfft_result_name(flagfftResult result);
std::string direction_name(int direction);
int parse_direction(const std::string& value);

bool has_cuda_device(std::string& reason);
std::string shell_quote(const std::string& value);
std::string executable_path(const char* argv0);
std::string executable_dir(const char* argv0);

void check_flagfft(flagfftResult result, const std::string& context);
void expect_flagfft(flagfftResult actual, flagfftResult expected,
                    const std::string& context);
void expect_true(bool condition, const std::string& context);

int exit_code_for_report(const json& report);
int emit_json_report(json report);

}  // namespace flagfft::cli
```

删除内容：
- `#include <cufft.h>`
- `PlanApi` 枚举
- `parse_plan_api()`, `plan_api_name()`
- `cufft_result_name()`
- `default_tune_db()`
- `check_cufft()`

新增内容：
- `parse_rank()`, `rank_name()`

- [ ] **Step 2: 修改 cli_utils.cpp**

先读取 `/workspace/FlagFFT-dev/.claude/worktrees/refactor-cli/src/cli_tools/common/cli_utils.cpp`

只做以下修改：

(1) 删除 `#include <cufft.h>`（第1行附近）

(2) 删除 `PlanApi` 相关函数 (`parse_plan_api`, `plan_api_name`)

(3) 删除 `cufft_result_name()` 函数

(4) 删除 `check_cufft()` 函数

(5) 删除 `default_tune_db()` 函数

(6) 添加两个新函数：

```cpp
int parse_rank(const std::string& value) {
  if (value == "1") return 1;
  if (value == "2") return 2;
  if (value == "3") return 3;
  throw AssertionFailure("invalid --rank: " + value + " (expected 1, 2, or 3)");
}

std::string rank_name(int rank) {
  return std::to_string(rank);
}
```

- [ ] **Step 3: 提交**

```bash
git add src/cli_tools/common/cli_utils.hpp src/cli_tools/common/cli_utils.cpp
git commit -m "refactor(cli): remove PlanApi and CUDA-specific utils, add rank parsing"
```

---

### Task 3: 清理 plan_handles.hpp / plan_handles.cpp

**Files:**
- Modify: `src/cli_tools/common/plan_handles.hpp`
- Modify: `src/cli_tools/common/plan_handles.cpp`

- [ ] **Step 1: 重写 plan_handles.hpp**

```cpp
#pragma once

#include "flagfft.h"

namespace flagfft::cli {

class FlagfftPlanHandle {
 public:
  FlagfftPlanHandle() = default;
  explicit FlagfftPlanHandle(flagfftHandle handle);
  ~FlagfftPlanHandle();

  FlagfftPlanHandle(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle& operator=(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle(FlagfftPlanHandle&& other) noexcept;
  FlagfftPlanHandle& operator=(FlagfftPlanHandle&& other) noexcept;

  flagfftHandle get() const noexcept;
  flagfftHandle release() noexcept;
  void reset(flagfftHandle handle = nullptr);

 private:
  flagfftHandle handle_ = nullptr;
};

}  // namespace flagfft::cli
```

删除整个 `CufftPlanHandle` 类和 `#include <cufft.h>`。

- [ ] **Step 2: 重写 plan_handles.cpp**

读取 `/workspace/FlagFFT-dev/.claude/worktrees/refactor-cli/src/cli_tools/common/plan_handles.cpp`，删除 `CufftPlanHandle` 实现（构造函数、析构函数、移动构造、移动赋值、get/put/reset）和 `#include <cufft.h>`。

- [ ] **Step 3: 提交**

```bash
git add src/cli_tools/common/plan_handles.hpp src/cli_tools/common/plan_handles.cpp
git commit -m "refactor(cli): remove CufftPlanHandle, keep only FlagfftPlanHandle"
```

---

### Task 4: 删除旧 runner.hpp / runner.cpp

**Files:**
- Delete: `src/cli_tools/common/runner.hpp`
- Delete: `src/cli_tools/common/runner.cpp`

- [ ] **Step 1: 删除文件**

```bash
git rm src/cli_tools/common/runner.hpp src/cli_tools/common/runner.cpp
```

- [ ] **Step 2: 提交**

```bash
git commit -m "refactor(cli): remove old runner (replaced by bench/runner)"
```

---

### Task 5: 创建 bench/runner.hpp / runner.cpp

**Files:**
- Create: `src/cli_tools/bench/runner.hpp`
- Create: `src/cli_tools/bench/runner.cpp`

- [ ] **Step 1: 创建 runner.hpp**

```cpp
#pragma once

#include <string>
#include <vector>

#include "cli_tools/common/case.hpp"

namespace flagfft::cli::bench {

struct TimingStats {
  double median_ms = 0.0;
  double p90_ms = 0.0;
  std::vector<float> samples;
};

struct BenchResult {
  TimingStats flagfft;
  TimingStats reference;
  double speedup = 0.0;
  std::string plan_description;
};

BenchResult run_benchmark(const CaseSpec& spec, int warmup, int iters,
                          bool include_path);

}  // namespace flagfft::cli::bench
```

- [ ] **Step 2: 创建 runner.cpp**

```cpp
#include "cli_tools/bench/runner.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "cli_tools/common/cli_utils.hpp"
#include "cli_tools/common/plan_handles.hpp"
#include "cli_tools/common/runtime_raii.hpp"
#include "flagfft.h"

namespace flagfft::cli::bench {
namespace {

double percentile(std::vector<float> times, double fraction) {
  std::sort(times.begin(), times.end());
  const std::size_t index = std::min(
      times.size() - 1,
      static_cast<std::size_t>(fraction * static_cast<double>(times.size())));
  return times[index];
}

struct BufferLayout {
  int n;
  int half;
  int padded;
  std::size_t scalar_bytes;
  std::size_t allocation_bytes;
};

BufferLayout layout_for(const CaseSpec& spec) {
  const int n = spec.shape[0];
  const int half = n / 2 + 1;
  const int padded = 2 * half;
  const std::size_t scalar =
      is_double_api(spec.api) ? sizeof(double) : sizeof(float);
  const bool real_forward = is_real_forward_api(spec.api);
  const bool real_inverse = is_real_inverse_api(spec.api);
  const bool complex = is_complex_api(spec.api);

  const int inplace_scalars = complex ? 2 * n : padded;

  const int in_scalars =
      real_forward ? n : (real_inverse ? 2 * half : 2 * n);
  const int out_scalars =
      real_forward ? 2 * half : (real_inverse ? n : 2 * n);

  const std::size_t eff_in =
      spec.placement == Placement::InPlace ? inplace_scalars : in_scalars;
  const std::size_t eff_out =
      spec.placement == Placement::InPlace ? inplace_scalars : out_scalars;

  const std::size_t in_bytes =
      static_cast<std::size_t>(spec.batch) * eff_in * scalar;
  const std::size_t out_bytes =
      static_cast<std::size_t>(spec.batch) * eff_out * scalar;

  return {n, half, padded, scalar, std::max(in_bytes, out_bytes)};
}

void seed_input(DeviceMemory& device, const BufferLayout& layout,
                const CaseSpec& spec) {
  const std::size_t count = layout.allocation_bytes / layout.scalar_bytes;
  const int n = layout.n;
  const int half = layout.half;
  const int padded = layout.padded;

  if (layout.scalar_bytes == sizeof(float)) {
    std::vector<float> host(count);
    for (std::size_t i = 0; i < count; ++i) {
      host[i] = std::sin(static_cast<float>(i + 1) * 0.173f);
    }
    if (is_real_inverse_api(spec.api)) {
      for (int b = 0; b < spec.batch; ++b) {
        host[static_cast<std::size_t>(b * padded + 1)] = 0.0f;
        if (n % 2 == 0)
          host[static_cast<std::size_t>(b * padded + n + 1)] = 0.0f;
      }
    }
    device.copy_from_host(host.data(), layout.allocation_bytes);
  } else {
    std::vector<double> host(count);
    for (std::size_t i = 0; i < count; ++i) {
      host[i] = std::sin(static_cast<double>(i + 1) * 0.173);
    }
    if (is_real_inverse_api(spec.api)) {
      for (int b = 0; b < spec.batch; ++b) {
        host[static_cast<std::size_t>(b * padded + 1)] = 0.0;
        if (n % 2 == 0)
          host[static_cast<std::size_t>(b * padded + n + 1)] = 0.0;
      }
    }
    device.copy_from_host(host.data(), layout.allocation_bytes);
  }
}

FlagfftPlanHandle make_flagfft_plan(const CaseSpec& spec,
                                     const BufferLayout& layout) {
  flagfftHandle raw = nullptr;
  flagfftResult result = FLAGFFT_NOT_SUPPORTED;
  if (spec.rank == 1) {
    result =
        flagfftPlan1d(&raw, layout.n, flagfft_type(spec.api), spec.batch);
  } else if (spec.rank == 2) {
    result = flagfftPlan2d(&raw, spec.shape[0], spec.shape[1],
                           flagfft_type(spec.api));
  } else if (spec.rank == 3) {
    result = flagfftPlan3d(&raw, spec.shape[0], spec.shape[1], spec.shape[2],
                           flagfft_type(spec.api));
  }
  check_flagfft(result, "create FlagFFT plan");
  return FlagfftPlanHandle(raw);
}

test_adaptor::RefPlanHandle make_ref_plan(const CaseSpec& spec,
                                           const BufferLayout& layout) {
  test_adaptor::RefPlanHandle plan;
  if (spec.rank == 1) {
    test_adaptor::ref_plan_1d(plan, layout.n, flagfft_type(spec.api),
                               spec.batch);
  } else if (spec.rank == 2) {
    test_adaptor::ref_plan_2d(plan, spec.shape[0], spec.shape[1],
                               flagfft_type(spec.api));
  } else if (spec.rank == 3) {
    test_adaptor::ref_plan_3d(plan, spec.shape[0], spec.shape[1],
                               spec.shape[2], flagfft_type(spec.api));
  }
  return plan;
}

void exec_flagfft(flagfftHandle plan, const CaseSpec& spec, void* input,
                  void* output) {
  switch (spec.api) {
    case FftApi::C2C:
      check_flagfft(flagfftExecC2C(plan, static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftComplex*>(output),
                                   spec.direction),
                    "flagfftExecC2C");
      break;
    case FftApi::Z2Z:
      check_flagfft(
          flagfftExecZ2Z(plan, static_cast<flagfftDoubleComplex*>(input),
                         static_cast<flagfftDoubleComplex*>(output),
                         spec.direction),
          "flagfftExecZ2Z");
      break;
    case FftApi::R2C:
      check_flagfft(flagfftExecR2C(plan, static_cast<flagfftReal*>(input),
                                   static_cast<flagfftComplex*>(output)),
                    "flagfftExecR2C");
      break;
    case FftApi::D2Z:
      check_flagfft(
          flagfftExecD2Z(plan, static_cast<flagfftDoubleReal*>(input),
                         static_cast<flagfftDoubleComplex*>(output)),
          "flagfftExecD2Z");
      break;
    case FftApi::C2R:
      check_flagfft(flagfftExecC2R(plan, static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftReal*>(output)),
                    "flagfftExecC2R");
      break;
    case FftApi::Z2D:
      check_flagfft(
          flagfftExecZ2D(plan, static_cast<flagfftDoubleComplex*>(input),
                         static_cast<flagfftDoubleReal*>(output)),
          "flagfftExecZ2D");
      break;
  }
}

void exec_ref(test_adaptor::RefPlanHandle& plan, const CaseSpec& spec,
              void* input, void* output) {
  switch (spec.api) {
    case FftApi::C2C:
      test_adaptor::ref_exec_c2c(plan, static_cast<flagfftComplex*>(input),
                                  static_cast<flagfftComplex*>(output),
                                  spec.direction);
      break;
    case FftApi::Z2Z:
      test_adaptor::ref_exec_z2z(
          plan, static_cast<flagfftDoubleComplex*>(input),
          static_cast<flagfftDoubleComplex*>(output), spec.direction);
      break;
    case FftApi::R2C:
      test_adaptor::ref_exec_r2c(plan, static_cast<flagfftReal*>(input),
                                  static_cast<flagfftComplex*>(output));
      break;
    case FftApi::D2Z:
      test_adaptor::ref_exec_d2z(plan, static_cast<flagfftDoubleReal*>(input),
                                  static_cast<flagfftDoubleComplex*>(output));
      break;
    case FftApi::C2R:
      test_adaptor::ref_exec_c2r(plan, static_cast<flagfftComplex*>(input),
                                  static_cast<flagfftReal*>(output));
      break;
    case FftApi::Z2D:
      test_adaptor::ref_exec_z2d(
          plan, static_cast<flagfftDoubleComplex*>(input),
          static_cast<flagfftDoubleReal*>(output));
      break;
  }
}

}  // namespace

BenchResult run_benchmark(const CaseSpec& spec, int warmup, int iters,
                          bool include_path) {
  const BufferLayout layout = layout_for(spec);

  DeviceMemory ff_in(layout.allocation_bytes);
  DeviceMemory ref_in(layout.allocation_bytes);

  DeviceMemory ff_out;
  DeviceMemory ref_out;
  if (spec.placement == Placement::OutOfPlace) {
    ff_out.allocate(layout.allocation_bytes);
    ref_out.allocate(layout.allocation_bytes);
  }

  seed_input(ff_in, layout, spec);
  seed_input(ref_in, layout, spec);

  FlagfftPlanHandle ff_plan = make_flagfft_plan(spec, layout);
  test_adaptor::RefPlanHandle ref_plan = make_ref_plan(spec, layout);

  auto ff_output = [&]() -> void* {
    return spec.placement == Placement::InPlace ? ff_in.get() : ff_out.get();
  };
  auto ref_output = [&]() -> void* {
    return spec.placement == Placement::InPlace ? ref_in.get() : ref_out.get();
  };

  Stream stream;
  Timer timer;

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
    exec_ref(ref_plan, spec, ref_in.get(), ref_output());
  }
  adaptor::synchronize();

  // Benchmark
  std::vector<float> ff_times;
  std::vector<float> ref_times;
  ff_times.reserve(iters);
  ref_times.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    if ((i & 1) == 0) {
      timer.start(stream.get());
      exec_ref(ref_plan, spec, ref_in.get(), ref_output());
      timer.stop(stream.get());
      ref_times.push_back(timer.elapsed_ms());

      timer.start(stream.get());
      exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
      timer.stop(stream.get());
      ff_times.push_back(timer.elapsed_ms());
    } else {
      timer.start(stream.get());
      exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
      timer.stop(stream.get());
      ff_times.push_back(timer.elapsed_ms());

      timer.start(stream.get());
      exec_ref(ref_plan, spec, ref_in.get(), ref_output());
      timer.stop(stream.get());
      ref_times.push_back(timer.elapsed_ms());
    }
  }
  stream.sync();

  TimingStats ff_stats{percentile(ff_times, 0.5), percentile(ff_times, 0.9),
                        ff_times};
  TimingStats ref_stats{percentile(ref_times, 0.5), percentile(ref_times, 0.9),
                         ref_times};
  double speedup =
      ff_stats.median_ms > 0.0 ? ref_stats.median_ms / ff_stats.median_ms : 0.0;

  BenchResult result{ff_stats, ref_stats, speedup, ""};
  if (include_path) {
    const char* desc = flagfftGetPlanDescription(ff_plan.get());
    result.plan_description = desc ? desc : "";
  }
  return result;
}

}  // namespace flagfft::cli::bench
```

- [ ] **Step 3: 提交**

```bash
mkdir -p src/cli_tools/bench
git add src/cli_tools/bench/runner.hpp src/cli_tools/bench/runner.cpp
git commit -m "feat(cli): add platform-agnostic bench runner"
```

---

### Task 6: 创建 bench/report.hpp / report.cpp

**Files:**
- Create: `src/cli_tools/bench/report.hpp`
- Create: `src/cli_tools/bench/report.cpp`

- [ ] **Step 1: 创建 report.hpp**

```cpp
#pragma once

#include <string>
#include <vector>

#include "cli_tools/bench/runner.hpp"
#include "cli_tools/common/case.hpp"

namespace flagfft::cli::bench {

std::string format_table(const std::vector<CaseSpec>& cases,
                         const std::vector<BenchResult>& results);

nlohmann::json format_json(const std::vector<CaseSpec>& cases,
                           const std::vector<BenchResult>& results,
                           int warmup, int iters);

}  // namespace flagfft::cli::bench
```

- [ ] **Step 2: 创建 report.cpp**

```cpp
#include "cli_tools/bench/report.hpp"

#include <iomanip>
#include <sstream>

#include "cli_tools/common/cli_utils.hpp"

namespace flagfft::cli::bench {

std::string format_table(const std::vector<CaseSpec>& cases,
                         const std::vector<BenchResult>& results) {
  std::ostringstream out;
  out << std::left << std::setw(12) << "shape"
      << std::setw(8) << "api"
      << std::setw(7) << "batch"
      << std::setw(11) << "direction"
      << std::setw(14) << "placement"
      << std::setw(17) << "flagfft_median_ms"
      << std::setw(14) << "ref_median_ms"
      << "speedup\n";

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto& spec = cases[i];
    const auto& r = results[i];

    std::string shape_str;
    for (std::size_t j = 0; j < spec.shape.size(); ++j) {
      if (j > 0) shape_str += "x";
      shape_str += std::to_string(spec.shape[j]);
    }

    out << std::left << std::setw(12) << shape_str
        << std::setw(8) << fft_api_name(spec.api)
        << std::setw(7) << spec.batch
        << std::setw(11) << direction_name(spec.direction)
        << std::setw(14) << placement_name(spec.placement)
        << std::fixed << std::setprecision(4)
        << std::setw(17) << r.flagfft.median_ms
        << std::setw(14) << r.reference.median_ms
        << std::setprecision(2) << r.speedup << "x\n";
  }
  return out.str();
}

nlohmann::json format_json(const std::vector<CaseSpec>& cases,
                           const std::vector<BenchResult>& results,
                           int warmup, int iters) {
  json j_cases = json::array();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto& spec = cases[i];
    const auto& r = results[i];
    json entry = case_json(spec);
    entry["timing"] = {
        {"flagfft_median_ms", r.flagfft.median_ms},
        {"flagfft_p90_ms", r.flagfft.p90_ms},
        {"ref_median_ms", r.reference.median_ms},
        {"ref_p90_ms", r.reference.p90_ms},
        {"speedup", r.speedup},
        {"warmup", warmup},
        {"iters", iters},
    };
    if (!r.plan_description.empty()) {
      entry["plan_description"] = r.plan_description;
    }
    j_cases.push_back(entry);
  }
  return {
      {"status", "passed"},
      {"command", "bench"},
      {"cases", j_cases},
  };
}

}  // namespace flagfft::cli::bench
```

- [ ] **Step 3: 提交**

```bash
git add src/cli_tools/bench/report.hpp src/cli_tools/bench/report.cpp
git commit -m "feat(cli): add bench output formatting (table + JSON)"
```

---

### Task 7: 创建 tune 占位符

**Files:**
- Create: `src/cli_tools/tune/tune.hpp`
- Create: `src/cli_tools/tune/tune.cpp`
- Delete: `src/cli_tools/tune/driver.hpp`, `driver.cpp`, `bench.hpp`, `bench.cpp`, `sqlite.hpp`, `sqlite.cpp`

- [ ] **Step 1: 删除旧 tune 文件**

```bash
git rm src/cli_tools/tune/driver.hpp src/cli_tools/tune/driver.cpp \
       src/cli_tools/tune/bench.hpp src/cli_tools/tune/bench.cpp \
       src/cli_tools/tune/sqlite.hpp src/cli_tools/tune/sqlite.cpp
```

- [ ] **Step 2: 创建 tune.hpp**

```cpp
#pragma once

namespace flagfft::cli::tune {

[[noreturn]] void tune_placeholder();

}  // namespace flagfft::cli::tune
```

- [ ] **Step 3: 创建 tune.cpp**

```cpp
#include "cli_tools/tune/tune.hpp"

#include <cstdlib>
#include <iostream>

namespace flagfft::cli::tune {

void tune_placeholder() {
  std::cerr << "tune is not yet supported\n";
  std::exit(1);
}

}  // namespace flagfft::cli::tune
```

- [ ] **Step 4: 提交**

```bash
git add src/cli_tools/tune/tune.hpp src/cli_tools/tune/tune.cpp
git commit -m "refactor(cli): replace tune subsystem with placeholder"
```

---

### Task 8: 重写 main.cpp

**Files:**
- Modify: `src/cli_tools/flagfft-cli/main.cpp`

- [ ] **Step 1: 重写 main.cpp**

```cpp
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cli_tools/bench/report.hpp"
#include "cli_tools/bench/runner.hpp"
#include "cli_tools/common/case.hpp"
#include "cli_tools/tune/tune.hpp"

namespace {

using namespace flagfft::cli;
using namespace flagfft::cli::bench;

struct Args {
  std::string command;
  std::vector<CaseSpec> cases;
  CaseSpec prototype;
  int warmup = 10;
  int iters = 100;
  bool print_path = false;
  bool json_output = false;
};

void print_usage() {
  std::cout
      << "Usage: flagfft-cli bench --rank 1|2|3 --shape <format> [options]\n"
         "       flagfft-cli tune\n"
         "\n"
         "Bench options:\n"
         "  --rank 1|2|3             Transform rank (required)\n"
         "  --shape A,B,...           Comma-separated shapes, x for dims\n"
         "                           rank 1: --shape 16,32,64\n"
         "                           rank 2: --shape 23x42,128x64\n"
         "                           rank 3: --shape 128x128x128\n"
         "  --api c2c|z2z|r2c|d2z|c2r|z2d   FFT type (default: c2c)\n"
         "  --batch N                 Batch size, rank 1 only (default: 1)\n"
         "  --direction forward|inverse       (default: forward)\n"
         "  --placement out-of-place|in-place (default: out-of-place)\n"
         "  --warmup N                Warmup iterations (default: 10)\n"
         "  --iters N                 Timed iterations (default: 100)\n"
         "  --json                    Output JSON report\n"
         "  --print-path              Include plan description in output\n";
}

int parse_positive(const std::string& name, const char* value,
                   bool allow_zero = false) {
  try {
    const std::string token = value;
    std::size_t consumed = 0;
    const int result = std::stoi(token, &consumed);
    if (consumed != token.size()) {
      throw AssertionFailure("invalid value for " + name);
    }
    if ((allow_zero && result < 0) ||
        (!allow_zero && result <= 0)) {
      throw AssertionFailure(name + " must be " +
                             (allow_zero ? "non-negative" : "positive"));
    }
    return result;
  } catch (const std::invalid_argument&) {
    throw AssertionFailure("invalid value for " + name);
  } catch (const std::out_of_range&) {
    throw AssertionFailure("invalid value for " + name);
  }
}

Args parse_args(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "--help" ||
      std::string(argv[1]) == "-h") {
    print_usage();
    std::exit(0);
  }
  Args args;
  args.command = argv[1];
  if (args.command != "bench" && args.command != "tune") {
    throw AssertionFailure("unknown command: " + args.command);
  }
  if (args.command == "tune") return args;

  std::vector<std::vector<int>> shapes;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    auto need = [&](const std::string& name) -> const char* {
      if (++i >= argc) throw AssertionFailure(name + " requires a value");
      return argv[i];
    };
    if (argument == "--help" || argument == "-h") {
      print_usage();
      std::exit(0);
    } else if (argument == "--rank") {
      args.prototype.rank = parse_rank(need("--rank"));
    } else if (argument == "--shape") {
      const std::string raw = need("--shape");
      std::size_t start = 0;
      while (start <= raw.size()) {
        std::size_t end = raw.find(',', start);
        const std::string part =
            raw.substr(start, end == std::string::npos ? end : end - start);
        if (!part.empty()) shapes.push_back(parse_shape(part));
        if (end == std::string::npos) break;
        start = end + 1;
      }
    } else if (argument == "--api") {
      args.prototype.api = parse_fft_api(need("--api"));
    } else if (argument == "--batch") {
      args.prototype.batch = parse_positive("--batch", need("--batch"));
    } else if (argument == "--direction") {
      args.prototype.direction = parse_direction(need("--direction"));
    } else if (argument == "--placement") {
      args.prototype.placement = parse_placement(need("--placement"));
    } else if (argument == "--warmup") {
      args.warmup = parse_positive("--warmup", need("--warmup"), true);
    } else if (argument == "--iters") {
      args.iters = parse_positive("--iters", need("--iters"));
    } else if (argument == "--json") {
      args.json_output = true;
    } else if (argument == "--print-path") {
      args.print_path = true;
    } else {
      throw AssertionFailure("unknown argument: " + argument);
    }
  }

  if (shapes.empty()) shapes.push_back(args.prototype.shape);
  for (auto& shape : shapes) {
    CaseSpec spec = args.prototype;
    spec.shape = std::move(shape);
    args.cases.push_back(std::move(spec));
  }

  // Validate batch only for rank 1
  for (const auto& c : args.cases) {
    if (c.rank != 1 && c.batch != 1) {
      throw AssertionFailure("--batch is only supported with --rank 1");
    }
    if (static_cast<int>(c.shape.size()) != c.rank) {
      throw AssertionFailure("--shape dimension count does not match --rank");
    }
  }

  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    if (args.command == "tune") {
      flagfft::cli::tune::tune_placeholder();
    }

    std::string reason;
    if (!has_cuda_device(reason)) {
      return emit_json_report({
          {"status", "skipped"},
          {"reason", reason},
      });
    }

    std::vector<BenchResult> results;
    for (const auto& spec : args.cases) {
      results.push_back(
          run_benchmark(spec, args.warmup, args.iters, args.print_path));
    }

    if (args.json_output) {
      std::cout << format_json(args.cases, results, args.warmup, args.iters)
                       .dump(2)
                << "\n";
    } else {
      std::cout << format_table(args.cases, results);
    }

    return 0;
  } catch (const flagfft::cli::CliException& error) {
    std::cerr << "error: " << error.what() << "\n";
    return error.exit_code();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return flagfft::cli::kExitRuntimeError;
  }
}
```

- [ ] **Step 2: 提交**

```bash
git add src/cli_tools/flagfft-cli/main.cpp
git commit -m "refactor(cli): rewrite main with new bench/tune dispatch"
```

---

### Task 9: 更新 CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 修改 CMakeLists.txt 第 90-139 行**

将 `FLAGFFT_BUILD_CLI` 块替换为：

```cmake
if(FLAGFFT_BUILD_CLI)
    # cuFFT is used only by test_adaptor (not by CLI directly).
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET CUDA::cufft)
        message(FATAL_ERROR "FLAGFFT_BUILD_CLI requires the CUDA::cufft target")
    endif()
    add_library(flagfft_cli_common OBJECT
        src/cli_tools/common/plan_handles.cpp
        src/cli_tools/common/cli_utils.cpp
        src/cli_tools/common/case.cpp
    )
    target_compile_features(flagfft_cli_common PUBLIC cxx_std_20)
    target_include_directories(flagfft_cli_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
    target_link_libraries(flagfft_cli_common PUBLIC
        flagfft
        nlohmann_json::nlohmann_json
    )

    add_executable(flagfft-cli
        src/cli_tools/flagfft-cli/main.cpp
        src/cli_tools/bench/runner.cpp
        src/cli_tools/bench/report.cpp
        src/cli_tools/tune/tune.cpp
    )
    target_compile_features(flagfft-cli PRIVATE cxx_std_20)
    target_include_directories(flagfft-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src/utils
        src/exec
        src
    )
    target_link_libraries(flagfft-cli
        PRIVATE
            flagfft
            flagfft_cli_common
            flagfft_test_adaptor
            CUDA::cufft
            nlohmann_json::nlohmann_json
    )
    install(TARGETS flagfft-cli RUNTIME DESTINATION "bin")
endif()
```

关键变化：
- `flagfft_cli_common` 移除 `runner.cpp`、移除 `CUDA::cufft` 链接
- `flagfft-cli` 源文件替换为新的 bench + tune 文件
- `flagfft-cli` 移除 `SQLite3::SQLite3` 链接

- [ ] **Step 2: 提交**

```bash
git add CMakeLists.txt
git commit -m "build(cli): update CMakeLists for new file structure"
```

---

### Task 10: 构建验证

**Files:** 无新建

- [ ] **Step 1: 构建**

```bash
cmake -S /workspace/FlagFFT-dev/.claude/worktrees/refactor-cli \
      -B /workspace/FlagFFT-dev/.claude/worktrees/refactor-cli/build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON -DFLAGFFT_BUILD_TESTS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++
ninja -C build flagfft-cli
```

注意：需要先确保 `build/_deps` 已预填充（复制自主构建目录）以避免网络下载。

- [ ] **Step 2: 验证 binary**

```bash
./build/flagfft-cli --help
```

预期输出：新的使用说明，包含 `bench` 和 `tune` 命令，不包含 `test`、`--suite`、`--plan-api`。

- [ ] **Step 3: 测试 bench 基本流程**

```bash
./build/flagfft-cli bench --rank 1 --shape 16 --warmup 2 --iters 5
```

预期：表格输出，包含 timing 和 speedup。

- [ ] **Step 4: 测试 JSON 输出**

```bash
./build/flagfft-cli bench --rank 1 --shape 16 --warmup 2 --iters 5 --json
```

预期：格式正确的 JSON，包含 `status: "passed"` 和 `cases` 数组。

- [ ] **Step 5: 测试 tune 占位符**

```bash
./build/flagfft-cli tune
```

预期：`error: tune is not yet supported`，退出码 1。

- [ ] **Step 6: 测试多 shape bench**

```bash
./build/flagfft-cli bench --rank 1 --shape 16,32,64 --warmup 2 --iters 5
```

预期：3 行的表格输出。

- [ ] **Step 7: 提交（如有修复）**

```bash
git add -u
git commit -m "fix(cli): build fixes after refactoring"
```

---

### Task 11: 更新 Python 测试

**Files:**
- Modify: `tests/cli/test_flagfft_cli.py`

- [ ] **Step 1: 重写测试文件**

读取 `/workspace/FlagFFT-dev/.claude/worktrees/refactor-cli/tests/cli/test_flagfft_cli.py`，进行以下修改：

(1) 删除所有 test 相关的测试函数：`test_plan_contract`, `test_api_errors_suite`, `test_correctness_api_and_direction`, `test_all_test_suite`, `test_expressible_unsupported_case`, `test_correctness_output_fields`

(2) 删除所有 tune 相关的测试：`test_tune_roundtrip_c2c`, `test_tune_retune`, `test_tune_validation`, `test_tune_lockfile`

(3) 修改 `test_help` 以匹配新的帮助输出：

```python
def test_help(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "--help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "flagfft-cli bench" in result.stdout
    assert "flagfft-cli tune" in result.stdout
    assert "--rank" in result.stdout
    assert "--shape" in result.stdout
    # 旧参数不应出现
    assert "test --suite" not in result.stdout
    assert "--plan-api" not in result.stdout
    assert "--launches-per-sample" not in result.stdout
```

(4) 修改参数拒绝测试：

```python
@pytest.mark.parametrize(
    "arguments",
    [
        ("bench", "--rank", "1", "--shape", "16garbage"),
        ("bench", "--rank", "2", "--shape", "16x4tail"),
        ("bench", "--rank", "1", "--shape", "16", "--batch", "2garbage"),
        ("bench", "--rank", "1", "--shape", "16", "--iters", "1tail"),
    ],
)
def test_rejects_integer_options_with_trailing_characters(
    invoke_cli, arguments
) -> None:
    result, report = invoke_cli(*arguments)
    assert result.returncode == 1
```

(5) 添加 bench 基础测试：

```python
def test_bench_basic(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "1", "--shape", "16",
         "--warmup", "2", "--iters", "5"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "speedup" in result.stdout


def test_bench_json(flagfft_cli) -> None:
    import json
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "1", "--shape", "16",
         "--warmup", "2", "--iters", "5", "--json"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "passed"
    assert report["command"] == "bench"
    assert len(report["cases"]) == 1
    assert "timing" in report["cases"][0]


def test_bench_multi_shape(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "1", "--shape", "16,32",
         "--warmup", "2", "--iters", "5"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    lines = result.stdout.strip().split("\n")
    # header + 2 data rows
    assert len(lines) >= 3


def test_tune_placeholder(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "tune"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "not yet supported" in result.stderr.lower()


def test_batch_only_rank1(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "2", "--shape", "16x16",
         "--batch", "4"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 1


def test_rank_shape_mismatch(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "1", "--shape", "16x16"],
        text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
```

- [ ] **Step 2: 提交**

```bash
git add tests/cli/test_flagfft_cli.py
git commit -m "test(cli): update Python tests for refactored CLI"
```

---

### Task 12: 运行完整测试套件并修复

**Files:** 无新建

- [ ] **Step 1: 运行 pytest**

```bash
pytest tests/cli/test_flagfft_cli.py -v
```

预期：所有测试通过（需要 CUDA 设备的测试可能跳过）。

- [ ] **Step 2: 运行 ctest**

```bash
cd build && ctest --output-on-failure
```

预期：ctest 套件不受影响，全部通过。

- [ ] **Step 3: 提交修复**

如有失败，修复后：

```bash
git add -u
git commit -m "fix(cli): address test failures after refactoring"
```
