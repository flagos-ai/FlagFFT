# CLI 重构设计

## 概述

移除 `test` 职能（已由 ctest 接管），重构 `bench` 使所有平台相关代码通过 `adaptor`/`test_adaptor` 封装，`tune` 变为占位符。

## 架构

```
src/cli_tools/
├── common/
│   ├── case.hpp / case.cpp       # CaseSpec, parse_shape, 无平台依赖
│   ├── cli_utils.hpp / cli_utils.cpp  # 参数解析, JSON/表格输出
│   ├── plan_handles.hpp / plan_handles.cpp  # FlagfftPlanHandle (仅 flagfft)
│   └── runtime_raii.hpp          # DeviceMemory, Stream, Timer 别名
├── bench/
│   ├── runner.hpp / runner.cpp   # bench 执行逻辑, 不包含任何 cufft*.h
│   └── report.hpp / report.cpp   # 输出格式化 (人类可读表格 + JSON)
├── tune/
│   └── tune.cpp / tune.hpp       # 占位符, 输出错误信息并 exit(1)
└── flagfft-cli/
    └── main.cpp                  # 入口, 参数解析, 命令分发
```

## CLI 命令

### bench

```
flagfft-cli bench --rank 1|2|3 --shape <format> [options]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rank` | 1/2/3 | (必需) | 变换维度 |
| `--shape` | string | (必需) | 逗号分隔, x 连接维度, 见下文 |
| `--api` | c2c/z2z/r2c/d2z/c2r/z2d | c2c | FFT 类型 |
| `--batch` | int | 1 | 批量大小, 仅 rank=1 有效 |
| `--direction` | forward/inverse | forward | 变换方向 |
| `--placement` | in-place/out-of-place | out-of-place | 内存布局 |
| `--warmup` | int | 10 | 预热迭代次数 |
| `--iters` | int | 100 | 计时迭代次数 |
| `--json` | flag | off | 输出 JSON 格式 |
| `--print-path` | flag | off | 输出 plan_description |

**shape 格式：**

```
--rank 1 --shape 16,32,64,128           # 4 个一维 shape
--rank 2 --shape 23x42,128x64           # 2 个二维 shape
--rank 3 --shape 128x128x128,64x64x64   # 2 个三维 shape
```

### tune

```
flagfft-cli tune [任意参数]
```

输出 "tune is not yet supported" 并以错误码退出 (exit 1)。旧 tune 代码全部删除。

### test

删除。用户使用 ctest 运行测试。

## bench 数据流

```
CaseSpec → generate_input() → device mem 分配 → FlagFFT plan + Ref plan
                                                      ↓
                                         warmup (N 次, 不计时)
                                                      ↓
                               计时循环 (M 次, 交替执行两方)
                                                      ↓
                               median/P90 → speedup = ref_median / flagfft_median
                                                      ↓
                                              report (table / json)
```

## runner 接口

```cpp
struct TimingStats {
    double median_ms;
    double p90_ms;
    std::vector<double> samples;
};

struct BenchResult {
    TimingStats flagfft;
    TimingStats reference;
    double speedup;
    std::string plan_description;
};

BenchResult run_benchmark(const CaseSpec& case, int warmup, int iters, bool print_path);
```

## 平台依赖隔离

| 层次 | 文件 | 平台依赖 |
|------|------|----------|
| adaptor | `adaptor/backend/cuda/adaptor.cpp` | CUDA Driver API (Memory, Stream, EventTimer) |
| test_adaptor | `adaptor/backend/cuda/test_adaptor.cpp` | cuFFT (RefPlanHandle, ref_plan_*, ref_exec_*) |
| bench runner | `cli_tools/bench/runner.cpp` | **无** — 仅通过 adaptor + test_adaptor 接口 |
| bench report | `cli_tools/bench/report.cpp` | **无** |
| CLI main | `cli_tools/flagfft-cli/main.cpp` | **无** |

## CaseSpec 清理

移除：
- `cufft_type()` — 平台相关
- `unsupported_reason()` — 原 test 预检查, bench 不需要
- `PlanApi::PlanMany` — 不支持

保留：
- `FftApi` 枚举 (C2C, Z2Z, R2C, D2Z, C2R, Z2D)
- `Placement` 枚举
- 所有 shape/batch/api/direction/placement 字段
- `parse_shape()` 解析器（适配新格式）

## 输出格式

### 默认人类可读表格

```
shape      api   batch  direction  placement   flagfft_median  ref_median  speedup
16         c2c   1      forward    out-of-place 0.012 ms        0.008 ms    0.67x
32         c2c   1      forward    out-of-place 0.015 ms        0.009 ms    0.60x
```

### JSON (--json)

```json
{
  "status": "passed",
  "command": "bench",
  "cases": [
    {
      "shape": [16],
      "api": "c2c",
      "batch": 1,
      "direction": "forward",
      "placement": "out-of-place",
      "timing": {
        "flagfft_median_ms": 0.012,
        "flagfft_p90_ms": 0.014,
        "ref_median_ms": 0.008,
        "ref_p90_ms": 0.009,
        "speedup": 0.67,
        "warmup": 10,
        "iters": 100
      },
      "plan_description": "..."
    }
  ]
}
```

## 退出码

| 状态 | 退出码 |
|------|--------|
| 成功 | 0 |
| 参数无效 | 1 |
| 运行时错误 | 2 |
| tune 不支持 | 1 |

## 已删除

- `src/cli_tools/common/runner.cpp` / `runner.hpp` — 逻辑移至 `bench/runner.*`
- `src/cli_tools/tune/driver.*` `bench.*` `sqlite.*` — 旧 tune 代码
- `case.cpp` 中的 `cufft_type()`, `unsupported_reason()`
- `cli_utils.cpp` 中的 `check_cufft()`, `cufft_result_name()`
- `plan_handles.cpp` 中的 `CufftPlanHandle`
- `main.cpp` 中的 test 命令、`--suite`、`--stream`、`--launches-per-sample`、tune 参数
