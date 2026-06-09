# FlagFFT 统一测试运行器设计文档

## 概述

为 FlagFFT 构建统一的测试运行器（`tools/run_tests.py`），参考 FlagGems 的 `run_tests.py` 架构，支持多 GPU 并行执行正确性测试和性能基准测试。使用双层 YAML 配置（算子元数据 + 测例参数矩阵），以测例为最小执行单位。

## 动机

当前 FlagFFT 存在两套独立的测试体系：
- **ctest 正确性测试**：43 个 C++ GTest 可执行文件，编译时硬编码尺寸数组
- **benchmark 性能测试**：通过 `flagfft-cli bench`，360 个预定义测例

问题：
1. 两套测试独立运行，无法统一管理
2. ctest 尺寸编译时固定，无法灵活选择测例子集
3. 缺少统一的 JSON 汇总报告
4. 不支持多 GPU 并行加速

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 算子定义 | 变换类型 × 维度（c2c_1d, r2c_2d 等） | 与现有 ctest 文件命名一致 |
| 测例定义 | 变换类型 + 尺寸 + 批大小 + 方向 + 缩放 | 最完整参数组合 |
| 配置格式 | 双层 YAML（operators.yaml + test_matrix.yaml） | 分离关注点 |
| 执行方式 | 直接调用 ctest / flagfft-cli bench | 最简单直接 |
| ctest 参数化 | 命令行参数 --nx, --batch, --direction, --scale | 单测例粒度 |
| 测例粒度 | 单测例（每进程一个参数组合） | 最细粒度控制 |
| 输出格式 | JSON summary + 控制台 LiveDisplay | 支持 CI 集成 |
| 与现有关系 | 替换现有 pytest 测试和 benchmark 框架 | 统一入口 |
| 架构参考 | 复刻 FlagGems run_tests.py | 成熟的多 GPU 模式 |

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    run_tests.py (主入口)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 加载配置      │  │ 展开测例      │  │ 启动 Workers │  │
│  │ operators.yaml│→│ 算子×参数矩阵 │→│ 多GPU并行    │  │
│  │ test_matrix   │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────┬───────┘  │
│                                              │          │
│  ┌───────────────────────────────────────────▼───────┐  │
│  │              Worker Process (per GPU)              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │  │
│  │  │ work_queue  │→ │ 正确性测试   │→ │ 性能测试  │  │  │
│  │  │ (测例列表)  │  │ ctest --nx  │  │ cli bench │  │  │
│  │  └─────────────┘  └─────────────┘  └──────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                                              │          │
│  ┌───────────────────────────────────────────▼───────┐  │
│  │           LiveDisplay (实时进度)                    │  │
│  │  进度条 + 每 GPU 状态 + 滚动结果                    │  │
│  └───────────────────────────────────────────────────┘  │
│                                              │          │
│  ┌───────────────────────────────────────────▼───────┐  │
│  │           summary.json (最终输出)                   │  │
│  │  环境信息 + 每算子精度/性能结果                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 组件设计

### 1. 配置层

#### conf/operators.yaml

算子元数据定义，每个算子对应一个变换类型×维度组合。

```yaml
ops:
  - id: c2c_1d
    description: "Complex-to-Complex 1D FFT"
    labels: [c2c, 1d, complex]
    ctest: test_exec_c2c          # ctest 可执行文件前缀
    cli_type: c2c                 # flagfft-cli --type 参数
    rank: 1                       # FFT 维度
    directions: [fwd, inv]        # 支持的方向
    algorithms: [ct, bs]          # 支持的算法
    stages:
      - stable: '1.0'

  - id: r2c_1d
    description: "Real-to-Complex 1D FFT"
    labels: [r2c, 1d, real]
    ctest: test_exec_r2c
    cli_type: r2c
    rank: 1
    directions: [fwd]
    algorithms: [ct, bs]
    stages:
      - stable: '1.0'

  - id: c2c_2d
    description: "Complex-to-Complex 2D FFT"
    labels: [c2c, 2d, complex]
    ctest: test_exec_c2c
    cli_type: c2c
    rank: 2
    directions: [fwd, inv]
    algorithms: [2d]
    stages:
      - stable: '1.0'

  # ... 其他算子: z2z_1d, d2z_1d, z2d_1d, c2r_1d,
  #     z2z_2d, r2c_2d, c2r_2d, d2z_2d, z2d_2d
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 算子唯一标识，格式 `{type}_{rank}d` |
| description | string | 算子描述 |
| labels | list | 标签，用于筛选 |
| ctest | string | ctest 可执行文件名前缀 |
| cli_type | string | flagfft-cli 的 --type 参数值 |
| rank | int | FFT 维度（1, 2, 3） |
| directions | list | 支持的方向（fwd, inv） |
| algorithms | list | 支持的算法（ct, bs, rader, 2d） |
| stages | list | 成熟度阶段 |

#### conf/test_matrix.yaml

测例参数矩阵，定义尺寸、批大小、缩放等参数空间。

```yaml
# 1D 尺寸 - Cooley-Tukey（平滑数，可被 {2..19} 的因数分解）
sizes_ct:
  - 16
  - 32
  - 64
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
  - 8192
  - 16384

# 1D 尺寸 - Bluestein/Rader（素数/复合数）
sizes_bs:
  - 23
  - 67
  - 997
  - 1009

# 2D 尺寸 (nx, ny)
sizes_2d:
  - [16, 16]
  - [32, 32]
  - [64, 64]
  - [128, 128]
  - [256, 256]
  - [512, 512]
  - [1024, 1024]
  - [16, 1024]
  - [997, 16]

# 批大小
batches:
  - 1
  - 4
  - 256

# 输入缩放因子
scales:
  - 0.00000000000000000001    # 2^-20
  - 1.0
  - 100000000000000000000.0   # 2^20

# 测例组合规则
# 每个规则定义：sizes 引用、batches 引用、scales 引用
combinations:
  ct:
    sizes: sizes_ct
    batches: [1]
    scales: [1.0]
  bs:
    sizes: sizes_bs
    batches: [1]
    scales: [1.0]
  full:
    sizes: sizes_ct
    batches: batches
    scales: scales
  2d:
    sizes: sizes_2d
    batches: [1, 4]
    scales: [1.0]
```

**组合规则说明**：

| 规则 | 用途 | 测例数量（每个算子） |
|------|------|---------------------|
| ct | 快速冒烟测试（CT 算法） | 11 sizes × 1 batch × 1 scale × directions |
| bs | 快速冒烟测试（BS 算法） | 4 sizes × 1 batch × 1 scale × directions |
| full | 完整测试（所有参数组合） | 11 sizes × 3 batches × 3 scales × directions |
| 2d | 2D 测试 | 9 sizes × 2 batches × 1 scale × directions |

### 2. 测试执行层

#### tools/run_tests.py

**命令行接口**：

```bash
# 运行所有 stable 算子
python tools/run_tests.py

# 运行指定算子
python tools/run_tests.py --ops c2c_1d,r2c_1d

# 运行指定阶段
python tools/run_tests.py --stages stable,beta

# 指定 GPU
python tools/run_tests.py --gpus 0,1,2,3

# 指定组合规则
python tools/run_tests.py --combination full

# 只运行正确性测试
python tools/run_tests.py --accuracy-only

# 只运行性能测试
python tools/run_tests.py --performance-only

# 输出详细日志
python tools/run_tests.py -v
```

**核心函数**：

```python
def load_operators(path: str) -> list[dict]:
    """加载 conf/operators.yaml"""

def load_test_matrix(path: str) -> dict:
    """加载 conf/test_matrix.yaml"""

def expand_test_cases(ops: list, matrix: dict, combination: str) -> list[dict]:
    """将算子 × 参数矩阵展开为具体测例列表"""

def get_env(gpu_ids: list[int]) -> dict:
    """返回 GPU 可见性环境变量"""

def worker_proc(gpu_id: int, work_queue: Queue, display_queue: Queue):
    """Worker 进程：从队列取测例，依次执行正确性和性能测试"""

def run_accuracy(gpu_id: int, case: dict) -> dict:
    """调用 ctest 可执行文件运行正确性测试"""

def run_performance(gpu_id: int, case: dict) -> dict:
    """调用 flagfft-cli bench 运行性能测试"""

def parse_accuracy_output(output: str) -> dict:
    """解析 ctest JSON 输出"""

def parse_perf_output(output: str) -> dict:
    """解析 flagfft-cli bench JSON 输出"""

def display_loop(queue: Queue, display: LiveDisplay, n_workers: int):
    """控制台实时进度显示"""

def main():
    """主入口：加载配置、展开测例、启动 workers、汇总结果"""
```

#### LiveDisplay 类

复刻 FlagGems 的终端实时进度显示：

```
[GPU 0] accuracy c2c_1d  nx=1024 batch=1  [OK    2.3s]
[GPU 1] accuracy c2c_1d  nx=2048 batch=1  [OK    3.1s]
[GPU 0] perf     c2c_1d  nx=1024 batch=1  [OK    5.2s]
[GPU 1] perf     c2c_1d  nx=2048 batch=1  [OK    6.8s]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45/360 (12.5%)
```

### 3. CTest 参数化改造

#### ctest/main.cpp 改造

添加命令行参数解析，支持运行时指定测例参数：

```cpp
#include <gtest/gtest.h>
#include "flagfft_test.h"
#include <cstring>
#include <cstdlib>

struct TestParams {
    int nx = 0;           // 0 = 使用默认尺寸数组
    int ny = 0;           // 0 = 使用默认尺寸数组
    int batch = 0;        // 0 = 使用默认批大小
    int direction = -1;   // -1 = fwd+inv, 0 = fwd, 1 = inv
    double scale = -1.0;  // -1 = 使用默认缩放
    bool json_output = false;
    const char* json_file = nullptr;
};

TestParams g_test_params;

int main(int argc, char** argv) {
    // 解析自定义参数（在 GTest 初始化之前）
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--nx=", 5) == 0) {
            g_test_params.nx = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--ny=", 5) == 0) {
            g_test_params.ny = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--batch=", 8) == 0) {
            g_test_params.batch = atoi(argv[i] + 8);
        } else if (strncmp(argv[i], "--direction=", 12) == 0) {
            const char* dir = argv[i] + 12;
            if (strcmp(dir, "fwd") == 0) g_test_params.direction = 0;
            else if (strcmp(dir, "inv") == 0) g_test_params.direction = 1;
        } else if (strncmp(argv[i], "--scale=", 8) == 0) {
            g_test_params.scale = atof(argv[i] + 8);
        } else if (strcmp(argv[i], "--json") == 0) {
            g_test_params.json_output = true;
        } else if (strncmp(argv[i], "--json-file=", 12) == 0) {
            g_test_params.json_file = argv[i] + 12;
            g_test_params.json_output = true;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);

    // 初始化 CUDA test adaptor（cuFFT 参考实现）
    flagfft::test::init_test_adaptor();

    // 注册 JSON 结果监听器
    if (g_test_params.json_output) {
        auto* listener = new flagfft::test::JsonResultListener(
            g_test_params.json_file);
        ::testing::UnitTest::GetInstance()->listeners().Append(listener);
    }

    return RUN_ALL_TESTS();
}
```

#### ctest/flagfft_test.h 改造

添加 `TestParams` 外部声明和辅助函数：

```cpp
// 在 flagfft_test.h 中添加
extern struct TestParams g_test_params;

// 辅助函数：根据命令行参数过滤尺寸
inline std::vector<int> filter_sizes_1d(const int* sizes, int count) {
    if (g_test_params.nx > 0) {
        // 检查请求的尺寸是否在原始数组中
        for (int i = 0; i < count; ++i) {
            if (sizes[i] == g_test_params.nx)
                return {g_test_params.nx};
        }
        // 不在原始数组中，仍然返回（允许运行任意尺寸）
        return {g_test_params.nx};
    }
    return std::vector<int>(sizes, sizes + count);
}

// 辅助函数：根据命令行参数过滤批大小
inline std::vector<int> filter_batches(const int* batches, int count) {
    if (g_test_params.batch > 0) {
        return {g_test_params.batch};
    }
    return std::vector<int>(batches, batches + count);
}

// 辅助函数：根据命令行参数过滤缩放
inline std::vector<double> filter_scales(const double* scales, int count) {
    if (g_test_params.scale >= 0) {
        return {g_test_params.scale};
    }
    return std::vector<double>(scales, scales + count);
}
```

#### ctest/test_exec_*.cpp 改造示例

以 `test_exec_c2c_fwd_ct_s.cpp` 为例：

```cpp
// 改造前
static constexpr int kSizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
INSTANTIATE_TEST_SUITE_P(CT, C2CFwdSingle, ::testing::ValuesIn(kSizes));

// 改造后
auto sizes = filter_sizes_1d(kDefaultSizesCT, std::size(kDefaultSizesCT));
INSTANTIATE_TEST_SUITE_P(CT, C2CFwdSingle, ::testing::ValuesIn(sizes));
```

#### JSON 结果监听器

```cpp
namespace flagfft::test {

class JsonResultListener : public ::testing::EmptyTestEventListener {
public:
    explicit JsonResultListener(const char* filename)
        : filename_(filename ? filename : "gtest_result.json") {}

    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        nlohmann::json result;
        result["total"] = unit_test.test_to_run_count();
        result["passed"] = unit_test.successful_test_count();
        result["failed"] = unit_test.failed_test_count();
        result["skipped"] = unit_test.skipped_test_count();
        result["duration"] = unit_test.elapsed_time();

        // 收集失败详情
        for (int i = 0; i < unit_test.total_test_suite_count(); ++i) {
            const auto* suite = unit_test.GetTestSuite(i);
            for (int j = 0; j < suite->total_test_count(); ++j) {
                const auto* test = suite->GetTestInfo(j);
                if (test->result()->Failed()) {
                    result["failures"].push_back({
                        {"name", test->name()},
                        {"message", test->result()->test_part_summaries()}
                    });
                }
            }
        }

        std::ofstream(filename_) << result.dump(2);
    }

private:
    std::string filename_;
};

} // namespace flagfft::test
```

### 4. flagfft-cli bench 扩展

确保 `flagfft-cli bench` 的 JSON 输出格式与 run_tests.py 期望的一致：

```json
{
  "type": "c2c",
  "direction": "fwd",
  "nx": 1024,
  "batch": 1,
  "latency_flagfft_us": 123.45,
  "latency_cufft_us": 150.67,
  "speedup": 1.22,
  "plan_description": "LeafPlanNode(radix=4, passes=5)"
}
```

### 5. 输出格式

#### summary.json

```json
{
  "timestamp": "2026-06-09 15:00:00",
  "env": {
    "flagfft": "0.2.3",
    "cuda": "12.4",
    "gpu": "NVIDIA A100-SXM4-80GB",
    "gpu_count": 4
  },
  "config": {
    "combination": "ct",
    "stages": ["stable"],
    "gpus": [0, 1, 2, 3]
  },
  "result": {
    "c2c_1d": {
      "accuracy": {
        "status": "Passed",
        "total": 11,
        "passed": 11,
        "failed": 0,
        "skipped": 0,
        "duration": 12.3,
        "details": {
          "failed": {},
          "skipped": {}
        }
      },
      "performance": {
        "status": "Passed",
        "duration": 45.6,
        "data": {
          "fp32": {
            "cases": [
              {
                "nx": 1024,
                "batch": 1,
                "latency_flagfft": 123.45,
                "latency_cufft": 150.67,
                "speedup": 1.22
              }
            ],
            "avg_speedup": 1.18
          }
        }
      }
    },
    "r2c_1d": {
      "accuracy": { "..." : "..." },
      "performance": { "..." : "..." }
    }
  },
  "summary": {
    "total_ops": 10,
    "accuracy_passed": 10,
    "accuracy_failed": 0,
    "performance_passed": 10,
    "performance_failed": 0,
    "total_duration": 320.5
  }
}
```

### 6. 算子清单

完整的算子列表（基于变换类型 × 维度）：

| ID | 变换类型 | 维度 | 方向 | 算法 |
|----|----------|------|------|------|
| c2c_1d | C2C | 1D | fwd, inv | ct, bs |
| z2z_1d | Z2Z | 1D | fwd, inv | ct, bs |
| r2c_1d | R2C | 1D | fwd | ct, bs |
| c2r_1d | C2R | 1D | inv | ct, bs |
| d2z_1d | D2Z | 1D | fwd | ct, bs |
| z2d_1d | Z2D | 1D | inv | ct, bs |
| c2c_2d | C2C | 2D | fwd, inv | 2d |
| z2z_2d | Z2Z | 2D | fwd, inv | 2d |
| r2c_2d | R2C | 2D | fwd | 2d |
| c2r_2d | C2R | 2D | inv | 2d |
| d2z_2d | D2Z | 2D | fwd | 2d |
| z2d_2d | Z2D | 2D | inv | 2d |

## 实施阶段

### Phase 1: 配置文件
- 创建 `conf/operators.yaml`
- 创建 `conf/test_matrix.yaml`

### Phase 2: CTest 参数化
- 修改 `ctest/main.cpp` 添加命令行参数解析
- 修改 `ctest/flagfft_test.h` 添加辅助函数
- 修改所有 `test_exec_*.cpp` 使用动态尺寸选择
- 修改 `test_2d_correctness.cpp` 支持参数化
- 添加 JSON 结果输出支持

### Phase 3: 测试运行器
- 实现 `tools/run_tests.py`
- 实现 `tools/consts.py`
- 实现 LiveDisplay 类
- 实现多 GPU worker 架构
- 实现 JSON summary 汇总

### Phase 4: 清理和文档
- 移除 `tests/ctest/`（pytest ctest 包装）
- 移除 `tests/cli/`（CLI 集成测试）
- 移除 `benchmark/test_bench.py` 和 `benchmark/utils/`
- 更新 `README.md` 文档
- 更新 `pyproject.toml` 配置

## 依赖

### 新增依赖
- PyYAML（加载 YAML 配置）
- nlohmann_json（CTest JSON 输出，已有 FetchContent）

### 可移除依赖
- pytest-benchmark（被 run_tests.py 替代）
- pytest-xdist（不再需要并行测试分发）

## 风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| CTest 参数化改动量大（40+ 文件） | 高 | 使用统一的辅助函数减少重复代码 |
| ctest 可执行文件不支持任意尺寸 | 中 | flagfft-cli bench 已支持任意尺寸，可作为性能测试的备选 |
| 多 GPU 环境不可用 | 低 | run_tests.py 支持单 GPU 模式 |
| JSON 输出格式不兼容 | 低 | 定义明确的输出规范 |
