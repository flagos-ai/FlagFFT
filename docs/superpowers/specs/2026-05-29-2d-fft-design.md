# FlagFFT 2D FFT 设计文档

## 概述

当前 FlagFFT 仅支持 rank=1 的 1D FFT。`flagfftPlan2d`/`flagfftPlan3d` API 存在但返回 `FLAGFFT_NOT_SUPPORTED`。本文档描述 2D FFT 的实现设计，参考 rocFFT 的三层分解策略。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 实现策略 | RTRT 先行，RC 后优化 | 复用现有 reshape kernel，实现风险低 |
| R2C 2D 策略 | 行 R2C + 列 C2C | 效率高，符合 cuFFT 语义 |
| Plan 架构 | 新增 TwoDimPlanNode | 清晰的 2D 语义，与现有 FourStepPlanNode 平行 |
| Transpose 实现 | 新增分块 transpose kernel | 大矩阵带宽利用率高 |

## 架构总览

```
用户调用 flagfftPlan2d(plan, nx, ny, type)
    ↓
build_plan() → 检测 rank=2 → 新路径
    ↓
构建 TwoDimPlanNode:
  ├── row_plan: 1D FFT (length=n[1], batch=batch*n[0])
  └── col_plan: 1D FFT (length=n[0], batch=batch*n[1])
    ↓
compile_raw_2d_node() → CompiledRaw2DNode
    ↓
RTRT 执行:
  Step 1: row_fft(input → temp1)
  Step 2: tiled_transpose(temp1 → temp2)
  Step 3: col_fft(temp2 → temp1)
  Step 4: tiled_transpose(temp1 → output)
```

## RTRT vs RC 对比

### RTRT（Row-Transpose-Row-Transpose）— 4 个 kernel

```
输入 (batch, M, N)
  → Row FFT: 沿 N 方向做 FFT（行连续，高效）
  → Transpose: (batch, M, N) → (batch, N, M)（需要额外显存 + 带宽）
  → Col FFT: 沿 M 方向做 FFT（现在变成了"行"，连续访问）
  → Transpose: (batch, N, M) → (batch, M, N)（写回原始布局）
```

- 优点：列 FFT 变成行 FFT，kernel 不需要特殊 IO 模式
- 缺点：2 次 transpose = 额外 2× 显存带宽 + 2 个临时缓冲区

### RC（Row-Column）— 2 个 kernel

```
输入 (batch, M, N)
  → Row FFT: 沿 N 方向做 FFT（行连续，高效）
  → Col FFT: 沿 M 方向做 FFT（列间距 = N，需要 strided_batch IO 模式）
```

- 优点：无 transpose，只用 1 个临时缓冲区，带宽最优
- 缺点：列 FFT 需要新的 `strided_batch` IO 模式

## Phase 1：Plan 描述符验证 + TwoDimPlanNode

### 目标

让 `flagfftPlan2d` 通过描述符验证，构建两个独立的 1D plan。

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/utils/include/flagfft/base.hpp` | `PlanNodeKind` 新增 `TwoDim` 枚举值 |
| `src/utils/include/flagfft/plan.hpp` | 新增 `TwoDimPlanNode` 结构体 |
| `src/exec/c_api_internal.cpp` | 新增 `is_supported_2d_desc()` 验证函数 |
| `src/exec/c_api_plan.cpp` | `build_plan()` 新增 rank=2 分支 |

### TwoDimPlanNode 定义

```cpp
struct TwoDimPlanNode : PlanNode {
    int64_t n0, n1;           // 2D shape: (n0, n1)
    PlanNodePtr row_plan;     // axis-1 FFT (length=n1, batch=batch*n0)
    PlanNodePtr col_plan;     // axis-0 FFT (length=n0, batch=batch*n1)
    PlanNodeKind kind = PlanNodeKind::TwoDim;
};
```

### 2D 描述符验证规则

`is_supported_2d_desc()` 验证：
- `desc.rank == 2`
- `desc.n.size() == 2`，`n[0] > 0 && n[1] > 0`
- `desc.istride == 1 && desc.ostride == 1`（紧凑行优先）
- `desc.idist >= n[0] * n[1]`，`desc.odist >= n[0] * n[1]`
- R2C/C2R 的 embed 尺寸验证（最内维 n[i]/2+1）

### build_plan() rank=2 路径

1. 为 axis-1（行方向，length=n[1]）构建 1D plan，batch = batch * n[0]
2. 为 axis-0（列方向，length=n[0]）构建 1D plan，batch = batch * n[1]
3. 创建 `TwoDimPlanNode` 组合两个 1D plan
4. 两个 1D plan 复用现有 `PlanBuilder::build(n)` 流程

### 验证

- `flagfftPlan2d(&plan, nx, ny, type)` 返回 `FLAGFFT_SUCCESS`
- `flagfftGetPlanDescription` 输出 2D plan 的节点树

## Phase 2：分块 Transpose + CompiledRaw2DNode

### 目标

在执行路径中实现 (batch, M, N) <-> (batch, N, M) 的分块 complex transpose，以及 2D FFT 的 RTRT 执行。

### 修改文件

| 文件 | 变更 |
|------|------|
| `python/flagfft_codegen/kernels.py` | 新增 `_build_tiled_transpose_kernel_source()` |
| `src/utils/include/flagfft/codegen.hpp` | `TritonCompiler` 新增 `compile_raw_2d_node()` |
| `src/codegen/compiler.cpp` | 实现 `compile_raw_2d_node()` |
| `src/exec/raw_nodes.hpp` | 声明 `CompiledRaw2DNode` |
| `src/exec/raw_nodes.cpp` | 实现 `CompiledRaw2DNode` |

### 分块 Transpose Kernel 设计

```python
def _build_tiled_transpose_kernel_source(dtype: str, tile_size: int = 32) -> str:
    """
    分块 (batch, M, N) -> (batch, N, M) transpose kernel。
    使用 shared memory tiling 提升大矩阵带宽利用率。

    每个 program 处理一个 tile_size×tile_size 的块：
    1. 从 global memory 读取 tile 到 shared memory（合并读取）
    2. 在 shared memory 中转置
    3. 写回 global memory（合并写入）
    """
```

### CompiledRaw2DNode 执行流程

```cpp
class CompiledRaw2DNode : public CompiledRawNode {
    // 持有：
    CompiledRawNodePtr row_fft;      // 行方向 1D FFT
    CompiledRawNodePtr col_fft;      // 列方向 1D FFT
    JitKernel transpose_fwd;         // (M,N)→(N,M) transpose
    JitKernel transpose_inv;         // (N,M)→(M,N) transpose

    void execute(input, output, context) {
        // RTRT:
        row_fft->execute(input, temp1, {batch*n0, ...});
        transpose_fwd->launch(temp1, temp2, {batch, n0, n1});
        col_fft->execute(temp2, temp1, {batch*n1, ...});
        transpose_inv->launch(temp1, output, {batch, n1, n0});
    }
};
```

### 关键实现细节

- **临时缓冲区管理**：在 `execute()` 中通过 context 获取临时内存，避免每次分配。需要 2 个缓冲区：temp1（input 大小）、temp2（transpose 后大小，当 n0≠n1 时不同）
- **batch 维度处理**：transpose 对每个 batch 独立处理，batch 维度保留在最外层
- **R2C/C2R 包装**：新建 `CompiledRaw2DR2CNode` 包装 `CompiledRaw2DNode`，处理 real→complex expand 和 half-spectrum pack

## Phase 3：测试 + R2C/C2R 2D + CLI

### 3.1 Correctness 测试

新增 `ctest/test_2d_correctness.cpp`，覆盖：

| 测试类别 | 尺寸 | 方向 | 批量 |
|----------|------|------|------|
| C2C 2D | 16×16, 64×64, 128×256, 256×256, 997×1021 | forward + inverse | 1, 4 |
| Z2Z 2D | 同上 | forward + inverse | 1, 4 |
| R2C + C2R 2D | 同上 | roundtrip | 1, 4 |
| D2Z + Z2D 2D | 同上 | roundtrip | 1, 4 |

测试框架：复用现有 `flagfft_test.h` 的精度比较函数，与 cuFFT 参考结果对比（`ref_plan_2d()` + `ref_exec_*` 已在 `test_adaptor.h` 中声明）。

### 3.2 2D R2C/C2R 实现

行 R2C + 列 C2C 路径：

```
R2C 2D forward:
  输入 (batch, n0, n1) real
  → 行 R2C: output (batch, n0, n1/2+1) complex
  → transpose: (batch, n0, n1/2+1) → (batch, n1/2+1, n0)
  → 列 C2C: output (batch, n1/2+1, n0) complex
  → transpose: (batch, n1/2+1, n0) → (batch, n0, n1/2+1)

C2R 2D inverse:
  输入 (batch, n0, n1/2+1) complex
  → 列 C2C inverse
  → transpose
  → 行 C2R: output (batch, n0, n1) real
```

注意：列方向的 transpose 需要处理 half-spectrum 尺寸（n1/2+1 而非 n1）。

### 3.3 CLI 集成

`flagfft-cli bench` 已有 `--rank 2` 选项，Phase 3 完成后只需确保 `flagfftPlan2d` 返回 `FLAGFFT_SUCCESS` 即可自动工作。

### 3.4 现有测试更新

- `ctest/test_plan.cpp`：将 2D 测试从 `EXPECT_EQ(FLAGFFT_NOT_SUPPORTED)` 改为 `EXPECT_EQ(FLAGFFT_SUCCESS)`
- `tests/ctest/test_correctness.py`：新增 2D 测试条目

## Phase 4：RC 路径优化（消除 Transpose）

### 目标

对于行列长度不同的 2D FFT，消除 transpose 的显存带宽开销。

### 新增 LeafIoMode: strided_batch

```python
LeafIoMode = Literal["contiguous", "four_step_row", "four_step_col", "strided_batch"]
```

`strided_batch` 模式：kernel 参数包含 `outer_stride`（行间距或列间距），每个 `program_id` 处理一个 1D FFT，通过 `outer_stride` 跳到下一行/列。

### RC 执行流程

```
Step 1: row_fft(input → temp, batch=batch*n0, fft_length=n1, inner_stride=1)
        — 每行连续，共 batch*n0 行

Step 2: col_fft(temp → output, batch=batch*n1, fft_length=n0, inner_stride=n1)
        — 每列间距 n1，共 batch*n1 列，无需 transpose
```

### 策略选择逻辑

```cpp
if (n0 <= 64 && n1 <= 64 && n0 * n1 * sizeof(complex) <= max_smem) {
    // Single kernel (Phase 5)
} else if (has_strided_column_kernel(n0)) {
    // RC: 2 kernels, no transpose
} else {
    // RTRT: 4 kernels (Phase 2)
}
```

### R2C/C2R RC 路径

- 行 R2C（inner_stride=1）→ 列 C2C（inner_stride=n1/2+1，half-spectrum）

## Phase 5（后续）：Single Kernel 优化

对小尺寸 2D FFT（≤32×32），实现单一 fused kernel：
- 在 shared memory 中完成两个维度的 FFT
- 需要新的 codegen：split radices into two groups, generate 2D twiddle tables
- 参考 rocFFT 的 CS_KERNEL_2D_SINGLE
- 此阶段为远期优化，不在本次实现范围内

## 性能分析

### 1D 执行效率瓶颈

| 瓶颈 | 影响 | 严重程度 |
|------|------|----------|
| 首次 kernel 编译（Python subprocess + Triton JIT） | 首次调用延迟 ~1-5s/kernel | 中（缓存后无影响） |
| FourStep reshape kernel（非 fused 路径） | 额外 global memory round-trip | 中 |
| plan mutex（每个 plan 一把锁） | 无并发执行同一 plan | 低 |
| 临时缓冲区 O(batch×n×depth) | 大尺寸 2D 可能 OOM | 需关注 |

### 2D 特有瓶颈

| 瓶颈 | RTRT | RC |
|------|------|-----|
| Transpose 带宽 | 2× extra global memory | 0 |
| 临时缓冲区 | 2× input size | 1× input size |
| Kernel launch 数 | 4（或 2 fused） | 2 |
| 行列长度不等时 | 两次编译 | 两次编译 |

## 文件变更清单

| 文件 | 变更类型 | Phase |
|------|----------|-------|
| `src/utils/include/flagfft/base.hpp` | 修改：PlanNodeKind 新增 TwoDim | 1 |
| `src/utils/include/flagfft/plan.hpp` | 修改：新增 TwoDimPlanNode | 1 |
| `src/exec/c_api_internal.cpp` | 修改：新增 is_supported_2d_desc() | 1 |
| `src/exec/c_api_plan.cpp` | 修改：build_plan 新增 rank=2 路径 | 1 |
| `python/flagfft_codegen/kernels.py` | 修改：新增 tiled transpose kernel | 2 |
| `src/utils/include/flagfft/codegen.hpp` | 修改：TritonCompiler 新增 compile_raw_2d_node | 2 |
| `src/codegen/compiler.cpp` | 修改：实现 compile_raw_2d_node() | 2 |
| `src/exec/raw_nodes.hpp` | 修改：声明 CompiledRaw2DNode | 2 |
| `src/exec/raw_nodes.cpp` | 修改：实现 CompiledRaw2DNode | 2 |
| `ctest/test_2d_correctness.cpp` | 新增 | 3 |
| `ctest/test_plan.cpp` | 修改：2D 测试期望 SUCCESS | 3 |
| `tests/ctest/test_correctness.py` | 修改：新增 2D 测试条目 | 3 |

## 实施顺序

```
Phase 1 (描述符 + Plan Node)
    ↓
Phase 2 (Transpose + 2D Execution)
    ↓  ← 功能完成，可跑 C2C/Z2Z 2D
Phase 3 (测试 + R2C/C2R + CLI)
    ↓  ← 测试通过，可交付
Phase 4 (RC 优化)
    ↓  ← 性能优化
Phase 5 (Single kernel) ← 后续
```

Phase 1-2 是最小可行路径，预计 1.5 天。Phase 3 是测试完善，预计 1 天。Phase 4 是性能优化，预计 0.5 天。
