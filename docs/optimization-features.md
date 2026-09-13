# FlagFFT 优化特性统计

本文以当前 `main`（`5e0c4bc`）为准，整理已经存在的性能特性、平台策略和相关修复。性能数据来自 `results/` 下的 MUSA S5000、NVIDIA A100 测试；“已验证范围”是实测范围，不代表代码只能在该范围工作。

## 特性总表

| 特性 | 作用层级 | 当前实现与默认状态 | 当前条件或参数 | 关闭/备用路径 | 验证证据 | 配置化建议 |
|---|---|---|---|---|---|---|
| 成对奇数基蝶形 | 内核生成 | radix 11/13/17/19 直接使用成对输出公式，默认开启 | 所有后端、精度、尺寸和调用路径 | 各 radix 的旧 bundled codelet 可由历史提交恢复 | MUSA/A100；2D、3D、Rader、DirectDFT；Python 数学测试 | 高优先级开关；作为通用生成选项 |
| 混合基数四步内核打包 | 内核生成 | 根据线程数、共享内存和叶计划自动计算 `inner_pack` | 由 dtype、叶长度、lanes、因子数和资源预算决定 | `inner_pack=1` 的普通四步内核 | MUSA 2D 慢 case；A100 2D/3D | 高优先级参数；平台资源放入配置 |
| MUSA 小混合叶 8 路打包 | 内核生成 | MUSA backend 下 128–256、单 lane、多因子叶允许最多 8 路 | `_mthreads_small_mixed_leaf()`；16 路已实测回退 | 通用资源计算或 `inner_pack=1` | MUSA `46189×48` 等 2D case | 平台配置项，默认值 8；保留 auto |
| TLE fused twiddle | 内核生成/四步执行 | 满足条件的四步内核融合 twiddle，默认按现有判定启用 | `use_tle_fused_twiddle()` 及尺寸、dtype 条件 | 非融合 twiddle 四步路径 | 既有 codegen 测试与 2D/3D 回归 | 平台特性开关；与打包策略分开 |
| Bluestein full-leaf 融合 | 计划编译/内核生成 | 将卷积 FFT 边界融合到单个叶内核 | 当前对 complex64，以及已验证的 CUDA/A100、MUSA FP64 batch=1 | 普通 Bluestein 多内核流水线 | A100/MUSA 2D BS correctness 与性能 | 算法特性开关；条件表配置 |
| Bluestein 四步边界融合 | 计划编译/内核生成 | 小叶四步卷积使用融合边界内核 | 叶长度小于 512，且满足 dtype/平台条件 | 普通四步 Bluestein 路径 | MUSA `8191×1009` 等 BS case | 算法特性开关；阈值配置 |
| MUSA 批量 Bluestein 边界融合 | 计划编译 | FP64 卷积叶节点和工作区满足资源条件时融合边界；16381 专用算法选择已移除 | MUSA capability 3.1、FP64、batch≥16、叶节点与字节预算约束 | 非融合分块；算法由默认计划或实测选择 | 见 [batch 泛化](batch-generalization.md) | 通用资源条件，保留既有 8191 策略 |
| Rader DC 复用 | 内核数据流 | 复用第一次 FFT 的零频项，删除 finalize 中按输出块重复的全输入归约 | 所有现有 Rader 路径，FP32/FP64、正逆向、batch | 默认计划选择或实测算法选择 | 见 [batch 泛化](batch-generalization.md) | 通用代数优化，无长度/设备特判 |
| Batch 复数算法与分解调优 | 计划选择 | C2C/Z2Z 比较默认计划、分解方向、质数 Rader/Bluestein，先校验再计时；移除 131072 专用偏好 | 显式 `tune --api ... --batch ...`，按设备、类型、方向、精确 batch 缓存 | 无有效缓存时默认 cost model | 见 [batch 泛化](batch-generalization.md) | 一次性实测调优，非隐式启动开销 |
| Batch packed-real child | 计划编译/执行 | 偶数 real transform 使用半长 complex child；紧密 batch 支持，特殊布局按需回退 | auto 为已验证平台 FP64 和资源范围；`FLAGFFT_PACKED_REAL=1` 可实验性强制 | 原始 real 路径；batch 子叶上限保守控制 | 见 [batch 泛化](batch-generalization.md) | half-length 算法不是对所有规模都更快 |
| MUSA S5000 2D graph 策略 | 执行 | batch=1 的 complex 2D 禁用 graph replay | MUSA capability 3.1、batch=1 | CUDA/A100 和其他情况保留 graph | MUSA 128² 约 3–5 倍收益；A100 A/B 证明 graph 更快 | 平台执行配置；禁止写死在编译器判断中 |
| Bluestein chunk byte budget | 执行/内存 | 按 256 MiB 工作区预算切分 batch，避免大卷积超出显存 | `kBluesteinChunkByteBudget` | 由 batch 一次性执行 | 大尺寸 BS batch 回归 | 通用内存策略；预算配置化 |
| kernel metadata `inner_pack` 读取 | 正确性/执行 | C++ 运行时读取生成器写入的 `inner_pack`，并用于 grid 计算 | 所有带打包参数的四步和融合内核 | 无；错误读取会导致网格错误 | 新旧 metadata 回归测试；MUSA BS 性能与 correctness | 必须始终生效，不做性能开关 |
| tuned plan 数据库 | 计划选择基础设施 | 可启用/禁用调优数据库并覆盖数据库路径 | `FLAGFFT_TUNE_DISABLE`、`FLAGFFT_TUNE_DB` | 默认 cost model | 现有 tuned-plan 代码路径 | 作为统一配置基础设施 |

## 分类结论

### 通用生成特性

成对蝶形、资源约束下的 `inner_pack`、TLE twiddle 和 Bluestein 融合属于生成器能力。它们应由统一的 `FeatureConfig` 控制，默认值使用 `auto`，由计划和资源条件决定是否启用。

### 平台调优特性

MUSA S5000 的 8 路打包、8191 点 Bluestein 选择、FP64 packed-real child 和 batch-1 2D graph 策略都依赖设备架构。它们不应散落为 `device_type == ...` 的判断，应该由平台配置表提供阈值、交叉点和默认策略。

### 必须始终生效的修复

`inner_pack` metadata 读取属于正确性修复；关闭它会使生成端和执行端的网格不一致，不能作为可选性能开关。类似的参数签名、缓存 key 和 metadata 完整性也应归入这一类。

## 配置化优先级

1. 先统一配置模型：`auto/on/off`、数值参数、平台默认值和实际生效配置记录。
2. 将成对蝶形、`inner_pack`、TLE twiddle、Bluestein 融合和 packed-real child 接入该模型。
3. 将 MUSA/A100 的架构阈值迁移到平台配置表，保留现有默认行为。
4. 最后把 tuned-plan 数据库、chunk budget 和 graph 策略接入同一配置入口。

每次 benchmark 应记录配置快照；凡是影响生成代码或计划的配置，都必须参与 JIT cache key，避免不同配置复用同一份内核缓存。

## 已知覆盖范围

- 成对蝶形已在 MUSA S5000 与 NVIDIA A100 的 FP32/FP64、2D/3D、Rader 和 DirectDFT 路径验证。
- MUSA 全量 1D+2D+3D（含 batch=1/256）为 592/592 Passed；分组结果见 `results/20260913_115203_musa_full_1d2d3d_incremental_csv/API_GROUPED_REPORT.md`。
- A100 对成对蝶形和 3D 混合基数有专项验证；纯二次幂控制组基本不受影响。
- 现有条件来自已测硬件，迁移到新 GPU 前应先保留 `auto` 并做小规模 A/B。
