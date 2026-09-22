# MACA 1D single 优化与验收工作单

基线：`45eb2a8`，single 指 batch=1；目标 speedup = mcFFT median / FlagFFT median >= 0.8。
历史 2026-09-18 验收仅用于定位问题，不能替代当前提交基线。

## 当前默认策略（本分支）

经完整 opt-in 1D single 矩阵验证后，MACA 的 `rank=1 && batch=1` 请求默认采用
以下已验证配置：`EXCHANGE=direct_all`、`MAX_WARPS=8`、`INNER_PACK=8`（仍受
FP64、mixed-radix、shared-memory 和寄存器安全门控约束）、`SPLIT_ORDER=lsb`，并
默认启用适用的 Bluestein boundary/four-step fusion。R2C/C2R 也属于这个作用域。

该策略不会作用于 MACA batch>1、2D 或 3D；其他后端保持原行为。设置
`FLAGFFT_MACA_1D_SINGLE=0` 可回退到旧默认，单项 `FLAGFFT_MACA_*` 变量仍优先，
便于 A/B 和紧急回滚。默认与 opt-out 生成物使用独立的 JIT cache 子目录和进程内
cache key，避免两种 codegen 互相覆盖。

完整 opt-in 1D single 验收（5 warmup / 30 iterations）中，136/136 个正确性 case
通过，全部 1D single case 几何平均为 `0.8224x`；该结果是开启本策略的性能依据，
不是对每个尺寸都达到 0.8x 的承诺。原始结果：
`results/20260922_081717_maca_1d_single_optin_dev637_4gpu`。

当前已完成小尺寸 FP32 C2C 的正式验收：1024/2048 使用 direct，997 使用
direct+leaf fusion；NumPy/mcFFT 双向正确性通过，三轮独立进程双向性能共
18/18 个结果均 >=0.8，最小 0.8110。每轮 warmup=2000、iters=100，不删样本。
这不等同于 12 个算子的每个尺寸都通过 0.8x；完整矩阵的汇总见上面的当前策略记录。

| 已验尺寸 | forward 三轮 speedup 范围 | inverse 三轮 speedup 范围 |
|---|---:|---:|
| 1024 | 1.0351–1.0536 | 1.0000–1.0175 |
| 2048 | 0.8919–0.8933 | 0.8800–0.8933 |
| 997 | 0.8110–0.8175 | 0.8189–0.8268 |

证据：`results/20260921_160430_maca_single_final_small` 的原始 6 个 JSON、
`final_summary.json` 与主 agent 独立复核的 `root_independent_review.json`。
大 CT、mixed 和大 prime 的正式复测仍在进行。

## 分工与边界

- `maca_validation`：环境核验、当前基线、唯一 GPU 测量调度者、候选独立验收。
- `maca_exchange`：MACA portable exchange 原型，先覆盖 FP32 1024/2048，再检查 four-step 1048576。
- `maca_prime`：小 prime 两段 leaf 融合，随后评估 four-step 边界融合和算法选择。
- 主 agent：方向选择、源码审阅、集成与验收判定。

各任务使用根目录下独立 `FlagFFT-dev-maca-single-*` worktree。本地主机修改并提交，再同步已提交版本到云端同名 worktree；容器只编译运行。暂不修改 dev 分支。

## 实验纪律

1. MACA 容器 `flagfft-maca-card4-adapt`，仅用物理 GPU 4。所有 GPU 测量由 validation 串行执行，容器锁 `/tmp/flagfft-maca-gpu4.lock`；运行前检查其他占用，不能仅依赖团队锁。
2. 每次运行记录 commit、源码校验、git status、import 路径、SDK/Triton/mcFFT/设备环境与实际 launch metadata。
3. 开关改变 codegen 时明确隔离生成模块、JIT cache 和 plan/tune cache；同一代码缓存不可混用为不同变体。
4. correctness 先于 performance；统一 acceptance runner 对 NumPy 校验。当前设备已经复现 5/50 预热不足，后续从 warmup=200、iters=100 开始，检查逐次样本是否稳定；候选交错 A/B 至少三轮。不能事后裁剪慢样本以通过阈值。
5. 结果统一存放根 `results/<YYYYMMDD_HHMMSS>_maca_single_*`，远端记录回传同一结果目录，保留原始结果与失败证据。
6. 正常 event 计时保留作为验收口径。kernel 诊断、连续提交等额外计时分开报告，不能混用或替代原口径。

## 阶段与停止条件

| 阶段 | 实验 | 判断依据 |
|---|---|---|
| 基线 | C2C 双方向：CT 16/1024/2048/8192/16384/185640/1048576；prime 997/1009/524287 | 当前代码正确性、逐 case mcFFT 对比、实际路径 |
| 交换 | join/静态张量交换对照原 gather+where | 1024/2048 完整 FFT 延迟与编译资源；不能仅用源码 gather 数宣称改善 |
| 复用 | 交换应用于 CT 1048576 与 prime 大卷积 | 端到端收益、两个 four-step pass 时间、输出正确性 |
| prime | 997 两段融合，再 1009 Rader/Bluestein 对照；大 prime 边界融合 | 实际 kernel 数、冷编译时间、steady-state 时间、双方向正确性 |
| 集成 | 胜出候选叠加，对照固定基线 | 独立验收，不简单相乘单项收益 |

若 lowering 不支持或出现静默错误，保留最小复现并修改表达方式；若收益不显著，保留实验结论而不默认启用。编译失败、超时或缺少 GPU 正确性不能称为完成的优化。

## 发布前验收

- 12 个 CT/prime single 算子完整尺寸与方向矩阵：C2C/Z2Z/R2C/C2R/D2Z/Z2D。
- 报告每个算子几何平均、每 case speedup、>=0.8 覆盖率、相对当前基线变化；不能用总平均遮盖严重回退。
- 检查共享代码影响的 batch、2D、3D 代表形状及 CPU/codegen 测试；开关关闭时保持原语义。
- 只有证据充分的配置才考虑默认启用；未通过完整验收的原型保持 opt-in。
- 目标未达到时明确报告实际水平与剩余瓶颈，不把原型交付等同于达成 0.8。

## 第一轮设备证据（1024 C2C forward）

以下是单轮 5 warmup / 50 iterations 的筛选结果，NumPy 与 mcFFT 正确性均通过；仍需重复 A/B、inverse 和完整矩阵验收。设备为 C550，Torch 2.8.0+metax3.7.2.0、Triton 3.6.0。

| 表达方式 | FlagFFT / μs | mcFFT / μs | speedup | LLVM IR barrier call | 动态 shared / B |
|---|---:|---:|---:|---:|---:|
| 基线 gather+where | 66.048 | 29.696 | 0.4496 | 159 | 4096 |
| join 后 gather | 47.872 | 29.952 | 0.6257 | 71 | 8192 |
| 静态 transpose，保留下一 stage gather | 47.872 | 29.952 | 0.6257 | 67 | 8192 |
| direct：静态置换并直接 split 到下一 stage 寄存器 | 27.648 | 29.696 | 1.0741 | 7 | 8192 |

direct 的 TTGIR gather 从 80 降为 0。最终 ELF 报告 private memory=0；基线与 direct 均没有 spill 证据。barrier 数来自 LLVM IR，不能冒称最终 ISA 动态执行计数。单点数据与编译产物共同支持继续优化级间交换；尚不能外推所有尺寸或将默认策略替换。

原始结果位于工作区根 `results/`：

- `20260921_143430_maca_single_baseline_focus`（基线为保留的 partial suite，已完成结果有效）
- `20260921_143720_maca_single_exchange_join`（同样为 partial suite）
- `20260921_144215_maca_single_transpose_1024`（完整单 case）
- `20260921_144503_maca_single_direct_1024`（完整单 case）

集成原型 `6ee6f03` 的容器 CPU/codegen 回归为 283 passed、8 subtests passed；GPU 验收继续由 validation 串行推进。

## 第二轮筛选与计时审计

下表仍是 5/50 单轮筛选，不能据此宣布达到目标。后续发现的预热不足使跨轮性能归因尤其不可靠；编译资源和正确性记录仍可独立使用。

| case / 候选 | FlagFFT / μs | mcFFT / μs | speedup | 设备正确性 |
|---|---:|---:|---:|---|
| 2048 / direct v1 | 45.824 | 35.072 | 0.7654 | forward PASS |
| 2048 / direct v2，先 pad 整个寄存器组 | 41.216 | 35.072 | 0.8509 | forward PASS |
| 997 / direct v2 + 两段融合 | 76.800 | 60.416 | 0.7867 | forward PASS |
| 1048576 / direct，inner pack=4 | 219.904 | 112.896 | 0.5134 | forward PASS |

- 2048 v2 的 LLVM IR barrier 从 71 降至 39，shared=16 KiB，mtreg=90，private memory=0。剩余 32 个 barrier 与拆分后的 layout conversion 对应。
- 997 prepare/finish 各为 39 barrier、16 KiB shared、mtreg=88、private memory=0；融合边界没有新增这些 barrier。
- MSB-first split 的两个融合 kernel 资源与 v2 完全相同，未得到支持继续投入的证据，默认仍为 LSB。
- 1048576 的四类 row/column 正反向 kernel 都没有 gather，均为 7 barrier、32 KiB shared、private memory=0。下一步测 inner pack 和复数向量化访存；不能继续把此 case 的差距归因于原 gather 链。

对应结果目录：`20260921_144955_maca_single_direct_2048`、`20260921_145420_maca_single_direct_v2_2048`、`20260921_145420_maca_single_direct_fused_v2_997`、`20260921_144955_maca_single_direct_1048576`。

### 预热不足的直接证据

`results/20260921_151240_maca_single_timing_stability/round1_w5_i50.json` 保存了同一个纯 fusion 997 程序的有序样本：前约 18 次 FlagFFT≈192 μs、mcFFT≈62 μs；随后两库共同下降，后半段分别稳定在约 73.5 μs、26.6 μs。奇偶迭代交换执行顺序仍出现同样趋势。前后设备时钟快照不足以判断瞬时原因，当前仅确认存在慢热/状态变化。

`FLAGFFT_BENCH_SAMPLES=1` 可选导出 `timing.flagfft_samples_ms` 和 `timing.ref_samples_ms`，两数组按原迭代顺序保存；偶数迭代先 reference，奇数迭代先 FlagFFT。导出不改变计时循环或默认 JSON。所有后续阈值验收必须先检查样本稳定性，再比较相同协议下的完整采样统计。

## 原型开关与尚未完成的验收

- `FLAGFFT_MACA_EXCHANGE=direct`：power-of-two leaf 的静态交换和直接寄存器拆分；mixed leaf 仅部分 join fallback。
- `FLAGFFT_MACA_EXCHANGE=direct_all`：mixed radix 的寄存器维补齐后 join/gather，尚未通过设备验收。
- `FLAGFFT_MACA_BLUESTEIN_LEAF_FUSION=1`：batch=1 的两段 leaf 边界融合。
- `FLAGFFT_MACA_BLUESTEIN_FOUR_STEP_FUSION=1`：batch=1、两个 child leaf <=1024 的 four-step 边界融合，尚未通过设备验收。

上述开关在原型提交时默认关闭；当前分支只在 MACA 1D single 作用域默认采用它们。
显式环境变量仍可覆盖。除 `FLAGFFT_MACA_1D_SINGLE` 这一策略位外，单项环境开关
不进入所有持久化 codegen/cache key，独立 A/B 实验仍必须隔离生成目录和 cache；
不能把单项实验接口当成可安全并发切换的发布配置。

尚欠稳定协议复测、双方向和 FP64/real 全矩阵、mixed、大 prime、batch/2D/3D 回归。当前交付状态是有正确性和编译证据的优化原型，不是整个 1D single 达成 0.8×。

## 充分预热后的同环境归因

`results/20260921_152650_maca_single_control_round1_repeat` 使用同 source
`6ee6f03`、相同 SHA256 的 CLI，按 A→C→D 顺序运行，各自隔离生成目录与 Triton
cache；每 case warmup=200、iters=100。下表为每个变体单进程内三次完整 case
重复，不等同于三个独立进程的交错验收。D 不重复测试不受融合开关影响的 CT。

| case | A 原路径 / μs | C direct / μs | D direct+fusion / μs | 候选 / mcFFT 性能比 |
|---|---:|---:|---:|---:|
| 1024 | 28.928–29.184 | 14.336–14.592 | — | C: 1.035–1.053 |
| 2048 | 40.704–40.960 | 18.944–19.200 | — | C: 0.892–0.893 |
| 997 | 73.472 | 34.048 | 32.512 | C: 0.774–0.782；D: 0.819–0.827 |

纯融合 B 在独立稳定性诊断中的三轮为 72.960–73.472 μs，速度比 0.354–0.362。
A 的实际 plan 是五 kernel generic Bluestein，B 为两 kernel boundary leaf；A 的
2048 FFT 最终 fatbin 与初始基线相同，排除了把融合缓存误认成原路径的疑点。
交换优化是主要收益，997 上 C→D 的融合增益约 1.047×。

主 agent 独立读取并检查全部有序样本，摘要在同目录 `root_control_review.json`。
C1024 首个 case 前后半存在轻微继续变快，后续重复稳定；最终跨进程复测将提高
warmup 至 2000，并保留所有样本。随后补齐 C997 双向、C1024/2048 inverse 和
D997 inverse 的 NumPy/mcFFT 正确性，均通过；C1024/2048 与 D997 forward
沿用先前同语义原型的 NumPy 证据。双向性能及 dtype/API 覆盖仍不完整。

### 大 CT 的负实验

`results/20260921_152150_maca_single_pack2_warm`：1048576 C2C forward，
direct + inner pack=2，NumPy 与平台正确性通过，200/100 下
391.168 μs / 112.640 μs = 0.288×。实际 4 warps / 256 threads。
与 pack4 相同的 7 barrier、0 gather、private memory=0，shared 从 32 KiB 减至
16 KiB，故不能把回退归因于新增同步或 spill。pack2 淘汰。IO 分段模型、作用域
检查与 mixed codelet 算术限制见
[`maca_exchange_audit.md`](maca_exchange_audit.md)。

### 大 CT 的 pack8 候选

1048576 C2C forward 使用相同 200/100 协议：

| pack | FlagFFT / μs | mcFFT / μs | speedup | 实际 shared | ELF private |
|---:|---:|---:|---:|---:|---:|
| 2 | 391.168 | 112.640 | 0.2880 | 16 KiB | 0 |
| 4 | 217.856 | 112.640 | 0.5170 | 32 KiB | 0 |
| 8 | 128.768 | 112.640 | 0.8748 | 32 KiB | 0 |

pack8 通过 NumPy 与平台正确性，实际 8 warps / 512 threads；相对 pack4 再快
1.692×。pack8 前后半样本接近，pack4 仍有小幅抖动（前后半 214.656/222.464 μs），
参考库中位数相同。还须补跨进程和 inverse，不外推 FP64 或 batch。

pack8 未增加 barrier，也未 spill；每线程寄存器上升，LLIR 同时出现 packed FMA。
global 访问仍为 scalar f32，故不能把总收益归因于宽访存或仅归因于 coalescing。
这些资源变化说明原 gather 实现的 pack 调优结论需要在新交换方式下重新测量。

结果：`20260921_153000_maca_single_pack8_warm`、
`20260921_154730_maca_single_pack4_warm`。统一 runner 的方向筛选回归
`tests/python/test_run_tests.py` 为 273 passed；正式 MACA 验收脚本默认改为
2000/100，诊断用的低预热对照仍保留。
