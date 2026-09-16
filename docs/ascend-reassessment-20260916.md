# Ascend 适配重新评估与后续计划（2026-09-16）

## 结论与范围

当前起点是已经运行的 NPU C API Direct DFT 基线，不再是从零实现 runtime。
下一目标是 **910B4 / CANN 9.0 上连续 1D complex64 C2C 的可扩展 FFT**：
保留 N≤128 的 Direct DFT；新增 16–65536 的二次幂 Stockham 执行路径。
正逆变换都不归一化，第一阶段正式验收 out-of-place 和用户 stream。
不把 Python 探针成功等同于 C++ JIT/C API 已经支持大尺寸。

本轮基于 `codex/ascend-plain-mvp-20260916`，原始 FlagFFT commit `322f099`，
外部 libtriton_jit commit `7c3720f`。后者需要通过
`FLAGFFT_TRITON_JIT_SOURCE_DIR` 显式传入，不能误用旧的 `deps/libtriton_jit`。
没有合并 main：main 已有 batch、IX 等后续更改，集成应单列验收，避免改变本次评估基线。

## 已有实现与实际边界

| 模块 | 现状 | 后续动作 |
|---|---|---|
| CMake / ACL adaptor | 已有 NPU 设备、内存、stream、event | 复用；补独立 C++ 生命周期/计时验收 |
| libtriton_jit | 已有加载、参数打包、launch、嵌入式导入支持 | 复用；验证多阶段、workspace、冷启动 |
| C API / planner | 连续 1D C2C complex64，正长度≤128，正 batch | 保留保护，按测试完成度开放新路径 |
| Direct DFT | 普通 tl.load/store/reduce；CPU 生成 DFT 表 | 作为小尺寸设备交叉检查；CPU FP64 仍是独立 oracle |
| StockhamPlanNode | 已有节点、描述、JSON 及部分 cost 支持 | 缺少 raw execution/codegen 接入，不能当现成 FFT 使用 |
| TLE GPU leaf / Four-Step | 现有 NVIDIA 映射不可直接启用 | 先不移植，不作为正确性版本的前提 |
| CLI / ctest / 矩阵 | NPU 顶层明确禁用通用 CLI/tests | 新增 CPU reference 路径，再接入统一 runner |

证据位置：`src/adaptor/backend/npu/adaptor.cpp`、`src/codegen/jit_kernel.cpp`、
`src/plan/auto_candidates.cpp`、`src/exec/c_api_plan.cpp`、
`src/exec/c_api_internal.cpp`、`src/exec/raw_nodes.cpp`、
`python/flagfft_codegen/kernels_special.py`、`python/flagfft_codegen/metadata.py`。

现有 Direct DFT 是 O(BN²) 运算和 O(N²) 表存储，不能通过提高 128 上限获得可扩展 FFT。
从 GPU/TLE 路线切换到基础 Triton 不要求第一版管理 L1/L0；以向量 butterfly 和 GM
中间结果跑通，再用测量决定是否需要显式 UB 和 TLE-DSA。

## 本轮验证方法

使用环境指导中已存在的 `flagsparse` 容器，镜像为
`flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base`。
只复用其已安装工具链，不重装共享环境。源码在本地主机提交，通过增量 Git bundle
快进同步云端同名 worktree，再用已提交的 archive 更新容器源码镜像。
容器只编译和运行，不编辑源码。原生库重新构建，不仅复用旧 `.so`。

结果集中保存在主机 `results/20260916_201307_ascend_reassessment/`：

- `validation.log`：重建、原 smoke、扩展矩阵首次失败记录。
- `validation_fixed.log`：NPU JIT launch 修复后的扩展矩阵。
- `stockham.log`：纯 Triton 多阶段探针。
- `environment.log`、`commits.txt`：工具链、动态库和版本信息。

扩展矩阵由 `tools/ascend_plain_validation.py` 提供：长度
1/2/3/7/8/16/31/32/64/127/128，batch 1/3/65/257，默认及用户 stream，
随机/脉冲/常量/单频，正逆方向，每项连续执行三次；检查输出、输入不变和输出保护区。
逆变换独立对照 `N * numpy.ifft(input)`，不只做 round-trip。
逐点容差 rtol=atol=3e-5，relative L2≤3e-5；NaN 不会被接受。
另验证大尺寸、real、FP64、多维、stride 和 padded distance 返回不支持且清空 handle。

`tools/ascend_stockham_probe.py` 用 runtime span 和常量 N 的 radix-2 Stockham：

1. 对每个 batch 的 k∈[0,N/2)，读 src[k] 和 src[k+N/2]。
2. span 从 1 倍增至 N/2，j=k%span，乘 exp(sign*2πij/(2span))。
3. 写 dst[2k-j] 与 dst[2k-j+span]，阶段间交替两块 GM 缓冲区。
4. 同一用户 stream 串行提交阶段，CPU 生成 FP32 twiddle 表。

探针长度 16/128/256/1024/4096/65536，batch 1/3，四类输入和两个方向。
block=256；最大逻辑 grid=384，用于验证超出物理核心数量的覆盖。
逐点容差包含 `3e-5 * max_batch_input_L2` 的绝对项，并要求 relative L2<3e-5。
日志的 compile_and_run_seconds 仅帮助排查编译耗时，**不是性能 benchmark**。

本轮新增结果：Python Stockham 的 96 项检查全部通过，最大 relative L2 为
`1.528e-7`。65536 点、batch=3、grid=384 已运行；这是普通 Triton 多阶段路线的
可行性证据，C API 的支持上限仍保持 128。

扩展 C API 首轮在 `N=1, batch=65` 发现输出 NaN 未被覆盖。根因是
`NpuBackend::launch_kernel` 无条件截断 blockNum，而安装的 FlagTree launcher
只有在 `enable_auto_blockify`（或其环境 fallback）开启时才截断。
默认 `TRITON_ALL_BLOCKS_PARALLEL=false`，编译器没有补齐逻辑 grid 的循环。
依赖修复 commit `4a11f4e` 保留 metadata 的可空布尔开关，按官方 launcher 的优先级
选择是否截断，并区分 AIV 与 AIC 物理 block 数。此次发现说明原先“任意正 batch”
只是代码接受范围，旧 smoke 的 batch=2/3 不足以证明该能力。

修复后重新构建并完成整个矩阵：**704 项数值检查（共 2112 次 execute）及 9 项
负向范围检查全部通过**，涵盖上述全部长度、batch 和两类 stream。
修复前日志保留，不以修复后的通过记录掩盖原有缺陷。版本、源码校验和及最终库哈希
见 `commits.txt`、`source_verification.log`、`launch_evidence.log`；
`summary.json` 汇总误差，`reproduce.sh` 提供容器中的构建和 C API 重跑命令。
本轮没有验收 auto-blockify 显式开启模式、非零 workspace、in-place、性能或 910C。

## 分阶段交付及剩余工程量

按一位熟悉本项目、可使用现有机器的工程师估计；人日是剩余工作预算，不含本轮已做评估。

| 阶段 | 工作与退出条件 | 预算 |
|---|---|---:|
| P0 后端可靠性 | 原生 C++ 冷启动；同一 Stockham kernel 经 libtriton_jit 运行；大 grid、用户 stream、workspace 生命周期；隔离不支持的 tuned plan | 2–3 人日 |
| P1 可扩展 FFT | Stockham emitter/metadata、编译与执行节点、twiddle 和 scratch 生命周期、planner 分流；C API 正逆/batch/N≤65536 二次幂全部通过 | 3–5 人日 |
| P2 可验收交付 | CPU reference 接入，负向/重复/边界测试，冷暖缓存、内存与析构检查，可复现构建说明、矩阵与计时 | 3–5 人日 |
| **正确性版本合计** | **已有基线上完成可扩展 C API** | **8–13 人日** |
| P3 首轮性能优化 | 固定形状集；融合相邻 stage、小 FFT 单 kernel、batch packing，再决定是否引入 UB/TLE | 另计 5–10 人日 |

8–13 人日约为单人 2–3 周。若新 kernel 暴露 NPU JIT ABI/workspace 问题，另留
3–5 人日调查修复；上游编译器阻塞不包含在此预算。不能据 Python 成功省略 P0。
不再重复旧计划中完整 adaptor/工具链从零接入的 5–8 人日。

基础 C API 完成后，按以下顺序扩展；这些预算互不计入上面的 8–13 人日，
也不代表达到 NVIDIA 或厂商 FFT 的性能水平：

| 后续里程碑 | 路线与验收 | 初步预算 |
|---|---|---:|
| FP32 R2C/C2R | 先完整复数路径加 pack/unpack、Hermitian 重建，验证 DC/Nyquist 和归一化；后续再做 half-size 优化 | 3–5 人日 |
| 连续 2D C2C | 复用 1D Stockham 加普通 Triton tiled transpose；先验收二次幂轴，包含非方阵和 batch | 3–5 人日 |
| 任意长度 1D | 先以 Bluestein 复用二次幂卷积，再按实测选择 mixed-radix/Rader；检查大素数内存峰值 | 5–8 人日 |
| 3D 与复杂布局 | 轴向分解、pack/unpack 或 stride 索引，逐项声明 in-place/alias 限制，回归 2D | 4–7 人日 |
| FP64 / graph / 多设备 | 先做型号与工具链能力专项，不与 FP32 默认路线捆绑 | 首次可行性评估 2–3 人日，再估实现 |

首轮性能优化应在扩展全功能前完成一组代表性形状，避免把低效的多次 GM 往返扩散到
所有算法。TLE 验证按操作逐项推进：先 UB alloc/copy/local_ptr 与必要同步，再融合
多个蝶形 stage；FFT 向量路径没有充分理由默认进入 L0A/L0B/L0C 或矩阵单元。
每次优化都与纯 Triton 基线比较精度、编译时间、设备时间和临时内存，失败时保持回退。

P1 的具体改动点：

- 在 `python/flagfft_codegen` 新增独立 Stockham stage emitter；显式登记 runtime span
  为整数参数，不能让当前 `_arg_signature` 默认误判成指针。探针中的 `batch` 与正式
  emitter 的 `nbatch` 参数协议需要统一。
- 在 kernel key/registry/metadata 中登记新 kernel；保留 N、dtype、方向、后端/设备键，
  明确 span 为运行时参数还是编译特化。第一版可以复用探针的 runtime span 降低编译量。
- 复用已有 StockhamPlanNode，补 `compile_raw_node`、raw-supported 检查和执行节点。
  不先建设通用 stage IR。两块 scratch 归 plan 所有，最后阶段直接写用户输出。
- 在 NPU planner 内选择 Direct DFT/Stockham；C API gate 仅开放经过验证的白名单。
  不能仅删除 `N<=128` 条件，使 NVIDIA leaf/tuned plan 意外可达。
- 正反方向共享算法结构，分别管理 twiddle；明确同一 plan 跨 stream 并发不属于首版
  验收，避免共享 scratch 被异步复用。

## 风险与必须补的证据

1. **Python 驱动与 C++ JIT 是两条 launch 路径。** 当前 ctypes 测试先 import torch_npu，
   不能完全证明原生 C++ 进程独立初始化。Stockham 探针也不证明新参数可经 C++ 打包。
2. **workspace 路径尚需专项验证。** NPU backend 按逻辑 block 数分配 workspace，
   后面限制物理 blockNum；应按所用编译器 metadata 核对计量单位。释放使用
   aclrtLaunchCallback，但仓库搜索未见对应 report 处理线程；这是待审计风险，
   不能从 workspace=0 的成功 kernel 推断非零 workspace 无泄漏。
3. **硬件特征有占位值。** adaptor 的 max_dynamic_smem_bytes=192KiB 不是真实 Ascend
   UB capacity 查询；架构回退为 Ascend910B4。优化前应去除对这些占位值的错误依赖。
4. **tuned plan 入口需要后端白名单。** lookup_or_build_root 先查 DB 再进 NPU
   auto_candidates；应过滤节点类型或禁用 NPU tuned DB，不能只依赖 auto_candidates。
   当前查询有 device_arch 和多个 fingerprint，但仍需审计 backend/toolchain 完整隔离。
5. **范围扩展不宜捆绑。** FP64、R2C/C2R、混合基数/大素数、2D/3D、复杂 stride、graph
   分别立项。先完成 power-of-two，再优先 R2C/C2R 和 2D，最后按业务矩阵决定
   Bluestein/Rader、3D、FP64；不能沿用旧项目的全功能性能承诺。

P2 验收至少覆盖所有 2^k（4≤k≤16），batch 1/3/65/257，正逆及四类信号；大尺寸
增加多 tile、尾部 mask、重复 plan 创建销毁。使用 CPU FP64 FFT 对照，逐点容差和
relative L2 双重检查，不只测试 round-trip。另测 N≤128 的非二次幂回归。
性能独立报告首次编译、预热后设备 event 时间和 host 提交时间，最低 warmup=3、iters=30。
当前尚未测得性能，因此不承诺对厂商库或 NVIDIA 的倍数。

## 公开资料核对

2026-09-16 查阅的 [FlagTree Ascend 用户手册](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend)
列出 Triton 3.5、910B/910C 和对应 CANN 9.0 镜像；本轮沿用已运行的 910B 镜像。
文档中的可安装版本不等于本机安装版本，最终以 environment.log 为准。
具体 kernel 能否运行以本轮实机记录为准，不再借用 HCU 的 lowering 问题推断 Ascend。
