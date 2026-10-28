# 修复版编译器上的 Ascend four-step FFT 实验

最新进展：2026-10-28 已将实际测试矩阵的 10 个 C2C case 接入 C API，
两轮逐例均超过 0.815×。具体范围和验收结果见文末“正式矩阵的 C API 接入”。

本实验针对原先因结构化置换 lowering 而停下的重写路线，验收条件是
FP32 C2C forward、N=2048、batch=2048，完整自然序 FFT 的设备时间
低于 0.15 µs/变换。它是独立 kernel 实验，不修改产品默认计划器，
也不代表完整 1D/2D/3D 算子矩阵已达到 SiP 性能门槛。

实现和运行入口是 [ascend_fourstep_probe.py](../tools/ascend_fourstep_probe.py)。

2026-10-27 的固定工具链复测已满足此微测门槛：三个随机种子的运行
中位数分别为 0.139040、0.138662、0.137104 µs/变换，三者中位数
0.138662，距 0.15 上限有 7.56% 余量。每图 256 次调用为 0.137154，
与 64 次捕获的结果相差 1.09%。9 组数值及清空后重放检查全部通过，
最大 rel_l2 为 1.72e-7 以下。逻辑 grid=2048，实际 launch blocks=40。

验证的算法源码提交为 `ba431c3`，最终工具链为此前修复的后续快照：
FlagTree compiler.py 匹配 `e59626316`，FLIR checkout 为 `0585f69`，
实际二进制、Python 后端和 bitcode 均按 SHA256 归档。性能结论只覆盖
本节的 FP32 forward 微测。

## 最终算法

把输入看成 `x[n1, n2]`，其中 N1=32、N2=64：

1. 在 N1 轴做 32 点 FFT，64 个连续列作为向量 lane。
2. 乘以跨段 twiddle `exp(-2πi*k1*n2/2048)`，在 UB 内转置。
3. 在 N2 轴做 64 点 FFT，32 个连续列作为向量 lane。
4. 交织实虚部，以自然频率顺序连续写回。

两段 FFT、twiddle 和转置融合在一个 kernel；中间结果不写回 GM。
每段使用 radix-4 DIF 和必要的 radix-2 尾级，并在各级连续打包输出，
直接得到自然序。两段都保留至少 32 个连续元素的行宽，避免把小行宽
切片或高维 split 交给有问题的对齐规划。

使用的设备操作为 Triton 算术、`reshape/split/join/trans` 及 Ascend
`extract_slice/insert_slice`。没有新增手写 FFT AIV/C++ 设备 kernel。
通用 split/join 设备原语由此前的 FlagTree/FLIR 修复版自动链接。

## 复现

先激活此前的隔离 FlagTree/FLIR 修复版，而非系统安装的 Triton。
CANN 为系统原版 9.0.0，设备为 Ascend 910B4。

```bash
python3 tools/ascend_fourstep_probe.py \
  --n 2048 --batch 2048 --factor 32 --fused-fourstep \
  --dispatch persistent --steps 64 --rounds 7 \
  --output /absolute/results/path/fft2048.json
```

`--dispatch persistent` 同时设置 `enable_auto_blockify=True` 和
`TRITON_ALL_BLOCKS_PARALLEL=1`，逻辑 grid 为 2048，实际 launch blocks
为 40。当前编译器的 SIMD 编译阶段和启动器读取不同的开关，二者必须
保持一致；实验入口负责同时设置，显式 CLI 选择覆盖环境默认值。
`TRITON_ALL_BLOCKS_PARALLEL=1` 在此工具链中表示启用自动映射，不能
按变量名字将它解释为启动 2048 个物理 block。结果 JSON 保存实际
launch 数、编译选项、源码 SHA256、误差和每轮计时。

计时使用已编译启动器。先检查整批输出，再捕获图；将输出清空后重放
并重新检查。每图捕获 64 次完整 FFT，event 时间除以实际调用数；
另用 256 次捕获检查收敛。twiddle 建表、编译和 CPU 参考校验不计入
设备执行时间。

完整数据、命令、编译产物、物理 IR 和结论放在外围工作区
`results/20261027_222426_fourstep_fixed/README.md`。

## 与 SiP 的完整 FFT 执行对比

同条件 N=2048、batch=2048 的后续配对测量：普通调用的设备耗时为
four-step **283.516 µs/整批**、SiP **110.354 µs/整批**，加速比
`SiP / four-step = 0.3892×`，新实现仍慢 **2.569 倍**。包含主机提交和
最终同步的 wall-time 比值为 0.4002×；每图 64 次完整 FFT 的设备比值
为 0.3832×。三个随机种子各 7 轮，全部输出及图重放校验通过。

两边均为设备驻留输入输出的完整 FFT 执行，不计 plan/JIT/建表及数据
传输。它仍不是产品 API 集成后的端到端测量。0.15 µs/变换的微测达标
不等于 SiP 性能达标。本轮结果归档于外围工作区
`results/20261027_232944_fourstep_sip/README.md`。


## transport P0 后的优化筛选（2026-10-28）

在 FlagTree `b25bd0e8c5` / FLIR `f9cf7cade1` 的固定 wheel 上，继续按
batch 打包、lane-local codelet、紧凑 twiddle、Vector/Cube 混合等方向
做了筛选。当前有效候选为 `ascend_packed_fft.py` 中每个逻辑任务打包
两次 FFT，并关闭 multibuffer；没有改变产品计划器默认值。

同条件 N=2048、batch=2048，三 seed 各 7 轮的普通调用设备时间中位数
由 283.398 降为 **267.492 µs/整批**（降低 5.61%），Graph 由 281.229
降为 **265.010 µs/整批**（降低 5.77%）。候选配对 SiP 为普通调用
109.484、Graph 107.381 µs，加速比分别 **0.4093× / 0.4052×**；仍未
达到 SiP 性能目标。主机提交加最终同步的普通调用比值为 0.4225×。
三个 seed 的数值和图重放检查通过，最大 rel_l2 小于 1.73e-7。

四路打包被 192 KiB UB 上限拒绝；紧凑 twiddle 的广播 lowering 明显
变慢；大规模 codelet 展开遇到编译时间上限。补偿 FP16 矩阵乘法可在
随机输入上保持约 1.8e-7 的 rel_l2，但双路 Cube 完整 FFT 仍约
0.62 ms。FP32 IEEE、单轴混合、16 点 Cube 叶子、ND/NZ 转换放到
Vector 以及 CV 自动平衡均未超过 Vector 打包候选。

msprof 的 Vector 基线活动率约 84%，Cube 原型 MAC 活动率约 8%，
同轮 SiP MAC 约 51%。这些指标可重叠，不能当作算术峰值利用率，
也不能据此复原 SiP 的算法。后续重点应是矩阵叶子的输入准备、物理
布局转换与输出接续，先证明含准备开销的叶子优于 Vector，再接回
four-step；当前证据不支持只增加 `tl.dot` 或继续扩大 batch tile。

完整筛选、失败原因、复测样本及计数器归档于外围工作区
`results/20261028_001023_fft_local_codegen/README.md`。


## 正式矩阵的 C API 接入（2026-10-28）

从 `conf/test_matrix.yaml` 固定选取以下 FP32 C2C out-of-place case，
每点包含 forward/inverse，共 10 例：1024/2048、batch=1/64，以及
16384、batch=1。选集在调优完成前冻结；没有删除第一次未达标的
16384 正反向。它是原矩阵的子集验收，不能代表 30 个算子完整矩阵。

- 1024/2048 使用单 kernel 的 32×(N/32) 结构化 FFT。
- 16384 使用两个 tile=8 的轴向 kernel，在首段融合 FFT、跨段 twiddle
  和转置写回。复用现有 `CompiledRawFourStepFusedNode` 的流和内存管理。
- 私有 twiddle 表按 `[k1,j2]` 的实虚平面保存，使每个 tile 连续加载。
  接入时复用旧交织转置表使首段明显变慢；修正后端到端由约 87 µs
  降至约 22 µs（同轮筛选），正式重新验收保留全部 case。
- 在 Ascend 910B、连续 FP32 C2C、上述已测长度和 batch 范围内默认启用。
  `FLAGFFT_NPU_STRUCTURED_FFT=0` 可回到原执行路径。实数、其他架构和
  未测尺寸不启用此选择。kernel 和表 ABI 通过独立 cache key 区分。

统一 `tools/run_tests.py` 经公开 C API 完成 10/10 NumPy accuracy、
10/10 SiP platform accuracy；最大 FlagFFT rel_l2 为 1.4054e-7 以下。
首轮 warmup=20、iters=100，最低 speedup=1.0464×，选集 GM=1.3823×。
独立复测 warmup=50、iters=500，最低 1.1335×，选集 GM=1.4120×；
两轮 20 条记录每条都超过 0.815×。

| N | batch | 复测 forward | 复测 inverse |
|---:|---:|---:|---:|
| 1024 | 1 | 2.2982× | 1.1416× |
| 1024 | 64 | 1.2788× | 1.2783× |
| 2048 | 1 | 1.2871× | 1.4033× |
| 2048 | 64 | 1.1335× | 1.3021× |
| 16384 | 1 | 1.4721× | 1.8715× |

计时为统一 runner 的完整 FFT C API 执行，包含内部所有 kernel 和
中间数据流；排除 plan/JIT/建表和 CPU/GPU 传输。两轮均运行在同卡，
配对 SiP 时间有波动，因此完整逐轮结果保留，不能仅选最高比值。

构建显式 `FLAGFFT_NPU_ENABLE_SIP_EXECUTION=OFF`，执行环境为
`FLAGFFT_NPU_EXECUTION_BACKEND=triton`；最终验证未设置算法开关，
证明默认路径生效。`libflagfft.so` 无 SiP FFT 未定义入口符号，SiP
仅作为验收参考。工具链仍为 P0 固定 wheel，系统 CANN 未改动。

新增生成体 CPU 数学/边界检查 8 项通过。额外运行的旧 codegen 测试
有 101 项通过、16 项失败；干净基线 `42568ea` 复现完全相同的 16 个
失败项，没有新增失败。详见结果目录中的基线对照记录。

算法源码验收 commit 为 `58690065`，完整数据、严格逐例 gate、
生成源文件和设备二进制放在外围工作区：
`results/20261028_164236_ascend_matrix_fourstep/README.md`。
