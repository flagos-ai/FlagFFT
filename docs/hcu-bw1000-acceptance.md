# BW1000 优化与验收（2026-11-16）

目标：默认 30 算子、400 个性能 case 的逐 case 加权几何平均加速比至少 1.0x，
每个 case 至少 0.7x；同时要求 NumPy 和 hipFFT 参考准确性通过。算子等权几何平均
单独记录，不与逐 case 口径混用。

代码工作分支为 `codex/hcu-general-opt`，从 `codex/hcu-3d-09@62e2e015` 继续，
该基线包含 `dev@04f49734` 和此前 prepared native dispatch 改进。
原 `FlagFFT-hcu-3d-09` 工作树的未提交改动保持原样；其中 CT 原型复制到本分支后
单独提交和验证。libtriton_jit 固定为 `40faf64ca8833d12b38d496924af28f23aa8357d`。

## 环境和结果位置

按外围工作区 `docs/baai-environment-guide.md` 使用 SSH alias `baai-hg`，
容器为 `8488fdc25f1a`，两张 BW1000、gfx936、64-lane wavefront、64 KiB LDS。
使用独立 `/opt/dtk-26.04`、Torch 2.9、FlagTree 0.7 和 native execution policy；
保留容器原有 `/opt/dtk` 指向。每次运行前重新检查空闲卡。

本地归档在外围工作区 `results/20261116_152600_hcu_general_opt/`，
远端工作区为 `/root/flagfft-hcu-general-opt/`，输出在其独立 `results/` 中。
构建脚本、环境、bundle、逐次原始 JSON/log 和 CPU 验证日志均随本地结果归档。
禁止将远端 build 或 JIT 缓存同步回本机。

远端按候选冻结独立源码快照。`FlagFFT-hcu-general-opt@f93314df` 为初始
CT/paired-I/O，`FlagFFT-hcu-radix@35e1b82b` 为 radix 组合，
`FlagFFT-hcu-ct-fixed@d17251ad` 修正 CT grid，`FlagFFT-hcu-buffer@21205183`
为循环单缓冲；`FlagFFT-hcu-real@dcd67243` 为紧凑实数与短 wave，
`FlagFFT-hcu-dense-wave@edf3a1b4` 为 FP32 连续 wave FFT，
`FlagFFT-hcu-wide-wave@433ccf4e` 为 FP64 与显式 wave64 布局。
源码均先在本地提交，经 bundle 同步；远端只编译和运行。
最终验收必须使用同一个冻结的源码、依赖和构建版本。

## 验收入口

```bash
FLAGFFT_NUMPY_CAPTURE_FULL_BATCH=1 python tools/run_tests.py --combination full --gpus <idle-gpus> \
  --build-dir <matching-build> --output-dir <external-result-directory> \
  --warmup 10 --iters 100 --timeout 600 --color never
python tools/check_acceptance.py <external-result-directory>
```

Gate 同时检查矩阵 SHA-256、30 算子/400 case 覆盖、NumPy/platform 准确性、完整性、有效且通过
验证的参考时延、时延与 speedup 比值一致，以及 GM/最低值门槛。性能单独运行、
缺项、跳过、无效参考或部分矩阵均不能通过。输出另带算子等权 GM 供对照。
此外要求完整 batch 模式及每个准确性 case 的实际 `capture_batch` 与请求 batch
一致。默认 capture 会遍历全部输入，但为节省设备资源将它们分块并按较小 batch
创建计划；这不等同于验证 batch64 的计划路径。此前未设置该开关的筛选结果只作
分块数值证据，正式验收不复用这些 accuracy 结果。

矩阵外验证使用 `--matrix-file`。已有 discovery/holdout 配置已移植；另外冻结
`conf/test_matrix_hcu_large_holdout.yaml`，SHA-256 为
`fbba997a316a5b8bdeb380f41f644553fb3d7a39a236c49f4a3a84ce830de5d7`。
四个 1D Four-Step real 算子展开为 60 个 accuracy + 60 个 performance case，
包含矩阵外长度和 batch=2/7。该 holdout 不用于选择阈值；3D 项还覆盖非立方体、
轴比例变化、batch=5 和非二次幂回退。

## 基线与已完成测量

历史同环境 `04f49734` 全量：400/400 accuracy、400/400 platform accuracy、
400/400 performance。逐 case GM 为 1.119951x，算子等权 GM 为 1.100876x；
最低 0.202232x，共 86 case 低于 0.7x，因此整体**未通过新门槛**。
原记录为 `results/20261113_103000_hcu_dev_04f49734/`。

本轮重复 baseline 完成首 74 个 performance case 后主动停止，将卡转给新候选。
这 74 项 GM=1.0754x、min=0.7324x；这是部分矩阵，不能代替全量。
runner 已保存中断状态和全部完成项；停止后 Python 队列退出挂起，保存结果后
终止了该进程。正式全量将在候选默认策略固定后重新运行。

`FLAGFFT_PACKED_REAL=0/1` 同卡 A-B-B-A，warmup=10 / iters=100，
覆盖 5 个长度、两种精度、正反实数 API、batch=1/64，共 40 个配对样本。
四轮原始数据为 `20261116_152600_packed_real_abba/`。
其中 32 个长度至少 65536 的样本，FlagFFT 绝对时延改善 GM=1.562923x。
16384 的单 batch FP32 有回退，说明不能无条件启用半长路线。

以 524288 为例，四轮配对结果如下。改善比计算 A 两轮/B 两轮的 FlagFFT 时延
几何平均；candidate speedup 为 B 两轮相对 hipFFT 的几何平均。

| API | batch | FlagFFT 改善 | candidate speedup |
|---|---:|---:|---:|
| R2C FP32 | 1 | 1.2355x | 0.7262x |
| C2R FP32 | 1 | 1.3663x | 0.6739x |
| R2C FP64 | 1 | 2.0338x | 0.6427x |
| C2R FP64 | 1 | 1.9718x | 0.6264x |
| R2C FP32 | 64 | 1.6315x | 0.5668x |
| C2R FP32 | 64 | 1.7301x | 0.5714x |
| R2C FP64 | 64 | 2.7235x | 0.5506x |
| C2R FP64 | 64 | 2.6626x | 0.5495x |

这些是候选性能筛选，不能当作新默认版本验收。524288 的四个 real 算子、
FP32/FP64、batch=1/64、三幅度 `2^-20, 1, 2^20`：24/24 NumPy 和 24/24
hipFFT 参考准确性全部通过，记录在 `20261116_152600_packed_accuracy/`。
后续筛选分别保存原始结果，不用局部候选指标代替默认版本验收。

### 本轮其他筛选结论

- 两遍 3D CT：最初的设备 VMFault 来自 C++ 未读取新 kernel 的
  `grid_x_override`，实际启动数多了 8 倍。已修正并保存全部故障日志。
  修正后的两个 smoke accuracy case（各一个 `2^-20` 幅度）通过；16 个性能
  样本中，128³ 和长轴形状约为原实现的 0.34–0.37 倍性能，256³ 为
  0.64–0.76 倍，仅较小的一个几何有收益。拒绝作为通用默认，不增加赢家白名单。
- Four-Step 循环单缓冲：同卡 ABBA、18 个配对样本，FlagFFT 时延改善
  GM=1.068163x。524288 batch64 的 FP64 complex 候选约 0.715855x，
  real 仍约 0.582–0.602x。524288 batch64 的两个 real API、两精度、三幅度，
  12/12 NumPy 和 12/12 platform 校验通过。
- 将循环偏移从 unit 改为按 wave 宽度推导：18 个 ABBA 配对样本，改善
  GM=0.980198x；不启用该偏移为默认。四个实数设备校验均通过。
- 2D 紧凑行调度：2048²、batch1、两实数 API/两精度首轮时延改善
  1.251–1.423x，speedup 0.565–0.629x；4/4 双参考校验通过。
  这是顺序筛选，仍需 ABBA、batch4 与矩阵外验证。
- 融合半长叶子和 wave 偏移的组合：2D 低于仅紧凑行版本；3D 256³ FP32
  R2C 约 0.486x，明显回退。组合保持关闭，不能把组合结果单独归因于融合叶子。
  3D 立方体四个 real case 双参考通过，但性能尚不满足门槛。
- 短 wave 与连续/Four-Step wave FFT：设备筛选进行中。
  `--max-cases` 是对整个展开列表截断，**不是每算子截断**；带该参数的
  tiny/dense smoke 只能按 manifest 的实际 case 数报告。完整验证不使用它。
- 显式 wave64 布局：当前 FlagTree 0.7 HCU 后端拒绝 `gpu.set_layout`。
  四个实数 case 均在编译阶段失败，失败日志完整保留；`185d3590` 已撤回实现。
  后续 FP64 自动布局实验仍使用冻结的 `433ccf4e` 快照，显式布局开关保持 0；
  它生成的自动布局数学代码与撤回后的版本一致。

## 实现与候选边界

- `FLAGFFT_HCU_VECTOR_IO=1`：复用可移植的复数成对 load/store，覆盖 dense leaf
  和 Four-Step 行列。默认 0；没有增加尺寸白名单。HCU artifact identity 使用
  有界哈希，避免配置目录超出 255 字节，并隔离所有 HCU 环境开关。
- `FLAGFFT_HCU_FOURSTEP_RADIX_CAP=8|16`：对公开 1D、大于等于 65536 的支持分解
  请求，二次幂子叶的指数平衡分配到有界 radix；混合长度在已有候选中限制复合
  radix 的上界，保留不可再拆的受支持素数，以减少每线程寄存器负担。默认 auto
  保留原策略。请求上下文包含该值；固定 thread-local 候选不能绕过限制；
  素数卷积及其他后端保留原路线。
  混合长度扩展在原 acceptance 340200 FP64 使用 `[28,18]` / `[27,25]`
  大寄存器蝶形的证据下加入，不依赖 holdout 结果；尚待设备资格验证。
- `FLAGFFT_HCU_3D_CT=1`：二次幂 FP32 3D 的两遍 Cooley-Tukey 原型。两遍 tile
  均由 8192 complex 元素预算推导，batch 使用独立地址，不是逐 shape 路由表。
  默认关闭，FP64/不支持几何保留原实现。
- `FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER=1`、`FLAGFFT_HCU_LEAF_SINGLE_BUFFER=1`：
  仅在所有覆盖写的中间阶段均可一次覆盖完整 FFT 时复用缓冲；每次覆盖前有
  CTA barrier，多组阶段自动回退。leaf 还根据原共享内存占用和设备容量限定范围。
- `FLAGFFT_HCU_2D_REAL_ROWS=1`：复用已有紧凑行调度，限 HCU dense 2D、
  大于现有小 RC 范围的首轴、偶数末轴；无 shape winner 表。
- `FLAGFFT_HCU_FUSED_PACKED_REAL=1`：复用 LeafPackedR2C/C2R，按半长 FFT
  和重建共六个标量缓冲的 LDS 上界限定候选，未通过性能资格，不默认开启。
- `FLAGFFT_HCU_SMALL_WAVE=1`：FP32、16–64 的二次幂连续叶子在一个 wave 内
  填满 FFT 子组，保留既有距离与尾 batch 掩码；FP64/其他 I/O 方式回退。
- `FLAGFFT_HCU_DENSE_WAVE=1`：FP32、64–2048 的二次幂连续叶子和 Four-Step
  行列复用 wave 内 Cooley-Tukey。`FLAGFFT_HCU_DENSE_WAVE_FP64=1` 额外允许
  FP64，使用附加在末级表后的双精度自然顺序旋转因子；保留原表前缀以供回退，
  不使用 FP32 三角函数。输出重排按半个查询 LDS 容量分块。
  FP32 packed 边界另按重建需要的双倍活跃 payload 限制在 64 个标量/物理 lane。
  设备验证前均保持 opt-in。
- 半长度实数策略首先按 FFT 工作量、原/子计划结构、布局及资源界限建立适用域，
  再经过同卡重复性能、三幅度准确性和 holdout 决定默认行为；不增加逐 case winner 表。
- 下一轮已冻结的 half-length 策略为 `FLAGFFT_HCU_LARGE_PACKED_REAL=1`（默认 0）：
  HCU gfx936、公开 dense 1D、偶数长度、原计划为 FourStep、总工作量
  `batch*N >= 65536`，半长子计划必须为 Leaf 或两个 Leaf 的 FourStep。
  它保留 `FLAGFFT_PACKED_REAL=0` 的显式关闭及已有布局回退。
  该条件在 large holdout 执行前固定；holdout 不能用来重新选择阈值。

CPU 检查：FP64 wave 提交的综合回归为 932 passed、88 skipped、10 subtests passed；
显式布局新增后的相关回归为 349 passed。192 组默认 FP32 wave 生成源码与修改前
逐字一致。远端 CPU planner target：27 个 GTest 通过。以上不代替设备验证。
尚未宣告完成 1.0x / 0.7x 的最终验收。
