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

远端实验使用两个源码快照：`FlagFFT-hcu-general-opt@f93314df` 跑 CT/paired-I/O；
`FlagFFT-hcu-radix@35e1b82b` 跑 radix 与 paired-I/O 组合。两者均为本地已提交代码，
远端只编译和运行。最终验收必须使用同一个冻结的源码、依赖和构建版本。

## 验收入口

```bash
python tools/run_tests.py --combination full --gpus <idle-gpus> \
  --build-dir <matching-build> --output-dir <external-result-directory> \
  --warmup 10 --iters 100 --timeout 600 --color never
python tools/check_acceptance.py <external-result-directory>
```

Gate 同时检查 30 算子/400 case 覆盖、NumPy/platform 准确性、完整性、有效且通过
验证的参考时延、时延与 speedup 比值一致，以及 GM/最低值门槛。性能单独运行、
缺项、跳过、无效参考或部分矩阵均不能通过。输出另带算子等权 GM 供对照。

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

这些是候选性能筛选，不能当作新默认版本验收。NumPy/hipFFT 的三幅度准确性
验证与后续成对访存/radix/CT 筛选分别存放，待全部完成后记录最终结论。

## 实现与候选边界

- `FLAGFFT_HCU_VECTOR_IO=1`：复用可移植的复数成对 load/store，覆盖 dense leaf
  和 Four-Step 行列。默认 0；没有增加尺寸白名单。HCU artifact identity 使用
  有界哈希，避免配置目录超出 255 字节，并隔离所有 HCU 环境开关。
- `FLAGFFT_HCU_FOURSTEP_RADIX_CAP=8|16`：对公开 1D、大于等于 65536 的二次幂
  请求，将子叶的指数平衡分配到有界 radix，减少每线程寄存器负担。默认 auto
  保留原策略。请求上下文包含该值；固定 thread-local 候选不能绕过限制；
  素数卷积及其他后端保留原路线。
- `FLAGFFT_HCU_3D_CT=1`：二次幂 FP32 3D 的两遍 Cooley-Tukey 原型。两遍 tile
  均由 8192 complex 元素预算推导，batch 使用独立地址，不是逐 shape 路由表。
  默认关闭，FP64/不支持几何保留原实现。
- 半长度实数策略首先按 FFT 工作量、原/子计划结构、布局及资源界限建立适用域，
  再经过同卡重复性能、三幅度准确性和 holdout 决定默认行为；不增加逐 case winner 表。

当前 CPU 检查：574 passed、88 skipped（runner、codegen、HCU layout/CT/vector、gate）；
远端 CPU planner target：27 个 GTest 通过。它们不代替设备数值与性能验证。
尚未宣告完成 1.0x / 0.7x 的最终验收。
