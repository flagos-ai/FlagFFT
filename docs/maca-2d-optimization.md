# MACA 2D FP32 优化与验收（2026-09-22）

目标口径：`speedup = mcFFT median / FlagFFT median`，目标至少 0.8。
本轮对象为 2D C2C、R2C、C2R，batch=1。代码基线为 dev `12cc497`。
FP64、batch>1、3D 和其他后端不扩大默认策略。

## 定位结果

1. 直接复用 1D direct exchange 可以改善 FFT leaf，但不能消除二维转置开销。
   2048² 的原转置约 175 µs/次，而两个 direct leaf 各约 88 µs。
2. 小 C2C 的 MACA graph replay 有明显成本：64² 同卡充分预热后，关闭 graph
   约 18.176 µs，开启约 32.256 µs。因此只在新 2D 作用域默认关闭 graph。
3. MACA 上把 complex64 当作一个 uint64 payload 做转置，避免分别读写实部/虚部。
   2048² 微基准：原转置 182.016 µs，双标量 `tl.trans` 411.520 µs，
   uint64 tile32 转置 65.152 µs。位级比较通过，不进行浮点数值转换。
4. 实数路径仍需边界融合：复用现有 R2C/C2R leaf，直接产生/消费紧凑半谱，
   去掉独立 expand/half-pack 或 Hermitian-expand/real-pack，并减少临时存储。
   这是减少中间读写，不是宣称 FFT 蝶形运算量已经减半。
5. 大行数矩阵中的短实数行需要打包：46189×48 每 block 一行时为
   0.752×/0.784×（R2C/C2R）；四行一组为 0.970×/0.981×，正确性通过。

没有采用的方向：2048 点跨行 strided FFT（约 0.05–0.06×）；对实/虚部分别
使用寄存器转置；全局扩大 batch pack。负实验保留为证据，不作为默认实现。

## 默认决策

原生编译器识别 MACA、FP32、rank=2、batch=1、非退化双轴、连续布局请求，
通过独立作用域把 2D 策略传到行列子 kernel，结束/异常时恢复。

| 条件 | 行为 |
|---|---|
| 符合上述 2D 作用域 | direct_all、LSB、VEC_IO=0、MAX_WARPS=8；four-step pack 仍用已有限幅规则 |
| 需要独立 complex64 转置 | uint64 payload、32×32 tile、4 warps |
| C2C graph | 默认关闭；不改变其他后端/维度的默认值 |
| real、n0>256、n1 为偶数，行 plan 为 leaf 或两个 leaf 的 four-step | 实数边界行融合 + 两次紧凑转置 |
| 上述实数边界 leaf，长度 16–128 | 每 block 四行；metadata 与源码由同一个函数推导 |
| n0≤256 real | 保留已有 RC 路径，短复数 leaf 不继承四行打包 |
| 奇数行宽、非 leaf/leaf-pair 行 plan | 保留原实数路径 |
| FP64、batch>1、退化单轴、3D、其他后端 | 不进入本轮 2D 策略 |

不默认开启 packed-real 半长度算法，也不扩大 Bluestein 的 batch fusion 范围。

`FLAGFFT_MACA_2D_SINGLE=0` 可整体回退本策略。单项调试覆盖仍可使用
`FLAGFFT_MACA_2D_GRAPH=0/1`、`FLAGFFT_MACA_2D_TRANSPOSE=legacy/packed`、
`FLAGFFT_MACA_2D_REAL_ROWS=0/1` 及已有 exchange/pack 开关。
正式使用不需要设置这些变量。

2D policy 与 1D policy 分别进入进程内 kernel key 和持久化 codegen 目录；
单项覆盖的 A/B 仍须使用不同 executable/cache 目录，不在同一进程切换环境变量。

## 验收结果

最终提交 `ba114a5`，无 opt-in 环境变量，`--scales all --warmup 5 --iters 30`，
独立构建目录 `maca-2d-qualified-ba114a5`。

`20260922_173600_maca_2d_qualified_matrix`：3 个 op，60 accuracy / 20 performance，
accuracy 与平台库交叉正确性 3/3 通过。

| op | cases | 几何平均 | 最低 | 最高 |
|---|---:|---:|---:|---:|
| 2d_c2c | 10 | 1.030 | 0.833 | 1.898 |
| 2d_c2r | 5 | 1.091 | 0.875 | 1.862 |
| 2d_r2c | 5 | 1.079 | 0.841 | 1.965 |
| 合计 | 20 | 1.057 | 0.833 | 1.965 |

`32×46189` 的 RC 快路径保护经 `20260922_173600_maca_2d_qualified_rc` 三轮独立进程
复测（200 warmup / 100 iterations）：C2C 1.900–1.911 / 1.882–1.890，
R2C 2.003–2.014，C2R 1.907–1.916。

作用域回归：关闭新 2D 策略后抽查 172 份 kernel 源码与 launch metadata，与 dev
完全一致，覆盖 FP32/FP64、1D 策略开关、实数与跨步 leaf。
C API 边界测试 8/8 通过：512×64/128/1024 异位与原位、DC/Nyquist、往返，
以及 FP64、batch=2、单轴、奇数宽度回退。

## 过程记录

- `20260922_161100_maca_2d_profile`：graph A/B 与逐 kernel 归因。
- `20260922_161400_maca_transpose_micro`：位级转置正确性和计时。
- `20260922_161700_maca_transpose_tiles`：tile16/64 对照。
- `20260922_161700_maca_2d_packed_screen`：C2C/real 初筛；其中 GPU5 有外部任务，不能单独作为最终性能证据。
- `20260922_163200_maca_2d_real_rows`：4 个 real case 的 NumPy/mcFFT 正确性与性能。
- `20260922_164400_maca_2d_rowpack4`：短行 P4 对照，2 个 case 正确性通过。
- `20260922_164100_maca_2d_default_full`：未加入短行 P4 的默认策略完整矩阵、三档幅度。
- `20260922_170000_maca_2d_repeat`：回退/候选独立进程对照。
- `20260922_171600_maca_2d_rc_guard`：RC 行子图策略保留对照。
- `20260922_173600_maca_2d_qualified_matrix`、`20260922_173600_maca_2d_qualified_rc`：最终验收。

初筛目录仅作归因证据，不作为验收数值。

复现矩阵：

```sh
python3 tools/run_tests.py --ops 2d_c2c,2d_r2c,2d_c2r \
  --gpus <空闲物理卡列表> --scales all --warmup 5 --iters 30 \
  --build-dir <本提交独立构建目录> --output-dir <工作区results下的新目录>
```

独立进程复测入口为 `tools/maca_2d_repeat.sh SOURCE BUILD OUTPUT PHYSICAL_GPU`，
使用 200 warmup/100 iterations，保留全部有序样本；回退一次、候选三轮。
运行前须确认卡未被其他任务占用，锁只能约束遵循同一约定的进程。
