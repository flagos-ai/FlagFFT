# MACA 1D single 长尾实验（2026-09-22）

本轮以 dev `12cc497` 为基线；新增运行时策略全部 opt-in，未合入 dev/main，
也未改变既有 MACA 1D single 默认策略。实验汇总在 `perf/maca-tail-integration`。
统一 MACA 构建和最后设备核对冻结在 `136bf40`，随后仅整理报告。

## 测量口径与证据

- C550、FP32/FP64、1D、batch=1；每次 **5 warmup / 30 iterations**。
- 速度比为 mcFFT/FlagFFT；0.8 对应 FlagFFT 耗时不超过 mcFFT 的 1.25 倍。
- 每组 A/B 使用同一物理 GPU、独立 executable/JIT cache，并持有该卡的 flock。
  分组使用 GPU0/1/2/4/5/7，未占用其他任务使用的 GPU3/6。
- 不剔除慢轮、不将 runner 的 `performance=Passed` 当成达到 0.8。
- 数值初筛对照 NumPy 与 mcFFT。扩展 C API 验证使用矩阵幅值
  `2^-20, 1, 2^20`，含输出 guard；C API 本身不测性能。
- 源码、原始计时、实际 plan、binary hash 和环境记录保存在根目录 results。
  性能复测目录中的 accuracy=NotRun 是有意只复测计时，不能代替独立正确性结果。

主要证据目录（均位于 workspace 根 results）：

| 内容 | 目录 |
|---|---|
| real N23 基线与 Kahan 融合 | `20260922_145640_maca_tail_baseline_real23` / `20260922_150630_maca_tail_real23_fused` |
| real FP64 tree | `20260922_152217_maca_tail_real23_tree` |
| real 独立进程复测 | `20260922_153130_maca_tail_real23_repeat_baseline` / `..._candidate` |
| real 同址/非同址各 12 项 | `20260922_152950_maca_tail_real23_capi_inplace` / `20260922_152951_maca_tail_real23_capi_outplace` |
| FP64 CT 分解/P4/warp 对照 | `20260922_150900_maca_tail_fp64_plans` / `20260922_151511_maca_tail_fp64_p4` / `20260922_152452_maca_tail_fp64_p4_w4` |
| FP64 CT 独立进程复测 | `20260922_153131_maca_tail_fp64_repeat_w8` / `..._w4` |
| FP64 CT 六项扩展数值验证 | `20260922_153240_maca_tail_fp64_capi_w4` |
| prime 长度对照 | `20260922_150903_maca_tail_prime1009_plans` / `20260922_150904_maca_tail_prime997_plans` |
| mixed 分解对照 | `20260922_150901_maca_tail_mixed663000_plans` / `20260922_150902_maca_tail_mixed328050_plans` |
| mixed 新交换：默认分解/组合分解 | `20260922_151200_maca_tail_mixed_exchange` / `20260922_152730_maca_tail_mixed_combined328050` / `20260922_152731_maca_tail_mixed_combined663000` |
| 统一版本构建 | `20260922_154100_maca_tail_integration_build` |
| 统一版本最终核对 | `20260922_154245_maca_tail_integrated_real` / `20260922_154246_maca_tail_integrated_ct` / `20260922_154247_maca_tail_integrated_prime` |

## 已证明有收益的方向

### 小 real：融合边界，再缩短 FP64 累加依赖

N23 基线 R2C/C2R 约 33/32.8 us，D2Z/Z2D 约 72–73 us。
边界融合后 FP32 为 20.992/20.480 us；FP64 Kahan 仍为 42.240/50.432 us。
FP64 tree 将独立乘积做平衡加法树，并利用位级对称 DFT 表改为连续访问；
初筛 D2Z/Z2D 为 30.976/29.952 us，独立进程复测为 28.160/29.952 us。
四 API 初筛及复测速度比均超过 0.8，最低约 0.940。

tree 保持 FP64 运算但不补偿求和，不能声称与 Kahan 位级一致。
CPU 覆盖抵消、端点、随机/DC/脉冲等；设备四 API 三幅值同址及非同址各
12/12 通过。FP64 tree 产物 shared=0、无 private spill；这是资源证据，
不代表所有尺寸均有相同性能。设备证据仅 N23，不能外推到门控允许的全部长度。

```sh
export FLAGFFT_MACA_REAL_DIRECT_DFT=1
export FLAGFFT_MACA_REAL_DFT_REDUCTION=tree
```

第一个开关仅用于 MACA rank1/batch1 已选中的 DirectDFT 计划、长度 1–128，
不强制 planner 选择 DirectDFT。第二个开关仅改变 MACA FP64、长度 <=32；
FP32 不变。完整同址保护会增加输入副本，性能数字为非同址，未测部分重叠。
Kahan/tree 的 module/cache basename 共用，必须隔离进程和 cache，不能原地切环境变量。

### FP64 CT 1M：平衡分解、受限 P4、4 warps 缺一不可

同 GPU2 对照：512×2048 为 727.808/552.192 us；1024×1024、P2 为
493.568/489.216 us。注意第一项显式设置 INNER_PACK=4（实际 row P4/col P1），
不是不带参数的生产默认 row P2，不应混称 dev 的原样基线。

平衡 P4、8 warps 为 307.968/343.808 us，复测 325.888/349.184 us，
仍未达标。平衡 P4、4 warps 首轮 282.112/286.720 us（0.839/0.821），
复测 290.560/291.840 us（0.822/0.813），双向均达到 0.8。
扩展六项 C API 数值通过，rel-L2 约 5.94e-16。

```sh
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_PACKED_REAL=0
export FLAGFFT_MACA_TAIL_PLAN=ct1024x1024
export FLAGFFT_MACA_FP64_REGISTER_PACK=1 FLAGFFT_MACA_INNER_PACK=4
export FLAGFFT_MACA_MAX_WARPS=4
```

8-warps 产物 shared=64 KiB、静态 barrier=7、mtreg=96/104；4-warps 改为
32 KiB、11、162/170，两版 private=0。同步数更少并不必然更快；shared 与
线程/寄存器布局的变化有利于解释结果，但没有计数器证明各因素占比。
新资源豁免只允许受限的 FP64 pow2 512/1024 leaf、至多 P4，并保留 padding
上限；不能由此开放 P8。资源开关不是完整的按根计划自动策略，不能全局常开
或外推到其他 leaf、batch、2D/3D。正式选型还需将实测组合固化到 scoped policy。

### prime：卷积更短不一定更快

997 FP64：BS2000（137.216/137.728 us）对照 BS2048，后者正向 77.312 us、
约 0.874×，双向初筛几何平均约 0.874×。两条均通过正确性，差异不是只减少
发射次数：BS2048 内部 power-of-two FFT 更适合现有交换路径。

1009 FP32：Rader1008 为 106.752/106.496 us；BS2048 为 78.080/78.080 us，
提升约 1.36 倍，但比值 0.790/0.807，不能说双向都达到 0.8。

```sh
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_PACKED_REAL=0
export FLAGFFT_MACA_TAIL_PLAN=bs2048
```

该显式候选只支持白名单尺寸/精度/1D single，`compare` 只用于 tuner。
不改变默认 planner。每个形状与配置用独立 cache；不能给完整矩阵统一强制此变量。

## 部分收益与负结果

- 328050 FP32：450×729 约 110 us；486×675 约 108–109 us，收益很小；
  405×810 双向 88.064 us，约 25% 提升，比值 0.686–0.692，仍未达标。
- 663000 FP32：375×1768 双向约 213–215 us，650×1020 和 780×850 都变慢。
  更平衡的因子并未消除大 radix 算术和 padding/访存成本，保留原分解。
- `FLAGFFT_MACA_MIXED_EXCHANGE=direct` 新原型在两尺寸均回退，保持关闭。
  328050 默认分解约 144–146 us；405×810 组合约 124–125 us（旧交换 88 us）；
  663000 默认分解约 305–307 us（旧交换 213–215 us）。功能通过不是性能有效。

mixed 新交换在源码层消除了中间整 leaf tensor，但 MACA lowering 对每个
digit gather 重复写入完整 joined bank，未复用 staging。405×810 forward
的静态 LLIR 对照如下（不是动态指令执行次数）：

| kernel | mtreg 旧→新 | barrier 旧→新 | shared store 旧→新 | shared load 旧→新 |
|---|---:|---:|---:|---:|
| row | 64→58 | 63→55 | 176→448 | 44→28 |
| col | 88→70 | 79→71 | 352→576 | 68→36 |

两版 shared 均 16 KiB，未发现 spill。下一步若继续 mixed，应解决 joined bank
重复 staging 和 padded radix 算术，而不是继续简单合并 gather 或调平衡因子。
本轮不追加新变量。

## 集成与验收边界

Python 回归：1546 passed、136 设备用例按默认开关 skipped、8 subtests passed；
独立 CPU planner 9/9 通过，MACA C++ 构建成功。数值与速度比门禁分开记录。
这不是完整 36-op 验收，更不是 2D/3D 或所有 1D 长度达标证明。

统一构建 `136bf40` 的最终设备结果如下：

| API / 长度 / 方向 | FlagFFT us | mcFFT us | 比值 |
|---|---:|---:|---:|
| R2C / 23 | 21.760 | 23.296 | 1.0706 |
| C2R / 23 | 12.032 | 13.056 | 1.0851 |
| D2Z / 23 | 29.184 | 28.928 | 0.9912 |
| Z2D / 23 | 30.464 | 29.184 | 0.9580 |
| Z2Z / 1048576 / forward | 280.064 | 236.288 | 0.8437 |
| Z2Z / 1048576 / inverse | 290.560 | 236.544 | 0.8141 |
| Z2Z / 997 / forward | 119.808 | 136.960 | 1.1432 |
| Z2Z / 997 / inverse | 72.704 | 64.512 | 0.8873 |

这八项按指定 5/30 的全样本中位数口径达到 0.8。统一版本另通过 24 项设备
C API 验证：N23 四 API 三幅值同址 12 项，1M/997 FP64 双向三幅值各六项。
上述统一测试均已结束。

**短预热的限制仍存在。** 997 FP64 forward 的首/后 15 个样本中位数为
FlagFFT 184.832/70.144 us、mcFFT 155.904/58.624 us；两库在测量中途同时
发生耗时阶跃。先前 BS2048 初筛也存在同类变化，因此不能把最终 1.143×
相较初筛 0.874× 的变化解释为代码收益，更不能称为稳态吞吐保证。
最终 C2R 的两库时延也都低于先前进程，但其轮内前后半稳定。
保留全部 30 个样本，不以尾部样本替换验收数字，未擅自增加 warmup。

按实际收益整理可吸收部分：小 real 融合/tree、FP64 1M 的受限资源/分解组合、
997 FP64 的 BS2048 候选；328050 的分解可保留为部分改善，1009 BS2048 仍有
正向长尾。mixed direct 仅保留负实验与测试证据，不进入默认路径。

每个分支候选最初使用冻结源码和独立二进制；存在 Python-only 改动复用原库的
情况，environment.txt 中 `FLAGFFT_BINARY_COMMIT` 与 SHA256 已记录，不能把源码
commit 当作二进制 commit。最终统一构建 `136bf40` 用于排除组合遗漏。

可以用 `tools/summarize_maca_tail_results.py <suite...>` 生成逐 case JSON。
脚本不修改原始结果，显式报告 `meets_0_8`；未知/缺失不算通过。
本轮完整逐 case 汇总在
[all_cases.json](../../results/20260922_155344_maca_tail_acceptance/all_cases.json)。
