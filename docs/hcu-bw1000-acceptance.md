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
- 冻结的 large holdout：`0b555921`、半长实数策略与循环单缓冲、cap=auto、
  scalar I/O；60/60 NumPy、60/60 hipFFT、60/60 performance 完成，均验证
  请求的完整 batch。候选 GM=0.863364x、min=0.534404x、13 项低于 0.7x。
  同版本关闭这两个候选的对照 GM=0.651567x、min=0.271795x、30 项低于门槛。
  60 个配对的 FlagFFT 绝对时延改善 GM=1.332332x，48 项改善，最差改善比
  0.590461x。因此已有整体收益，但泛化和最差项仍未达标，不能默认启用。
  原始输出为 `20261116_152600_policy_large_holdout{,_control}/`，配对结果
  为 `policy_holdout_comparison.json`；该数据不用于重选工作量阈值。
- 1D 联合筛选（`8e0d65cb`）：cap=16、paired I/O、半长实数与循环缓冲，
  340200 complex 和 524288/663000 real 共 12 个完整批量准确性 case 的双参考
  均通过。两种 batch 下的连续性能筛查发现 FP64 的 scalar I/O 比 paired 更稳：
  scalar+cap16 的 20 个实数样本最低 0.757x，而 FP64 complex 的最低仍为
  0.693x；不能把部分结果当作全量通过。FP32 paired 的大长度有收益，但
  限制 radix 的版本会使部分混合长度回退，因此不采用统一强制 cap。
- FP32 三维 wave store：较大预算原型完成 16/16 完整 batch 的 NumPy 与 hipFFT
  校验（两形状、batch1/4、两方向、两精度，其中 FP64 为原生数学控制）。
  64-scalar 预算的首轮单 batch FP32 性能为 256³ 0.754x、长轴 0.826x，
  同轮原生控制为 0.681x、0.540x。强制 FP64 走相同整体布局反而退化，
  所以后续按精度和支持的几何选择，不全局关闭 hybrid。仍待 ABBA、
  batch4 和矩阵外性能；这些数字不能代替完整验收。

## 实现与候选边界

- `FLAGFFT_HCU_VECTOR_IO=1`：复用可移植的复数成对 load/store，覆盖 dense leaf
  和 Four-Step 行列。默认 0；没有增加尺寸白名单。HCU artifact identity 使用
  有界哈希，避免配置目录超出 255 字节，并隔离所有 HCU 环境开关。
  `packed` 候选在相同 I/O 范围内将 FP32 复数按位装入一次 64 位访问，保持
  每个复数一个逻辑 lane；FP64 维持标量方式。它不改变蝶形或地址掩码，
  用于检查成对张量布局转换的成本，尚未取得设备性能资格。
  `fp32` 则在 FP32 保留成对张量、FP64 保留标量路径，作为另一种按精度选择。
  首次完整 batch 的 FP64 paired 对照在 1024 点子叶触发 67584 字节 LDS，
  超过设备的 65536 字节；原失败记录保留。显式 `1` 模式现在另预留一个
  complex/lane 的布局交换空间，再计算分组；`packed` 仅启用 FP32，避免
  将其错误地当作 FP64 的自动优化。FP32 524288 batch64 的正反 complex
  检查已通过，FP64 拒绝发生在生成阶段，未将此组记录算作通过。
- `FLAGFFT_HCU_FOURSTEP_RADIX_CAP=8|16`：对公开 1D、大于等于 65536 的支持分解
  请求，二次幂子叶的指数平衡分配到有界 radix；混合长度在已有候选中限制复合
  radix 的上界，保留不可再拆的受支持素数，以减少每线程寄存器负担。默认 auto
  保留原策略。请求上下文包含该值；固定 thread-local 候选不能绕过限制；
  素数卷积及其他后端保留原路线。
  混合长度扩展在原 acceptance 340200 FP64 使用 `[28,18]` / `[27,25]`
  大寄存器蝶形的证据下加入，不依赖 holdout 结果；尚待设备资格验证。
  `FLAGFFT_HCU_FOURSTEP_PRESERVE_SPLIT=1` 另外隔离全局行列分解与局部蝶形：
  使用独立的未限 radix 的 cost/node cache 选择原分解，再对两个叶子限制 radix。
  只覆盖原有支持的公开 1D 域；非两个叶子的计划回退原策略，默认 0。
  它用于检查混合长度的回退是否来自分解变化，不引入逐长度路由表。
  `RADIX_CAP=fp64` 仅对 FP64 应用 cap16；`PRESERVE_SPLIT=irreducible` 仅在
  根分解含有大于所选 cap 的不可拆素因子时保留原切分。两者仍默认关闭，
  用于验证按精度和数学分解类别组合策略，不引用长度赢家表。
- `FLAGFFT_HCU_FOURSTEP_PACKING=fp32_wave`：对 FP32 两阶段、相同 lane 数且
  符合循环单缓冲条件的叶子，将每 CTA 的 FFT 分组限制在一个查询 wave 内。
  不同 lane 的阶段、FP64 和 legacy profile 保留资源分组；默认 resource。
  该候选用于检查跨 wave 同步/布局转换的成本，尚待设备资格与性能验证。
- `FLAGFFT_HCU_FP64_CONV_RADIX_CAP=8|16`：在选好质数算法和卷积长度后，
  对至少 65536 点的受支持 FP64 卷积 FFT 子计划，使用独立上下文应用有界
  蝶形。它不修改公开请求、环境变量、卷积 extent 或父计划的算法成本，
  小卷积、FP32 和其他后端保留原策略。默认 auto；尚待设备资格验证。
- `FLAGFFT_HCU_3D_CT=1`：二次幂 FP32 3D 的两遍 Cooley-Tukey 原型。两遍 tile
  均由 8192 complex 元素预算推导，batch 使用独立地址，不是逐 shape 路由表。
  默认关闭，FP64/不支持几何保留原实现。
- `FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER=1`、`FLAGFFT_HCU_LEAF_SINGLE_BUFFER=1`：
  仅在所有覆盖写的中间阶段均可一次覆盖完整 FFT 时复用缓冲；每次覆盖前有
  CTA barrier，多组阶段自动回退。leaf 还根据原共享内存占用和设备容量限定范围。
  Bluestein 的独立 FourStep prepare/pointwise/finish 阶段也必须采用一致的
  缓冲数：原资源分组已按单缓冲计算，发射体先前只识别普通 FourStep 前缀。
  已统一单 FFT 阶段的分配及循环地址；包含两次 FFT 的 full-leaf 不在此范围。
  扩展源代码检查通过，仍需 large-prime 设备验证。
- `FLAGFFT_HCU_2D_REAL_ROWS=1`：复用已有紧凑行调度，限 HCU dense 2D、
  大于现有小 RC 范围的首轴、偶数末轴；无 shape winner 表。
- `FLAGFFT_HCU_2D_REAL_COL_STORE=1` 在紧凑行路线中复用现有列 FFT 的融合
  置换写回，删除一次完整矩阵转置；限列为 Leaf 的计划，默认关闭。
  wave outer store 允许奇数 span，以每个矩阵独立向上取整计算启动数，
  保留每个尾 tile 的掩码。CPU 地址检查覆盖奇数半谱、batch1/4 和两精度；
  设备资格与性能尚待验证，未将此开关设为默认。
- `FLAGFFT_HCU_FUSED_PACKED_REAL=1`：复用 LeafPackedR2C/C2R，按半长 FFT
  和重建共六个标量缓冲的 LDS 上界限定候选，未通过性能资格，不默认开启。
- `FLAGFFT_HCU_SMALL_WAVE=1`：FP32、16–64 的二次幂连续叶子在一个 wave 内
  填满 FFT 子组，保留既有距离与尾 batch 掩码；FP64/其他 I/O 方式回退。
- `FLAGFFT_HCU_DENSE_WAVE=1`：FP32、64–2048 的二次幂连续叶子和 Four-Step
  行列复用 wave 内 Cooley-Tukey。`FLAGFFT_HCU_DENSE_WAVE_FP64=1` 额外允许
  FP64，使用附加在末级表后的双精度自然顺序旋转因子；保留原表前缀以供回退，
  不使用 FP32 三角函数。输出重排按半个查询 LDS 容量分块。
  FP64 普通 wave 叶子另按输入/输出重排的寄存器预算回退大 tile：2048 点实测
  已用到 256 VGPR 且有 112 字节 private segment，因此不能仅凭 LDS 合法就选择。
  FP32 packed 边界另按重建需要的双倍活跃 payload 限制在 64 个标量/物理 lane。
  设备验证前均保持 opt-in。
  FP64 packed R2C 的首次二维验证暴露 Nyquist store 的指针张量未显式扩展到
  value/mask 形状，编译失败，未执行设备 kernel。已统一两种精度的显式广播；
  CPU 数学对照新增 packed 正反实数路径，并要求 store 的广播结果不能扩大
  指针形状。失败日志保留，修复后需重新进行设备验证。
- `FLAGFFT_HCU_WIDE_WAVE_LANES=auto`：FP64 连续叶子按每个逻辑 lane 的
  输入字节预算选择 64 或 128 个逻辑 lane，后者由两个物理 wave 协作；默认
  `64` 保留较大叶子的原生回退。FourStep 和 packed 边界保留 64-lane ABI。
  发射和 metadata 同时调整 warp 数与输出分块，CPU 直接执行生成的 2048 点
  数学代码，C2C 正反向、R2C/C2R、尾 batch=3 与 NumPy 对照通过。
  此候选针对 FP64 单 wave 的寄存器溢出，尚待设备资格验证。
- `FLAGFFT_HCU_WAVE_ROOT_TABLE=1`：FP32 wave 路径也复用自然顺序旋转因子表，
  FourStep 行读取已有跨维度系数，packed 边界读取已有实数重建表，移除设备端
  三角函数。FP64 packed wave 同样使用双精度表，候选不再经过 FP32 三角函数。
  `FLAGFFT_HCU_DENSE_WAVE_PACK` 的 16-row 实验会按每物理 lane 的活跃 payload
  自动缩小；重建按双倍 payload 计，不仅按共享内存容量放大 tile。
- `FLAGFFT_HCU_WAVE_ROOT_PERIOD=4|8` 对 FP64 的 wave 内系数分块递推，每隔
  固定数量重新从精确类型的表取值，限制累积误差并减少同时存活的独立系数。
  它允许另行筛选大 FP64 叶子的单 wave 实现，避开已拒绝的跨 wave FFT。
  CPU 对生成数学进行 C2C 正反向、R2C/C2R、packed 和置换写回对照，
  连续与 packed 路径的最差逐 batch L2 误差低于 4e-15、Linf 低于 6e-15。
  默认 0，尚待设备资源、正确性与性能资格；FP32 数学保持原路径。
  `FLAGFFT_HCU_DENSE_WAVE=leaves` 仅替换连续/packed 叶子，保留 FourStep
  行列的原生实现，以便独立验证这两类内存访问方式。
- `FLAGFFT_HCU_WAVE_SCALAR_BUDGET=128` 是默认 64 以外的 FP32 寄存器预算实验，
  用于在 LDS 分块约束下测试更宽的合并写回；FP64 仍保留原预算。该实验没有
  设备资格，不作为默认。wave 中轴的 batched 地址按每个 cube 的行数计算，
  即使一个 CTA 跨越两个 cube 也不混用输出 stride；保留原始参数 ABI。
  CPU 对生成地址求值，与独立 NumPy 轴置换逐项比较，通过 batch2/3/5 和
  不同外轴长度的验证；仍需设备数值测试。
- 同时开启 `FLAGFFT_HCU_3D_WARP_FFT=1` 与 `FLAGFFT_HCU_WAVE_ROOT_TABLE=1`
  时，对 dense FP32 C2C、三个轴均为 64–2048 的二次幂且均为叶子的计划，
  默认选择三个融合置换 store，替代 hybrid 中轴的显式转置；静态 span/batch
  默认随之启用。原有环境覆盖仍优先。FP64、复制输入、实数根、非支持轴和
  非叶子计划保留既有图结构，不使用尺寸赢家表。该接入仍是 opt-in，
  尚需对自动路由本身进行统一候选的完整设备验收。
- `FLAGFFT_HCU_3D_WARP_FP64=1` 额外筛选 FP64 的复数及实数前向融合置换 store。
  按 32 个 FP64 标量/物理线程和半个 LDS 容量限定分组；超出一个 wave 的
  输入字节预算时回退原生，行对齐使自动分组小于四行时也回退。保持 FP64
  原有 hybrid/末级转置选择，不复用 FP32 整体图的强制覆盖。
  CPU 直接执行发射数学，覆盖 outer/inner、正反向、R2C、尾 tile，并与
  NumPy FFT 后独立轴置换对照；尚未取得设备资格，默认 0。
- 两个 wave 协作的 FP64 连续叶子已完成二维 8/8 双参考校验，但 18 个性能
  样本中候选仅约 0.216–0.304x，而同轮原生约 0.571–0.874x；拒绝该自动
  路线。FP64 融合置换候选因此保留单 wave 的输入预算，不扩展大叶子。
- 半长度实数策略首先按 FFT 工作量、原/子计划结构、布局及资源界限建立适用域，
  再经过同卡重复性能、三幅度准确性和 holdout 决定默认行为；不增加逐 case winner 表。
- 下一轮已冻结的 half-length 策略为 `FLAGFFT_HCU_LARGE_PACKED_REAL=1`（默认 0）：
  HCU gfx936、公开 dense 1D、偶数长度、原计划为 FourStep、总工作量
  `batch*N >= 65536`，半长子计划必须为 Leaf 或两个 Leaf 的 FourStep。
  它保留 `FLAGFFT_PACKED_REAL=0` 的显式关闭及已有布局回退。
  该条件在 large holdout 执行前固定；holdout 不能用来重新选择阈值。

CPU 检查：packed wave 提交的综合回归为 965 passed、88 skipped、10 subtests passed；
root-table 与寄存器回退新增后的相关回归为 367 passed。192 组默认 FP32 wave
生成源码与扩展前逐字一致。远端 CPU planner target（含混合 radix）：28 个 GTest
通过。以上不代替设备验证。
尚未宣告完成 1.0x / 0.7x 的最终验收。


## 后续筛选记录（2026-11-16 夜间）

`0fac7c1c` 的 FP64 卷积 radix cap=16 已完成大质数公开接口的 8/8 完整 batch
NumPy 与 hipFFT 验证。524287 点、batch64 的 C2C/R2C/C2R，原约
0.674/0.684/0.684x，候选为 1.120/1.118/1.112x；batch1 约 1.405–1.420x。
该策略只在质数算法及卷积长度确定后限制子 FFT 的蝶形大小，不更换卷积长度。
原始结果为 `20261116_152600_convolution_accuracy/` 和 `convolution_screen/`。

同一版本的自动 FP32 3D wave 图已完成 16/16 双参考数值验证。256³ 和
128×2048×64 的 C2C，batch1 约 0.752/0.834x，batch4 约 0.767/0.914x。
FP64 wave 置换 store 虽数值通过，却使 cube 的 C2C、R2C、C2R 明显回退，
因此不作为统一默认策略。数据见 `20261116_152600_wide_store_accuracy/`
及 `wide_store_screen/`。

`35431c76` 的 33 项远端 CPU GTest 通过，包含奇数 span 的逐平面尾 CTA
覆盖检查。FP32 uniform FourStep 单 wave 打包实验已拒绝：同轮 batch64
131072 点从 0.688x 回退到 0.595x，524288 点从 0.841x 回退到 0.484x。
保留原 `resource` 打包，不因单个资源指标较低而启用候选。

`f79f1ded` 起新增默认关闭的 `FLAGFFT_HCU_LEAF_PACKING=wave|two_waves|four_waves`，
按实际 stage lanes 填充物理 wave，并由共享内存预算限制分组。
`b1ac471c` 将安全单缓冲复用计入资源上界；不满足完整中间 stage 读取条件的
计划继续按双缓冲核算。其效果仍待设备验证及完整候选验收。

`a6e812eb` 新增默认关闭的 `FLAGFFT_HCU_FP32_ROW_TWIDDLE=1`，在完整工作集
超过 LDS、两个子轴均不超过 1024 时，将已有 FP32 系数表的读取从列 FFT 输入
移到行 FFT 输出，避免跨列系数访问；不使用设备近似三角函数。
`11358358` 新增 `FLAGFFT_HCU_CONV_BALANCE=1|fp64`，只在已选卷积 FFT 为
FourStep 时枚举可用叶子因子对，选择长宽比最小的矩阵；保持质数算法与卷积长度。
这两项均无尺寸赢家列表，尚未取得设备性能资格。

目前未完成统一配置的 400 case 完整 batch 验收，不能据筛选结果宣告达到目标。
