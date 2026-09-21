# MACA single prime 路由与实验覆盖

证据口径：以下自动路由来自 `results/20260918_124300_maca_acceptance36/1d_prime_single_{c2c,z2z,r2c,c2r,d2z,z2d}/performance_result.json` 的运行 plan。`14834ae..45eb2a8` 的 `src/plan` 差异只有 NPU 分支与格式整理，MACA 的候选、成本模型、因子选择未变；因此在同 C550、无 tuned-plan DB 的条件下，它们也是本轮预期路由。997 FP32 另有本轮实测 plan 确认。历史 plan 的 packing/warp metadata 不作为本轮配置依据。

FP32 指 C2C/R2C/C2R；FP64 指 Z2Z/D2Z/Z2D。两个融合开关均默认关闭，表中“覆盖”指显式开启后可触发的路径，不代表已通过设备验收。

| n | FP32 自动路由 | FP64 自动路由 | leaf 融合覆盖 | four-step 融合覆盖 | direct exchange 预期覆盖 |
|---|---|---|---|---|---|
| 23 | DirectDFT(23) | 同左 | 否 | 否 | 否；这是独立直接 DFT kernel，非 CT leaf 的单 radix 模式 |
| 997 | Bluestein，卷积2048，leaf [16,16,8] | Bluestein，卷积2000，leaf [20,10,10] | 两精度均是 | 不适用 | FP32 完整；FP64 混合 radix 回退，不能由2048的FP64测试代表 |
| 1009 | Rader，卷积1008，leaf [7,6,6,4] | 同左 | 否（根为Rader） | 否 | 非2幂 mixed leaf 不能完整进入 direct；小 radix 的局部回退不等于整路径优化 |
| 8191 | Rader，卷积8190，four-step 63×130；[7,3,3] / [13,5,2] | 同左 | 否 | 否（根为Rader） | 两个 mixed leaf 均非完整 direct |
| 16381 | Rader，卷积16380，four-step 90×182；[5,3,3,2] / [13,7,2] | 同左 | 否 | 否（根为Rader） | 两个 mixed leaf 均非完整 direct |
| 524287 | Bluestein，卷积1048576，four-step 1024×1024；两叶[16,8,8] | Bluestein，卷积1048576，four-step 512×2048；[8,4,4,4] / [16,16,8] | 否 | FP32 是；FP64 否（child2048超过<=1024条件） | 两精度的2幂叶均可进入 direct，但FP64边界融合仍关闭 |

8191/16381 的默认 four-step 子叶尺寸确实全部小于1024，但属于 Rader 的卷积，不能因此推断 Bluestein gate 会生效。若通过现有 tuner 选择 Bluestein 候选，FP32 卷积分别为16384与32768（源码 `next_supported_convolution_length` 推导）；候选具体 child plan 应从 tuner/print-path 确认后再验收，不能沿用 Rader 的63×130、90×182尺寸。

1009 的 FP32 Bluestein 候选会使用2048卷积，因而可能同时获益于 direct exchange 与两段 leaf 融合。既有 `flagfft-cli tune --shape 1009 --max-candidates 2 --finalists 2 --no-save` 可比较两条算法；不需要修改默认 prefer_bluestein 策略。

正确性覆盖约束：新增针对性 Python 测试固定997/2048，用于双精度、双方向与四种输入的边界核数值检查；FP64生产路由的2000卷积仍须由六 API 的997端到端 NumPy矩阵覆盖。524287 当前four-step融合测试应先限定FP32，不能把结果外推到512×2048的FP64路由。


## 997 稳态对照与归因（2026-09-21）

短 warmup 的结果不能直接比较：同一纯融合 CLI 的 warmup=5、iters=50 样本中，初始 FF/ref 中位约193.024/62.336 µs，后段降至73.472/26.624 µs，两序列相关系数0.992。该变化在奇偶提交顺序中均出现。warmup=200、iters=100 后三轮稳定，故下面使用该稳态口径；它不替代旧验收口径的全量历史分数。

严格对照的A/C/D位于 `results/20260921_152650_maca_single_control_round1_repeat/`；同一二进制、同一codegen源码，各变体独立可执行目录与cache，`FLAGFFT_TUNE_DISABLE=1`。每变体三次是同进程重复shape，保留每case的独立建plan、warmup和原有交错event测量。A/C的997是cases索引2/5/8，D是0/1/2；不能按相同case_id去重。

| 变体 | exchange | 两段leaf融合 | FF中位µs（三重复） | mcFFT中位µs | mcFFT/FF |
|---|---|---|---:|---:|---:|
| A | 原gather | 关 | 73.472 / 73.472 / 73.472 | 26.368 / 26.368 / 26.368 | 0.358885 |
| C | direct v2 | 关 | 34.048 / 34.048 / 34.048 | 26.368 / 26.624 / 26.368 | 0.774436–0.781955 |
| D | direct v2 | 开 | 32.512 / 32.512 / 32.512 | 26.624 / 26.880 / 26.624 | 0.818898–0.826772 |
| B（补充参考） | 原gather | 开 | 73.472 / 72.960 / 73.472 | 26.624 / 25.856 / 26.624 | 0.354386–0.362369 |

B来自 `results/20260921_151240_maca_single_timing_stability/round{2,3,4}_w200_i100.json` 的独立进程复测，源码为较早的cc4d2e7，不能冒充A/C/D同source严格2×2里的第四格。其两段kernel资源已独立核实，但严格性能归因采用A→C→D。

- A→C：时间减少39.424 µs，FFT级间交换优化使整体约2.158倍提速。
- C→D：再减少1.536 µs，融合使整体额外约1.047倍提速（时间减少4.51%）。
- A→D：整体约2.260倍提速。按这条实验顺序分摊总40.960 µs减时，交换占96.25%，融合占3.75%；这不是宣称两因子无交互的通用贡献比例。
- D在997 C2C forward的稳态初筛越过0.8，余量很小；三次同进程重复不等于跨进程稳定性或双方向/全精度验收。

## 路由、缓存与编译资源复核

A的运行plan确为 `CompiledRawBluestein`，稳态每次execute发射5个kernel（prepare、FFT、pointwise、FFT、finalize）。D/B为 `CompiledRawBluesteinLeaf`，稳态发射2个kernel。首次卷积核B的FFT预计算在warmup内完成，不计入上述稳态发射数。这是源码和plan核对，不是硬件timeline动态发射计数。

A的2048 FFT fatbin SHA256 `17d00d72a30ed5565549d15d0a80c9641739143d7b3a3a6a844ff146b343062f` 与早先基线完全相同；A的三个独立边界kernel没有shared memory或LLIR barrier。已排除A误用融合或direct缓存的情况。

| kernel | LLIR静态barrier call | Triton动态shared | mtreg/streg | ELF private bytes |
|---|---:|---:|---:|---:|
| A普通2048 FFT | 223 | 8 KiB | 76/56 | 0 |
| B prepare FFT | 223 | 8 KiB | 76/60 | 0 |
| B finish FFT | 223 | 8 KiB | 76/62 | 0 |
| direct v2普通2048 FFT | 39 | 16 KiB | 90/24 | 0 |
| D prepare FFT | 39 | 16 KiB | 88/28 | 0 |
| D finish FFT | 39 | 16 KiB | 88/28 | 0 |

这些是LLIR静态站点与最终ELF资源，不能称为最终ISA barrier数量或动态执行次数。ELF static shared=0与Triton动态shared不矛盾。融合没有减少原gather路径内部的223个同步站点；direct v2将其降到39。MSB split实验仍生成同样39个站点与相同资源，未构成独立改进方向。

相关完整IR/ELF/哈希摘要保存在：

- `results/20260921_152650_maca_single_control_round1_repeat/artifacts/A/resource_summaries.json`
- `results/20260921_145530_maca_single_fused_997/artifacts/triton-cache/resource_summaries.json`
- `results/20260921_145420_maca_single_direct_fused_v2_997/artifacts/triton-cache/resource_summaries.json`

## 最小设备正确性清单与状态

当前已验证的小prime快速数值入口是997 C2C forward，尚不能代表下面整套矩阵完成。最终候选使用同一codegen环境完成：

1. 997端到端六API、scale=1：C2C/Z2Z各forward/inverse，R2C/D2Z forward，C2R/Z2D inverse，共8个case。FP64必须记录plan确为卷积2000、factors20/10/10，并用NumPy独立判定。使用统一runner的六个 `1d_prime_single_*` operator、`--accuracy-only --shapes 997 --scales 1`，不要加`--max-cases 1`截断方向。
2. 仅运行新增 `test_maca_codegen.py -k bluestein_boundary_leaf`：4个参数组（两精度×双向），每组随机复数、纯实、纯虚、末端脉冲；这是固定2048卷积的边界核检查，不替代上项FP64生产2000路由。
3. batch gate-off可复用单个 `test_maca_runtime` 的 `MacaRuntime.ForcedRaderAndBluesteinSupportStreamsGraphsAndInPlace`：固定257/batch7，两个精度、两个方向、两个算法，已有dynamic_cast断言Bluestein仍走generic node。不运行旧leaf测试的数百case大矩阵。
4. 524287 four-step融合先验FP32 C2C双方向；此时必须看到1024×1024及CompiledRawBluesteinFourStep。FP64默认512×2048不满足gate，只能作为direct交换的单独验证，不能把FP32融合结果外推。
5. 1009/8191/16381先用现有tuner无保存地比较Rader/Bluestein，再将胜出的确切plan放进NumPy验收；不全局改prefer_bluestein。

默认关闭条件已审阅：两个新gate只在MACA、实际compile batch=1、对应leaf结构以及精确环境值1时打开；real wrappers的外层转换保持原执行语义，内部complex dtype正规化后可进入gate；non-MACA、batch>1及超界four-step保持原路线。没有增加仅复制条件表达式的测试。唯一发现并修复的既有一致性问题是prepare/finish leaf未列入CONTIGUOUS_BATCH_PACK_KERNELS；CPU测试比较真实生成batch stride与metadata，在16/32/256/2048上覆盖该问题。

## MACA single prime tuner 的卷积候选扩展

旧 `tune` 对prime只比较默认Bluestein与Rader；Bluestein的长度和child分解仍由静态成本选一个。因此997 FP64不会产生BS2048，524287 FP64不会产生1024×1024。单独tune1048576的DB结果也不会被Bluestein child自动继承。

新增候选仅限显式 `tune`、MACA、batch=1：先保留automatic及既有算法候选，再追加 `ceil_pow2(2n-1)` 的自动child；若child为four-step，则追加最平衡且两子节点均为leaf的分解。所有候选按完整PlanKey去重并遵守max-candidates，默认planner与融合gate不变。单精度已用相同幂次/平衡分解时不会重复。新入口在调用signed power helper前拒绝无法表示卷积幂次的极端长度。

下面由validation在独立cache、稳定warmup条件下串行执行；命令中的CLI须包含候选扩展，codegen须使用已验收direct v2，结果写入独立套件目录。997用3个finalists是为了让两个Bluestein方案都进入长测，而不会被短测先筛掉：

```sh
FLAGFFT_MACA_EXCHANGE=direct FLAGFFT_MACA_BLUESTEIN_LEAF_FUSION=1 \
  "$FFT_CLI" tune --api z2z --shape 997 --batch 1 \
  --max-candidates 3 --finalists 3 --screen-warmup 200 --screen-iters 100 \
  --final-warmup 2000 --final-iters 100 --no-save --json

FLAGFFT_MACA_EXCHANGE=direct FLAGFFT_MACA_BLUESTEIN_FOUR_STEP_FUSION=1 \
  "$FFT_CLI" tune --api z2z --shape 524287 --batch 1 \
  --max-candidates 2 --finalists 2 --screen-warmup 200 --screen-iters 100 \
  --final-warmup 2000 --final-iters 100 --no-save --json
```

997 FP64候选顺序为BS2000、Rader996、BS2048；524287 FP64为BS1048576(512×2048)、BS1048576(1024×1024)。每条命令的双方向均由tuner内部正确性门控，但其计时不采用bench的交错次序，选择winner后仍需标准bench与NumPy验收。大prime命令同时比较分解和融合是否可用；若要单独归因分解，应先将four-step融合开关设0，再测开启后的winner，不能直接把两候选差额全部归因给分解。

CPU验证：独立 `test_maca_prime_tune` 直接链接planner源码与C550能力stub（64 KiB，来自本轮manifest），不链接GPU runtime或JIT。4个测试通过，覆盖上述真实候选、顺序/限额、默认plan前后不变、non-MACA与batch>1原算法候选、FP32去重与非法长度。日志/XML：`results/20260921_155300_maca_single_prime_tuner_cpu/`。此验证只证明候选生成，新增FP64路由的GPU正确性与速度尚待validation验收。
