# 有界 MACA 长尾计划实验

基于 dev `12cc497`。`FLAGFFT_MACA_TAIL_PLAN` 复用现有 `PlanBuilder`、
`tune` 的候选编译/双方向正确性检查/计时，以及 C API 的普通建计划链路。
没有修改自动 planner、成本模型或 `compiler.cpp`，也不自动选择新默认。

## 固定矩阵

| 长度 | 精度 | 指定计划名（比较顺序） | 候选数 |
|---|---|---|---|
| 997 | FP64 | `bs2000`, `bs2048` | 2 |
| 1009 | FP32 | `rader1008`, `bs2048` | 2 |
| 1048576 | FP64 | `ct512x2048`, `ct1024x1024` | 2 |
| 663000 | FP32 / FP64 | `ct375x1768`, `ct650x1020`, `ct780x850` | 3 |
| 328050 | FP32 / FP64 | `ct450x729`, `ct486x675`, `ct405x810` | 3 |

`bs` 数字是卷积长度；`rader1008` 是 1009 点 Rader 的 1008 点卷积；
`ctAxB` 是根节点 FourStep 的有序 n1/n2。子计划继续使用现有后端感知
planner 构建，完整结构可从 tuner 的 `plan_key` 或 bench 的 `--print-path` 核对。

- 未设置或空值：保持原有建计划及 tuner 候选行为。
- `compare`：仅用于 `tune`，准确返回表中全部候选；上限大于表中数量也不扩展搜索。
  `--max-candidates` 小于候选数时报错，不静默截断。将 `--finalists` 设为候选数，
  让全部通过正确性门控的候选进入 final 测量。
- 具体计划名：普通建计划强制该根节点；`tune` 只返回这个候选。
- `default`：显式使用当前自动计划；`tune` 只返回该自动计划。
- 只对 `device_type=maca`、`raw_dim=1`、`batch=1`、当前请求的根长度生效。
  其他后端、batch、维度、为另一长度构建的内部 child 忽略此变量，包括非法值。
  作用域内不支持的长度、精度、名字或拼写报错。FP32/FP64 支持同精度的
  complex→complex、real→complex、complex→real 请求，拒绝混合精度。

实验必须同时设置 `FLAGFFT_TUNE_DISABLE=1 FLAGFFT_PACKED_REAL=0`。
DB 查询在 C API 调用 planner **之前**，因此必须在进程启动前禁用 DB，
不能依赖 planner 的参数检查拦截已经命中的 DB。关闭 packed-real 防止编译时将
全长计划替换成 N/2 计划。普通 MACA 优化策略及其他 codegen 开关沿用运行环境；
对照组应保持一致。`tune` 的 `--no-save` 仍须显式传入，禁用 DB 查询不等于禁用保存。

## main 的设备实验命令（本任务不执行）

由 main 在对应 MACA 容器中使用包含本提交的 CLI 串行安排。
每个阶段统一 **5 warmup / 30 iter**，没有后台调优或额外长度搜索。
下列 `FFT_CLI` 应替换成 main 构建的二进制绝对路径；结果统一进入工作区 `results/`。

```sh
FFT_CLI=/path/to/maca/build/flagfft-cli
FFT_SOURCE=/workspace/FlagFFT-dev-maca-tail-plans
export PYTHONPATH="$FFT_SOURCE/python"
OUT=/workspace/results/$(date +%Y%m%d_%H%M%S)_maca_tail_plans
mkdir -p "$OUT"
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_PACKED_REAL=0

compare_tail() {
  api=$1
  n=$2
  count=$3
  FLAGFFT_MACA_TAIL_PLAN=compare "$FFT_CLI" tune \
    --api "$api" --shape "$n" --batch 1 \
    --max-candidates "$count" --finalists "$count" \
    --screen-warmup 5 --screen-iters 30 \
    --final-warmup 5 --final-iters 30 --no-save --json \
    > "$OUT/tune_${api}_${n}.json" 2> "$OUT/tune_${api}_${n}.log"
}

compare_tail z2z 997 2
compare_tail c2c 1009 2
compare_tail z2z 1048576 2
compare_tail c2c 663000 3
compare_tail c2c 328050 3
# 如主套件为 FP64，使用相同候选：
compare_tail z2z 663000 3
compare_tail z2z 328050 3
```

`tune` 本身只接受 C2C/Z2Z，自动校验并测量 forward 和 inverse。检查 JSON 中
`candidate_count`、每个 `plan_key`、双方向 `correctness` 和 `final`。
编译错误或正确性失败的候选会显示 `error` / `invalid`，不会进入 final；
出现 winner 不代表其他候选已通过，也不代表 real API 已验收。

以下单计划 bench 通过真实 C API 链路强制每个根计划，并测量双方向和 real wrappers。
沿用上面的 `FFT_CLI`、`OUT` 和两个必要环境变量：

```sh
bench_tail() {
  precision=$1
  n=$2
  shift 2
  for plan in "$@"; do
    if [ "$precision" = fp64 ]; then
      cases='z2z:forward z2z:inverse d2z:forward z2d:inverse'
    else
      cases='c2c:forward c2c:inverse r2c:forward c2r:inverse'
    fi
    for case_spec in $cases; do
      api=${case_spec%:*}
      direction=${case_spec#*:}
      FLAGFFT_MACA_TAIL_PLAN="$plan" "$FFT_CLI" bench \
        --rank 1 --shape "$n" --batch 1 --api "$api" --direction "$direction" \
        --warmup 5 --iters 30 --print-path --json \
        > "$OUT/bench_${api}_${n}_${plan}_${direction}.json" \
        2> "$OUT/bench_${api}_${n}_${plan}_${direction}.log"
    done
  done
}

bench_tail fp64 997 bs2000 bs2048
bench_tail fp32 1009 rader1008 bs2048
bench_tail fp64 1048576 ct512x2048 ct1024x1024
bench_tail fp32 663000 ct375x1768 ct650x1020 ct780x850
bench_tail fp32 328050 ct450x729 ct486x675 ct405x810
bench_tail fp64 663000 ct375x1768 ct650x1020 ct780x850
bench_tail fp64 328050 ct450x729 ct486x675 ct405x810
```

在 `sh` / `bash` 中执行以上函数（zsh 默认不对 `$cases` 做分词）。
`default` 可代替任何一组计划名运行自动计划对照。bench 是性能入口；数值验收
仍由 main 的 NumPy runner 安排，设置同一个具体计划名、只传匹配的一个 shape，
并用 `--output-dir` 指向 `results/<时间戳>_<套件名>/`。不要用 `compare` 跑 bench
或 NumPy runner，也不要在一次强制计划命令中混入其他长度。

## 1024×1024 FP64：固定计划的 P2 / P4 对照

本提交负责强制 `ct1024x1024`，可与 main 的
`FLAGFFT_MACA_FP64_REGISTER_PACK=1` codegen 实验组合。该 codegen 改动不在本提交中；
运行前应由 main 合入两者，并将 `FFT_SOURCE` / `PYTHONPATH` 指向**同一集成 worktree**。
不能只设置新变量却仍导入未实现该变量的 baseline Python 包。

固定 `INNER_PACK=4` 和 `direct_all`，`FP64_REGISTER_PACK=0` 仍由原 FP64 SMEM
上限限制为 P2，`=1` 才允许实验 P4；二者都固定 1024×1024，以免混入拆分差异。
不必给 planner 增加 P2/P4 别名；pack 是 codegen 参数，root PlanKey 应相同。

```sh
# FFT_CLI / FFT_SOURCE 使用 main 的集成构建；不要在本分支 baseline 上声称测到 P4。
export PYTHONPATH="$FFT_SOURCE/python"
for register_pack in 0 1; do
  for direction in forward inverse; do
    FLAGFFT_TUNE_DISABLE=1 FLAGFFT_PACKED_REAL=0 \
    FLAGFFT_MACA_TAIL_PLAN=ct1024x1024 \
    FLAGFFT_MACA_EXCHANGE=direct_all FLAGFFT_MACA_INNER_PACK=4 \
    FLAGFFT_MACA_MAX_WARPS=8 FLAGFFT_MACA_FP64_REGISTER_PACK="$register_pack" \
      "$FFT_CLI" bench --rank 1 --shape 1048576 --batch 1 --api z2z \
      --direction "$direction" --warmup 5 --iters 30 --print-path --json \
      > "$OUT/bench_z2z_1048576_ct1024x1024_register${register_pack}_${direction}.json" \
      2> "$OUT/bench_z2z_1048576_ct1024x1024_register${register_pack}_${direction}.log"
  done
done
```

main 负责 GPU4 串行运行及编译产物资源核对。P4 仍须满足 padded tensor 上限、
pack≤4，且只限 512/1024 幂次 leaf；2048 guard 不变。
CPU 测试验证强制 1024×1024 的真实子叶满足该入口条件；本分支没有验证 P4 codegen。

## CPU 验证

新增 standalone CMake 目标，直接链接真实 planner 和 C API request/节点支持性代码，
复用既有 prime tuner 测试的 C550 64 KiB 能力 stub；不加载 GPU runtime 或 JIT。
GoogleTest 可通过系统包或 `FLAGFFT_GTEST_SOURCE_DIR` 指定的源码提供。

在主机执行（容器只编译和运行，代码均由主机编辑）：

```sh
docker exec flagtree-dev3 sh -lc '
  set -eu
  export PATH=/root/.pyenv/versions/3.12.13/bin:$PATH
  export PYTHONPATH=/workspace/FlagFFT-dev-maca-tail-plans/python
  cmake -S /workspace/FlagFFT-dev-maca-tail-plans/ctest/maca_tail_plans \
    -B /workspace/FlagFFT-dev-maca-tail-plans/build/maca-tail-plans-cpu -DCMAKE_BUILD_TYPE=Release
  cmake --build /workspace/FlagFFT-dev-maca-tail-plans/build/maca-tail-plans-cpu -j 8
  OUT=/workspace/results/$(date +%Y%m%d_%H%M%S)_maca_tail_plans_cpu
  mkdir -p "$OUT"
  /workspace/FlagFFT-dev-maca-tail-plans/build/maca-tail-plans-cpu/test_maca_tail_plans \
    --gtest_output=xml:"$OUT/gtest.xml" > "$OUT/gtest.log" 2>&1
'
```

测试检查闭合矩阵、因子乘积和卷积长度、Rader 排列表、真实 `raw_supported_node`、
强制根计划与 tuner PlanKey 一致、双方向和 real request、同一个 builder 切换设置后
默认恢复、无设置时旧 prime tuner 行为，以及其他后端/batch/维度不变。
CPU 测试不证明 Triton 编译、设备数值或性能；这些由 main 的设备实验补齐。

2026-09-22 容器 CPU 验证：9/9 通过（5 个新增测试 + 4 个既有 prime tuner
回归测试）。测试显式设置本 worktree 的 `PYTHONPATH`。日志和 XML 位于
`/rjs/llb/fft-dev/results/20260922_150006_maca_tail_plans_cpu/`。
