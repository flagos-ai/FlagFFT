# MACA exchange 实验审阅记录（2026-09-21）

本记录覆盖 `direct` / `direct_all` 的作用域与布局 guard。已测 direct v1 为
`211e110`，整组 padding 前移的 v2 为 `d54dcc2`；`00e556e` 的 MSB split
是独立 opt-in 负实验，默认仍为 LSB。所有实验开关默认关闭。

## 容器 CPU 验证

主机修改代码，容器只运行。实际回归命令：

```sh
docker exec flagtree-dev3 /root/.pyenv/versions/3.12.13/bin/python -m pytest \
  /workspace/FlagFFT-dev-maca-single-exchange/tests/python/test_codegen.py \
  /workspace/FlagFFT-dev-maca-single-exchange/tests/python/test_maca_exchange.py -q
```

结果：271 passed（10.76 秒，MSB 候选提交前）。覆盖 FP32/FP64、LSB/MSB、
1024/2048、padding 倍数 2/4/8、batch slot padding、inner-pack=4 interleave，
以及 390/476/450/729/378/900/375/1768/1008 的混合因子。CPU 检查是逐元素
路由语义对照，不等同于这些组合都已通过 MACA 编译和 GPU 数值验收。

另执行了以下源码作用域审计（两段实际执行的内联 Python 合并为可复现命令）：

```sh
docker exec -i -w /workspace/FlagFFT-dev-maca-single-exchange flagtree-dev3 \
  /root/.pyenv/versions/3.12.13/bin/python - <<'PY'
import ast, os, re, sys
sys.path.insert(0, 'python')
from flagfft_codegen.backend_profile import BackendProfile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_leaf import (
    _build_leaf_kernel_source, _build_leaf_kernel_source_for_io,
)
plan = LeafPlan(1024, (16, 8, 8), 1, 64, 2, (), 1024)
for backend in ('cuda', 'musa', 'ppu', 'npu'):
    set_profile(BackendProfile(backend=backend, device_arch='audit'))
    os.environ.pop('FLAGFFT_MACA_EXCHANGE', None)
    _, baseline = _build_leaf_kernel_source(plan)
    for mode in ('join', 'transpose', 'direct', 'direct_all'):
        os.environ['FLAGFFT_MACA_EXCHANGE'] = mode
        _, source = _build_leaf_kernel_source(plan)
        assert source == baseline, (backend, mode)
    print(backend, 'byte-identical source for all exchange knobs')
set_profile(BackendProfile.from_device(
    {'backend': 'maca', 'device_arch': '102', 'warp_size': 64}, 'legacy'))
modes = (
    'contiguous', 'strided', 'permuted_store', 'contiguous_r2c',
    'contiguous_c2r', 'four_step_row', 'four_step_col',
    'four_step_row_strided', 'four_step_col_strided', 'four_step_r2c_col',
    'four_step_c2r_col', 'four_step_real_row', 'four_step_hermitian_row',
    'bluestein_prepare_leaf', 'bluestein_finish_leaf', 'bluestein_full_leaf',
    'bluestein_four_step_prepare_row', 'bluestein_four_step_pointwise_row',
    'bluestein_four_step_finish_col',
)
checks = 0
for mode in modes:
    for pack in (1, 2, 4):
        os.environ['FLAGFFT_MACA_BATCH_PACK'] = str(pack)
        os.environ['FLAGFFT_MACA_INNER_PACK'] = str(pack)
        for exchange in ('direct', 'direct_all'):
            os.environ['FLAGFFT_MACA_EXCHANGE'] = exchange
            _, source = _build_leaf_kernel_source_for_io(
                plan, io_mode=mode, four_step_n1=1024, four_step_n2=1024,
                prime_n=997)
            known = set()
            for statement in ast.parse(source).body[0].body:
                names = [x for x in ast.walk(statement)
                         if isinstance(x, ast.Name) and re.fullmatch(
                             r'smem_[ab]_register_[ri][0-9]+', x.id)]
                for node in names:
                    if isinstance(node.ctx, ast.Load):
                        assert node.id in known, (mode, pack, exchange, node.id)
                known.update(x.id for x in names if isinstance(x.ctx, ast.Store))
            checks += 1
print('register definition/use guard checks:', checks)
PY
```

结果：CUDA/MUSA/PPU/NPU 的 contiguous 源逐字不变；19 个 IO mode × 3 个
pack × 2 个模式，共 114 份 MACA kernel 的源码 AST 与寄存器先定义后使用检查通过。
这证明 guard 配对，不证明全部 IO mode 的数值正确性。

## Guard 与布局结论

- `portable_exchange` 仅在 MACA 后端启用；load bypass 还要求 portable 为真。
  `_emit_distributed_split_tree` 新增参数默认保持旧行为，其他调用没有启用它。
- direct 需要全部 radix 为 2 的幂、`slot_stride == n`、`size == n * pack`。
  构造端还检查 `lane_block >= n / radix`；该关系由 leaf builder 对所有 stage
  lanes 取最大值并向上取 2 的幂保证，因此与读取端 guard 一致。
- batch-pack ≥ 4 时现有布局使用 `smem_size + 1` slot stride，明确回退到 joined
  gather，不会从未生成的 stage registers 读取。混合 radix 的 `direct_all` 仅将
  register digit 补至 2 的幂，索引 stride 使用补齐后的 radix。
- 连续 batch 是 `[slot, lane]`；four-step inner packing 是 `[lane, slot]`。
  构造端的 `register_lane_stride > 1` 与调用端 `inner_pack > 1` 一致。
- padded lanes 在 split 前整组补零。invalid batch/inner slots 从 masked input 的
  零值经线性蝶形传播；GPU 的不整除 pack 尾部仍需要额外数值验收。
- full Bluestein 第一 pass 的最后一 stage 使用 natural-order 中间缓冲；第二
  pass stage 0 仍用普通 gather 读取，后续 stage 才 bypass。审计没有发现引用旧
  pass 未重建寄存器的情况。

## 1048576 四步的 pack 与 metadata

用相同 MACA target `maca:80:64`、legacy profile、leaf `[16,8,8]`、
`FLAGFFT_MACA_EXCHANGE=direct`、`FLAGFFT_MACA_MAX_WARPS=8`，分别调用
`_build_leaf_kernel_source_for_io` 与 `_metadata` 检查 row/col，结果一致：

| INNER_PACK | 源码 vector block | metadata warps | 物理线程 | grid.x（每 pass） |
|---:|---:|---:|---:|---:|
| 2 | 256 | 4 | 256 | 512 |
| 4 | 512 | 8 | 512 | 256 |
| 8 | 1024 | 8 | 512 | 128 |

三个配置源码均无 gather。P8 的 1024 个逻辑元素由 512 线程承载，合法；它不要求
16 warps。默认 cooperative warp cap 已是 8，实验显式写 8 用于排除环境覆盖。
若覆盖成 16，则现有 metadata 算法可产生 1024 物理线程，触及当前 MACA launcher
的 512 线程限制。不同 execution policy 可能进一步按共享内存预算减小实际 pack，
不能只读环境变量判断 launch。

## IO 事务估计与 VEC_IO 验证条件

以下是地址模型推导，不是硬件计数器测量。按 C500 文档的 16-lane / 128B 聚合，
假设无跨界、不计 cache 重用、线程沿当前连续 lane_vec 映射，FP32 complex：

| pack P | row load 涉及段数 | row store 涉及段数 | col load/store 各自段数 |
|---:|---:|---:|---:|
| 2 | 8 | 2 | 8 |
| 4 | 4 | 4 | 4 |
| 8 | 2 | 8 | 2 |

row/col input 与 col output 的形式是 `fft_index * 1024 + inner`，每组 P 个
complex 连续；row output 是 `inner * 1024 + fft_index`，每个 inner 的相邻
FFT 元素连续。P4 每条 scalar real/imag 指令在每个 128B 段仅使用 16B，合计
complex payload 32B。如果 VEC_IO 形成每线程 64-bit 访问，可减少两条 scalar
指令及 partial write，但它本身不会消除这些地址段之间的稀疏性。

最小验证保持 direct、pack4、MAX_WARPS8，只改变 `FLAGFFT_MACA_VEC_IO=0/1`。
需同时确认：NumPy 与平台正确性；metadata 的 pack/warps 一致；LLIR load/store
类型是否变宽、数量是否减少；新增 layout conversion/barrier/shfl、寄存器、shared
和 private memory；最后看独占 GPU 的交错 A/B 端到端时间。`tl.join`/`tl.split`
源码不保证最终宽访存，若编译器改变 component 的 lane 分布，必须重新计算事务。

## 尚未覆盖的验收与资源边界

- GPU 已测的 1024/2048 FP32 complex 及 1048576 four-step 不能外推到所有
  FP64、实数、small multi-stage 和不整除 pack 的尾部。
- mixed `direct_all` 的 join 张量可能比有效状态大数倍；1768 的 radix17
  补至32尤其需要检查 shared。FP64 的字节量翻倍，不能直接复用 FP32 pack 上限。
- 实验环境变量不进入现有 C++ JIT key；A/B 必须使用独立源码镜像、可执行文件目录
  和 Triton cache，并检查实际生成源码及 metadata。
- 记录中的 barrier/load/store 数取自 lowered LLIR，不冒称最终 ISA 指令数；
  mtreg/streg/private 来自最终 ELF note，动态 shared 来自 Triton metadata。
