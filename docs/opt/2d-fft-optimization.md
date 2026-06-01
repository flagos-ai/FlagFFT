# 2D FFT 优化方向

当前 2D FFT 使用 RTRT（Row-Transform, Transpose, Row-Transform, Transpose-back）分解策略，
是正确性基线。本文档记录可选的优化方向，按 transform 类型和优先级组织。

## 2D

### C2C（单精度复数 complex64）

#### 策略级优化

- **RC 路径（消除 Transpose）**
  - 当前状态：RTRT 路径需要 2 次 transpose，额外 2× global memory 带宽 + 2 个临时缓冲区
  - 方案：列方向使用 `strided_batch` IO 模式的 SBCC kernel 直接按列做 FFT，无需 transpose
  - 预期收益：大矩阵（≥256²）减少约 50% global memory traffic，kernel launch 从 4 降至 2
  - 复杂度：中。需扩展 `kernels.py` 的 `LeafIoMode` 支持 strided IO
  - 参考：rocFFT `node_factory.cpp:Decide2DScheme()`、`tree_node_2D.cpp:RC2DNode`

- **策略选择器（Decide2DScheme）**
  - 当前状态：固定 RTRT，无运行时策略选择
  - 方案：小矩阵（fit LDS）→ 2D_SINGLE；列长度有 SBCC 且 n0≥56 → RC；其余 → RTRT
  - 预期收益：为后续多策略优化奠定基础
  - 复杂度：低。纯逻辑判断
  - 参考：rocFFT `node_factory.cpp:Decide2DScheme()`（第 743-753 行）

- **2D_SINGLE 融合 kernel（远期）**
  - 当前状态：不存在。小矩阵仍走 4-kernel RTRT
  - 方案：n0≤32, n1≤32 时，单 kernel 在 LDS 中完成 row FFT + col FFT。LDS padding（pow2→n+1）避免 bank conflict
  - 预期收益：小矩阵从 4 kernel + 2 buffer 降至 1 kernel + 0 buffer，可获 2-3× 加速
  - 复杂度：高。需 codegen 层支持两组 radices 的 Stockham 分解
  - 参考：rocFFT `stockham_gen_2d.h:StockhamKernelFused2D`、`tree_node_2D.cpp:Single2DNode`

#### Kernel 级优化

- **Transpose 使用 Shared Memory Tiling**
  - 当前状态：`_build_tiled_transpose_kernel_source()` 不使用 shared memory，直接 global memory 读写
  - 方案：标准 smem transpose：合并读→smem in-place 转置（padding 避免 bank conflict）→合并写
  - 预期收益：大矩阵 transpose 带宽从约 50% 提升至接近理论峰值，整体 10-30% 加速
  - 复杂度：中。重写 transpose kernel
  - 参考：rocFFT `rtc_transpose_gen.cpp`

- **FuseShim：FFT + Transpose 融合**
  - 当前状态：row FFT 和 transpose 为独立 kernel，中间结果写回 global memory
  - 方案：修改 Stockham kernel 输出 stride 为转置后 stride，消除独立 transpose kernel。条件：transforms_per_block ≥ 8（单精度）
  - 预期收益：RTRT 从 4 kernel 降至 2 fused kernel
  - 复杂度：高。需修改 leaf kernel 输出 stride 和 plan 层 fusion 检测
  - 参考：rocFFT `fuse_shim.cpp:RTFuseShim`

- **Twiddle-Transpose 融合**
  - 当前状态：`_twiddle_transpose_complex_kernel()` 已实现但 2D 路径未使用
  方案：FourStep 路径中用此融合 kernel 替代分开的 twiddle + transpose
  - 预期收益：大长度 2D FFT 减少 1 次 global memory 遍历
  - 复杂度：低。kernel 已存在
  - 参考：`kernels.py:_twiddle_transpose_complex_kernel()`

- **Transpose 自适应 tile_size**
  - 当前状态：tile_size 固定为 32
  - 方案：小矩阵用 16，大矩阵用 64；双精度减半
  - 预期收益：小矩阵减少 grid overhead，大矩阵提升 SM 利用率
  - 复杂度：低
  - 参考：VkFFT `AxisBlockSplitter.h`

#### 内存优化

- **临时缓冲区池化**
  - 当前状态：每个 plan 独立持有 temp1/temp2，构造时分配
  - 方案：全局 buffer pool 管理，execute() 时获取/归还
  - 预期收益：减少首次调用延迟和显存碎片
  - 复杂度：中
  - 参考：VkFFT `vkFFT_Structs.h:tempBuffer`

- **In-place 转置（方阵）**
  - 当前状态：transpose 始终 out-of-place
  - 方案：n0==n1 时用单 buffer in-place transpose（分块递归算法）
  - 预期收益：方阵减少 1 个临时缓冲区
  - 复杂度：中

- **CUDA Graph**
  - 当前状态：4 个 kernel 串行 launch，每次约 5-10μs CPU 开销
  - 方案：将 launch 序列捕获为 CUDA Graph
  - 预期收益：小矩阵（<100μs）时减少 20-40μs launch 开销
  - 复杂度：中

### Z2Z（双精度复数 complex128）

所有 C2C 优化方向均适用，差异点：

- **Shared Memory 容量**：complex128 每元素 16 字节（vs complex64 的 8 字节），2D_SINGLE 适用尺寸上限缩小（32×32 → ~22×22）
- **RC 路径 SBCC 可用性**：双精度 SBCC kernel 可用长度集合通常小于单精度
- **Transpose tile_size**：双精度应从 32 降至 16，避免 smem 超限
- **FuseShim 最小行数**：双精度 minRows=4（单精度=8），因每行占用更多寄存器

参考：rocFFT `node_factory.cpp:use_CS_2D_SINGLE()`（LDS 容量检查）、`fuse_shim.cpp`（minRows 阈值）

### 通用优化

- **Plan Cache 2D 支持**
  - 方案：将 2D plan key 纳入 plan cache，避免重复 JIT 编译
  - 复杂度：低

- **R2C 2D half-spectrum transpose**
  - 方案：专用 half-spectrum transpose kernel，只处理 n1/2+1 列
  - 预期收益：R2C 2D transpose 数据量减少近一半
  - 参考：rocFFT `fuse_shim.cpp:R2CTrans_FuseShim`

- **Autotuning tile_size**
  - 方案：扩展 `flagfft-cli tune` 支持 2D transpose tile_size 搜索
  - 候选：{16, 32, 64}

## 优化优先级

| 优先级 | 优化项 | 类型 | 预期收益 | 复杂度 |
|--------|--------|------|---------|--------|
| P0 | 策略选择器 | 策略 | 基础设施 | 低 |
| P0 | RC 路径 | 策略 | 25-40%（大矩阵） | 中 |
| P1 | Shared Memory Transpose | Kernel | 10-30% | 中 |
| P1 | Transpose 自适应 tile_size | Kernel | 5-10% | 低 |
| P1 | Twiddle-Transpose 融合 | Kernel | 5-15% | 低 |
| P2 | FuseShim | Kernel | 20-40% | 高 |
| P2 | 2D_SINGLE 融合 kernel | 策略/Kernel | 2-3×（小矩阵） | 高 |
| P2 | Plan Cache 2D | 通用 | 消除首次延迟 | 低 |
| P3 | 临时缓冲区池化 | 内存 | 减少碎片 | 中 |
| P3 | CUDA Graph | 通用 | 20-40μs（小矩阵） | 中 |
| P3 | 双精度自适应 tile_size | Kernel | 正确性保障 | 低 |
| P3 | R2C half-spectrum transpose | 内存 | 50% transpose 减少 | 中 |
