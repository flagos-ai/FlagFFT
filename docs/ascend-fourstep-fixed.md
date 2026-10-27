# 修复版编译器上的 Ascend four-step FFT 实验

本实验针对原先因结构化置换 lowering 而停下的重写路线，验收条件是
FP32 C2C forward、N=2048、batch=2048，完整自然序 FFT 的设备时间
低于 0.15 µs/变换。它是独立 kernel 实验，不修改产品默认计划器，
也不代表完整 1D/2D/3D 算子矩阵已达到 SiP 性能门槛。

实现和运行入口是 [ascend_fourstep_probe.py](../tools/ascend_fourstep_probe.py)。

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
