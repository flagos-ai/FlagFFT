# FlagFFT 测试环境搭建

本文以已经准备好的 FlagTree 后端镜像为起点，说明 CUDA、IX、MUSA、MACA
和 Ascend/NPU 环境的依赖安装、编译和测试流程。

## 约定

- 以下命令中的镜像名使用占位符，例如 `<flagtree-cuda-image>`，替换成实际的 FlagTree 镜像名。
- 代码在主机 worktree 中维护，容器只用于安装依赖、编译和运行测试。
- 主机 worktree：`/rjs/llb/fft-dev/FlagFFT-dev`
- 容器内路径：`/workspace/FlagFFT-dev`
- 所有后端都需要 Python 3.10 及以上，推荐 Python 3.12。
- 不要额外安装官方 `triton` 覆盖镜像中的 Triton/TLE。FlagTree 镜像应提供与后端匹配的 Triton 运行时。

## 1. 启动 FlagTree 基础镜像

### CUDA

```bash
docker run --gpus all --ipc=host --network=host \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  -it <flagtree-cuda-image> bash
```

镜像需要包含 CUDA Toolkit、驱动可见性和与 CUDA 匹配的 FlagTree/Triton。

### IX（Iluvatar/Tianshu）

```bash
docker run --ipc=host --network=host \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  -it <flagtree-ix-image> bash
```

按 IX 镜像说明添加设备映射和运行时参数。镜像需要包含 CoreX SDK、ixFFT
以及 Iluvatar 版本的 FlagTree/Triton。当前项目示例使用：

```text
flagtree==0.5.1+iluvatar3.1
```

### MUSA

```bash
docker run --ipc=host --network=host \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  -it <flagtree-musa-image> bash
```

按 MUSA 镜像说明添加设备映射。镜像需要包含 MUSA SDK、muFFT 和 MUSA
适配版 FlagTree/Triton。

### MACA（MetaX）

```bash
docker run --ipc=host --network=host \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  -it <flagtree-maca-image> bash
```

按 MetaX 镜像说明添加设备映射。本文假设 MACA SDK 安装在
`/opt/maca`，并且镜像提供 `cmake_maca` / `make_maca` 包装命令。

### Ascend/NPU

```bash
docker run --ipc=host --network=host \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  -it <flagtree-ascend-image> bash
```

按 Ascend 镜像说明添加 `/dev/davinci*` 等设备映射。镜像需要包含 CANN
9.0 工具链、Ascend runtime，以及 Ascend 适配版 FlagTree/Triton。

## 2. 安装系统依赖

在容器内执行：

```bash
apt-get update
apt-get install -y \
  git cmake ninja-build build-essential \
  sqlite3 libsqlite3-dev \
  pkg-config ca-certificates
```

要求：

- CMake >= 3.18
- 支持 C++20 的 GCC 11+ 或 Clang 14+
- SQLite3 开发库（`libsqlite3-dev`）

如果基础镜像已经提供这些工具，不需要重复安装。

## 3. 初始化子模块

```bash
cd /workspace/FlagFFT-dev
git submodule update --init --recursive
```

必须存在：

```text
deps/libtriton_jit
```

## 4. 安装 Python 依赖

先确认 Python：

```bash
python3 --version
python3 -m pip --version
```

安装项目测试依赖：

```bash
python3 -m pip install -e '.[test]'
```

该命令安装 `numpy`、`PyYAML` 和 `pytest`。

安装构建辅助库：

```bash
python3 -m pip install ninja cmake nanobind pybind11
```

### 各后端的 Python 运行时

PyTorch 和 FlagTree/Triton 已由各后端 FlagTree 基础镜像预装，不需要再次
通过 pip 安装。尤其不要用 CUDA 版本的 wheel 覆盖 IX、MUSA 或 MACA 镜像中
已经适配硬件的运行时。

不同镜像应分别提供对应版本，例如 CUDA 镜像提供 CUDA 版运行时，IX 镜像提供
Iluvatar 版运行时（项目曾使用 `flagtree==0.5.1+iluvatar3.1`）。

检查 Python 依赖：

```bash
python3 - <<'PY'
import numpy
import torch
import yaml
import flagtree

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("flagtree:", getattr(flagtree, "__version__", "installed"))
print("torch device available:", torch.cuda.is_available())
PY
```

## 5. 后端 SDK 检查

### CUDA

FlagFFT 的 CLI 和测试需要 CUDA Toolkit、CUDA runtime、CUDA driver 和 cuFFT：

```bash
which nvcc
nvcc --version
test -f "$CUDA_HOME/include/cufft.h" || test -f /usr/local/cuda/include/cufft.h
nvidia-smi
```

如果 CUDA 安装在非默认路径，配置时指定：

```text
-DCUDAToolkit_ROOT=/path/to/cuda
```

### IX

IX 的 CMake 复用 CUDA-compatible 的 CoreX SDK，配置时指定 CoreX 路径：

```bash
export COREX_HOME=/usr/local/corex
```

例如：

```text
-DCUDAToolkit_ROOT=/usr/local/corex-4.4.0
```

测试参考库是 ixFFT。IX 当前验收策略跳过 FP64 的 `Z2Z`、`Z2D` 和 `D2Z`。

### MUSA

默认 SDK 路径是 `/usr/local/musa`，也可以显式指定：

```bash
export MUSA_HOME=/usr/local/musa
```

测试需要：

```text
$MUSA_HOME/include/musa_runtime_api.h
$MUSA_HOME/include/mufft.h
$MUSA_HOME/lib64/libmufft.so    # 或 $MUSA_HOME/lib/libmufft.so
```

### MACA

默认按项目验证环境使用 `/opt/maca`：

```bash
export MACA_PATH=/opt/maca
```

测试需要：

```text
$MACA_PATH/include/mcfft/mcfft.h
$MACA_PATH/lib/libmcfft.so
$MACA_PATH/lib/libmcruntime.so
```

MACA 构建和运行应使用 MetaX 提供的 `cmake_maca` / `make_maca` 包装命令。

### Ascend/NPU

必须使用 CANN 9.0 环境，并加载 CANN 环境变量：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

FlagFFT 的 CLI 和测试还需要编译好的 CANN `ops-fft` 参考库：

```bash
export ASCEND_OPS_FFT_ROOT=/path/to/ops-fft
test -f "$ASCEND_OPS_FFT_ROOT/include/cann_ops_fft.h"
find "$ASCEND_OPS_FFT_ROOT" -name 'libcann_ops_fft.so' -o \
  -name 'libcann_ops_fft.a'
```

`ASCEND_OPS_FFT_ROOT` 必须指向包含 `cann_ops_fft.h` 和
`libcann_ops_fft` 的已构建 `ops-fft` 目录。

## 6. 编译

### CUDA

```bash
cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=CUDA \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-cuda -j"$(nproc)"
```

### IX

```bash
cmake -B build-ix \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=IX \
  -DCUDAToolkit_ROOT=/usr/local/corex-4.4.0 \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-ix -j"$(nproc)"
```

### MUSA

```bash
cmake -B build-musa \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=MUSA \
  -DMUSA_HOME="${MUSA_HOME:-/usr/local/musa}" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-musa -j"$(nproc)"
```

### MACA

```bash
cmake_maca -B build-maca \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=MACA \
  -DMACA_PATH="${MACA_PATH:-/opt/maca}" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
make_maca -C build-maca -j"$(nproc)"
```

### Ascend/NPU

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_OPS_FFT_ROOT=/path/to/ops-fft

cmake -B build-npu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=NPU \
  -DASCEND_OPS_FFT_ROOT="$ASCEND_OPS_FFT_ROOT" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-npu -j"$(nproc)"
```

如果 MetaX 镜像仍使用普通 CMake/Make 包装方式，则将最后两条替换为：

```bash
cmake -B build-maca ...
cmake --build build-maca -j"$(nproc)"
```

## 7. 运行测试

先执行不占用设备的矩阵检查：

```bash
python3 tools/run_tests.py --dry-run
```

然后运行完整验收。需要将 `--gpus` 改成当前容器可见的设备编号：

```bash
python3 tools/run_tests.py --build-dir build-cuda --gpus 0 \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_cuda_acceptance

python3 tools/run_tests.py --build-dir build-ix --gpus 0 \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_ix_acceptance

python3 tools/run_tests.py --build-dir build-musa --gpus 0 \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_musa_acceptance

python3 tools/run_tests.py --build-dir build-maca --gpus 0 \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_maca_acceptance

python3 tools/run_tests.py --build-dir build-npu --gpus 0 \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_npu_acceptance
```

Python codegen 测试：

```bash
pytest tests/python/ -v
```

C++ 测试：

```bash
cd build-cuda       # 或 build-ix/build-musa/build-maca/build-npu
ctest --output-on-failure
```

## 8. 常见问题

### CMake 找不到 SQLite3

```bash
apt-get install -y sqlite3 libsqlite3-dev
```

### CMake 找不到 PyTorch ABI

确认 CMake 使用的 Python 与安装依赖的 Python 是同一个：

```bash
which python3
python3 -c 'import torch; print(torch.__file__)'
cmake -B build -DPython_EXECUTABLE="$(which python3)" ...
```

### MUSA 找不到 muFFT

确认 `MUSA_HOME` 指向 SDK 根目录，并检查 `libmufft.so` 是否位于
`lib64` 或 `lib` 下。

### MACA 找不到 mcFFT

确认 `MACA_PATH` 指向 SDK 根目录，并检查 `include/mcfft/mcfft.h`、
`lib/libmcfft.so` 和 `lib/libmcruntime.so`。

### Ascend 找不到 ops-fft

确认已执行 CANN 9.0 的 `set_env.sh`，并且
`ASCEND_OPS_FFT_ROOT` 下存在 `cann_ops_fft.h` 和 `libcann_ops_fft`。

Ascend 910B 当前验收支持 FP32 的 C2C、R2C、C2R；FP64 的 Z2Z、Z2D、D2Z
会按平台策略跳过。ops-fft 对 1D、2D 和 3D 的支持范围也不是完整矩阵，
测试报告会将不支持的参考库阶段标记为 `Skipped`。

### IX 误用了 CUDA 环境

IX 必须使用 CoreX SDK 和 Iluvatar 适配版 FlagTree/Triton。不要只把
`-DBACKEND=CUDA` 改成 `-DBACKEND=IX` 后继续使用 CUDA 镜像。
