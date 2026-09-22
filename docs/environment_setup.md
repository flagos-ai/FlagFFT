# FlagFFT 测试环境搭建

本文以已经准备好的 FlagTree 后端镜像为起点，说明 CUDA、IX、MUSA、MACA、
Ascend/NPU 和海光 BW1000/HCU 环境的依赖安装、编译和测试流程。

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

### Hygon BW1000（HCU）

使用 [FlagTree HCU 使用手册](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-hcu)
中的 HCU 3.6、DTK 26.04 镜像。宿主机需要加载 `hycu` 驱动，并将
`/dev/kfd`、`/dev/mkfd`、`/dev/dri` 和 `/opt/hyhal` 提供给容器：

```bash
docker pull harbor.baai.ac.cn/flagtree/flagtree-hcu-py310-torch2.10.0-dtk26.04-ubuntu22.04:202608-3.6-vllm0.24.0

docker run -dit --network=host --ipc=host --privileged=true \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal \
  -v /etc/localtime:/etc/localtime:ro \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  --name flagtree-dev-hcu \
  harbor.baai.ac.cn/flagtree/flagtree-hcu-py310-torch2.10.0-dtk26.04-ubuntu22.04:202608-3.6-vllm0.24.0 \
  bash
```

如果 registry 直拉受代理影响，可以按 HCU 使用手册下载同版本 tar 包后执行
`docker load -i <image-tar>`。不要用宿主机裸环境替代该容器作为 HCU 的编译或
验收环境。

本项目已验证的海光主机为 8 张 BW1000，PCI ID 为 `1d94:6320`，宿主机驱动为
`hycu`，DTK 目录为 `/opt/dtk-26.04`。当前验收使用物理卡 6、7；设置
`HIP_VISIBLE_DEVICES=6,7` 后，它们在容器内对应逻辑卡 0、1。

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

- CMake >= 3.25（`deps/libtriton_jit` 的最低版本要求）
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
通过 pip 安装。尤其不要用 CUDA 版本的 wheel 覆盖 IX、MUSA、MACA、NPU 或 HCU 镜像中
已经适配硬件的运行时。

不同镜像应分别提供对应版本，例如 CUDA 镜像提供 CUDA 版运行时，IX 镜像提供
Iluvatar 版运行时（项目曾使用 `flagtree==0.5.1+iluvatar3.1`）。

HCU 镜像需要安装与 Triton 版本匹配的 FlagTree HCU wheel；不要安装普通 CUDA
版 FlagTree 覆盖它：

```bash
python3 -m pip install \
  'flagtree===0.7.0rc2+hcu3.6' \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
python3 -m pip install -e '.[test]' --no-build-isolation
```

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

IX 的 CMake 复用 CUDA-compatible 的 CoreX SDK。先配置运行时、工具链和
Python 包路径；`COREX_HOME` 只保留一个实际路径，不能连续导出两个路径让前一个
被覆盖：

```bash
# 如果 /usr/local/corex 是指向版本化目录的软链接，也可以使用这个路径。
# 本环境实际使用版本化目录：
# export COREX_HOME=/usr/local/corex
export COREX_HOME=/usr/local/corex-4.4.0
export PATH="$COREX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$COREX_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$COREX_HOME/lib64/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
```

配置时 `CUDAToolkit_ROOT` 必须与 `COREX_HOME` 指向同一套 SDK：

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

默认按项目验证环境使用 `/opt/maca`。除了 `MACA_PATH`，MACA 的 cu-bridge、
编译架构、动态库和公共头文件路径也需要显式配置：

```bash
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export PATH="$CUCC_PATH/tools:$PATH"
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca
export TORCH_CUDA_ARCH_LIST=8.0
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="/opt/maca/include/common${CPATH:+:$CPATH}"
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
export CANN_HOME=/usr/local/Ascend/cann-9.0.0
source "$CANN_HOME/set_env.sh"
```

本项目的 NPU CMake 会从 `ASCEND_TOOLKIT_HOME` 查找 `include/`、
`lib64/libascendcl.so` 和 `lib64/libruntime.so`，并从 `CANN_HOME` 查找
架构相关的 `pkg_inc`。在当前验证用 CANN 9.0 容器中，这两个变量都指向
`/usr/local/Ascend/cann-9.0.0`；如果某台机器将 Toolkit 和 CANN 拆成了不同
目录，应分别填写实际路径：

```bash
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$CANN_HOME}"
test -f "$ASCEND_TOOLKIT_HOME/lib64/libascendcl.so"
test -f "$ASCEND_TOOLKIT_HOME/lib64/libruntime.so"
test -d "$CANN_HOME"
```

`ASCEND_HOME_PATH` 不是 FlagFFT 或 `libtriton_jit` CMake 使用的变量，不需要
依赖它完成构建。`set_env.sh` 设置的其他变量应保留，不要用 CUDA 版环境覆盖。

910B CANN 容器运行 FlagTree/Triton 和测试时，使用下面的运行时配置；其中
`TRITON_NPU_COMPILER_PATH` 必须对应同一套 CANN：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_BACKEND=torch_npu
export FLAGTREE_BACKEND=ascend
export TRITON_JIT_BACKEND=NPU
export TRITON_ASCEND_ARCH=Ascend910B4
export TRITON_NPU_COMPILER_PATH="$ASCEND_TOOLKIT_HOME/tools/bishengir/bin"
export PYTHONNOUSERSITE=1
# 当前 Python 找不到 torch_npu 时，再显式指定其包目录；CMake 默认会从
# 当前 Python 的 site-packages 自动探测它。
# export TORCH_NPU_PATH=/path/to/site-packages/torch_npu
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

### Hygon BW1000（HCU）

在 HCU 容器中加载 DTK 26.04 环境。下面的路径对应 HCU 3.6 官方镜像；如果
镜像把 DTK 安装到 `/opt/dtk-26.04`，将 `/opt/dtk` 和 `/opt/dtk/hip` 一并替换
为对应版本化路径：

```bash
source /opt/dtk/env.sh

export FLAGTREE_BACKEND=hcu
export HIP_VISIBLE_DEVICES=6,7       # 示例：使用物理卡 6、7
export FLAGFFT_EXECUTION_POLICY=native
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export PATH=/opt/dtk/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:$PATH

# DTK 自带的 rocm_smi 必须排在宿主机 /opt/hyhal 库之前。
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:${LD_LIBRARY_PATH:-}
```

检查设备、HIP/hipFFT 和 Python 运行时：

```bash
/opt/hyhal/bin/hy-smi || /opt/dtk/.hyhal/bin/hy-smi
command -v hipcc
hipcc --version
command -v rocminfo
rocminfo | grep -m4 -E 'Name:|gfx936'

python3 - <<'PY'
import importlib.metadata
import torch
import triton

print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("flagtree:", importlib.metadata.version("flagtree"))
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

BW1000 的运行时目标通常显示为 `gfx936:sramecc+:xnack-`，FlagFFT 归一化为
`hcu:gfx936:64`；`64` 是 HCU 的 wavefront 宽度。验收时应使用实际物理卡号，
不要把 `HIP_VISIBLE_DEVICES` 设置成容器内重新编号后的逻辑卡号。

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

确保已经执行上一节的 IX 环境变量配置，并使用同一个 `COREX_HOME`：

```bash
cmake -S . -B build-ix \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=IX \
  -DCUDAToolkit_ROOT="$COREX_HOME" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON \
  -DBUILD_TESTING=OFF
cmake --build build-ix -j"$(nproc)"
```

这里的 `BUILD_TESTING=OFF` 只关闭 `libtriton_jit` 自身的测试；
`FLAGFFT_BUILD_TESTS=ON` 仍然会构建 FlagFFT 的测试目标。

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
cmake_maca -S . -B build-maca \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=MACA \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DMCFFT_INCLUDE_DIR=/opt/maca/include/mcfft \
  -DMCFFT_LIB=/opt/maca/lib/libmcfft.so \
  -DMACA_PATH=/opt/maca
make_maca -C build-maca -j"$(nproc)"
```

如果 MACA SDK 不在 `/opt/maca`，需要同时修改前面的环境变量和这里的
`MCFFT_INCLUDE_DIR`、`MCFFT_LIB`、`MACA_PATH`；不要只修改其中一项。

### Ascend/NPU

编译命令默认复用上一节已加载的 CANN 环境。若从本节开始执行，先运行：

```bash
export CANN_HOME=/usr/local/Ascend/cann-9.0.0
source "$CANN_HOME/set_env.sh"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$CANN_HOME}"
```

如果 CANN 和 Toolkit 是分开安装的，把 `ASCEND_TOOLKIT_HOME` 改为实际包含
`lib64/libascendcl.so` 的 Toolkit 根目录，`CANN_HOME` 仍指向包含
`set_env.sh` 和架构目录的 CANN 根目录。

`ASCEND_OPS_FFT_ROOT` 可以指向已安装的 ops-fft 根目录，也可以指向源码构建
目录。项目 CMake 会在以下位置查找头文件和库：

```bash
test -f "$ASCEND_OPS_FFT_ROOT/include/cann_ops_fft.h" || \
test -f "$ASCEND_OPS_FFT_ROOT/include/math_libs/cann_ops_fft.h" || \
test -f "$ASCEND_OPS_FFT_ROOT/src/include/cann_ops_fft.h"
find "$ASCEND_OPS_FFT_ROOT" \( -name 'libcann_ops_fft.so' -o \
  -name 'libcann_ops_fft.a' \) -print
```

构建命令如下：

```bash
cmake -S . -B build-npu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=NPU \
  -DASCEND_TOOLKIT_HOME="$ASCEND_TOOLKIT_HOME" \
  -DASCEND_OPS_FFT_ROOT="$ASCEND_OPS_FFT_ROOT" \
  -DFLAGFFT_TRITON_JIT_SOURCE_DIR="$FLAGFFT_TRITON_JIT_SOURCE_DIR" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON \
  -DBUILD_TESTING=OFF
cmake --build build-npu -j"$(nproc)"
```

`BUILD_TESTING=OFF` 同样只关闭 `libtriton_jit` 的内部测试，不会关闭
FlagFFT 的 `ctest`/capture 目标。NPU 的统一验收入口仍然是
`tools/run_tests.py`；`ops-fft` 不支持的 API 或形状会被记录为 `Skipped`。

### Hygon BW1000（HCU）

HCU 的主构建依赖 DTK 提供的 HIP 和 hipFFT。容器可以访问 GoogleTest 源码时，
和其他后端一样直接打开 `FLAGFFT_BUILD_TESTS`；一次主构建会同时生成
`flagfft-cli`、GTest 目标和 `numpy_fft_capture`：

```bash
cmake -S . -B build-hcu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=HCU \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-hcu -j"$(nproc)"

build-hcu/flagfft-cli device-info --json
# tools/run_tests.py 使用这个 capture：
test -x build-hcu/ctest/numpy_fft_capture
```

如果 GoogleTest 下载受网络限制，才使用下面的 fallback：主构建关闭 GTest，
再用独立的 CMake 工程构建同一个 HCU native capture。这个 fallback 不会生成
GTest 目标，但不影响 `flagfft-cli` 或 `tools/run_tests.py` 的 accuracy/performance
验收：

```bash
cmake -S . -B build-hcu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=HCU \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=OFF
cmake --build build-hcu -j"$(nproc)"

cmake -S tools/numpy_fft_validation -B build-hcu-capture \
  -DFLAGFFT_SOURCE_DIR="$PWD" \
  -DFLAGFFT_BUILD_DIR="$PWD/build-hcu" \
  -DBACKEND=HCU \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hcu-capture -j"$(nproc)"
```

<!--
The NPU section above intentionally mirrors the repository's CMake discovery
rules. Keep the explicit libtriton_jit source option: the default submodule is
not a guarantee that the container has the Ascend-enabled Triton checkout.
-->

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

python3 tools/run_tests.py --combination full --gpus 6,7 \
  --build-dir build-hcu \
  --capture-bin build-hcu/ctest/numpy_fft_capture \
  --output-dir ../results/$(date +%Y%m%d_%H%M%S)_hcu_acceptance \
  --timeout 600 --warmup 3 --iters 30 --color never -v
```

HCU 的 `--gpus` 使用宿主机物理卡号；runner 会为每个 worker 设置单卡
`HIP_VISIBLE_DEVICES`。结果必须写入当前工作区父目录的 `results/` 时间戳目录，
不要写入 `FlagFFT-dev/results/`。如果使用远端 staging，完成后再将结果回收到
主机侧约定目录。

Python codegen 测试：

```bash
pytest tests/python/ -v
```

C++ 测试：

```bash
cd build-cuda       # 或 build-ix/build-musa/build-maca/build-npu/build-hcu
ctest --output-on-failure
```

如果采用网络受限的 fallback，上面的 `--capture-bin` 改为
`build-hcu-capture/numpy_fft_capture`。HCU 的完整工具验收本身不依赖 GTest；
GTest 只负责 C++ 单元/集成测试。

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

### HCU 找不到 hip/hipFFT 或设备

确认已经在 HCU 容器内执行 `source /opt/dtk/env.sh`，并设置了
`ROCM_PATH=/opt/dtk`、`HIP_PATH=/opt/dtk/hip`；同时检查容器具有
`/dev/kfd`、`/dev/mkfd`、`/dev/dri` 和 `/opt/hyhal`。宿主机的 CANN 8.2 或裸
HIP runtime 不能作为 HCU 验收基准。

### HCU 出现 UnicodeDecodeError

HCU Triton driver 读取生成文件时依赖 UTF-8 locale。补上以下变量后重新启动
相关进程：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### HCU 出现 shared-memory 超限或退出时 rocm-smi 崩溃

BW1000 的 dynamic shared memory 上限为 64 KiB，保持：

```bash
export FLAGFFT_EXECUTION_POLICY=native
```

不要用 `legacy` 作为 BW1000 的默认验收策略；它可能为 FP64 四步 kernel 选择
超过设备上限的 pack。若出现 `librocm_smi64.so` 冲突，确保
`/opt/dtk/.hyhal/rocm_smi/lib` 位于 `/opt/hyhal/lib` 之前，避免混用两套
`rocm_smi` 运行库。

### IX 误用了 CUDA 环境

IX 必须使用 CoreX SDK 和 Iluvatar 适配版 FlagTree/Triton。不要只把
`-DBACKEND=CUDA` 改成 `-DBACKEND=IX` 后继续使用 CUDA 镜像。
