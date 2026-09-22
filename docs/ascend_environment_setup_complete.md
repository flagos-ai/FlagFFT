# Ascend/CANN 9.0 构建与测试环境

本文给出 FlagFFT 在 Ascend 910B、CANN 9.0 容器中的完整流程，覆盖进入容器、依赖准备、CANN/torch_npu/Triton 环境、ops-fft、CMake 构建、边界测试和完整验收。

## 1. 环境边界和路径

Ascend kernel 必须在 CANN 9.0 容器中编译和验证。主机上的 CANN 8.x 只用于诊断，不能作为 kernel 验证基准。容器需要包含 CANN 9.0、torch_npu 和 Ascend 适配的 Triton/libtriton_jit，并能访问 `/dev/davinci*`。

下面的路径可以按实际环境修改：

```bash
export SRC=/workspace/FlagFFT-dev
export BUILD=/workspace/FlagFFT-build-ascend
export TRITON_JIT=$SRC/deps/libtriton_jit
export OPSFFT=/tmp/ops-fft
export RESULTS=/workspace/results/$(date +%Y%m%d_%H%M%S)_ascend
```

本次实际远端验证使用：

```text
源码：/home/FlagFFT-dev-ascend-dev-validation
构建：/home/FlagFFT-dev-ascend-dev-validation-build
libtriton_jit：/home/flagfft-npu/deps/libtriton_jit
ops-fft：/tmp/ops-fft
```

## 2. 进入容器

已有平台容器可直接进入：

```bash
docker exec -it flagsparse bash
```

自行启动时应使用包含 CANN9、torch_npu 和 Ascend Triton 的镜像，并映射 Ascend 设备；镜像名和设备节点按服务器实际情况替换：

```bash
docker run --rm -it --ipc=host --network=host --privileged \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  -v /rjs/llb/fft-dev:/workspace \
  -w /workspace/FlagFFT-dev \
  <flagtree-ascend-cann9-image> bash
```

进入后检查：

```bash
test -f /usr/local/Ascend/cann-9.0.0/set_env.sh
command -v npu-smi
python3 --version
npu-smi info
```

## 3. 源码和 Python 依赖

```bash
cd "$SRC"
git submodule update --init --recursive
test -f "$TRITON_JIT/CMakeLists.txt"
python3 -m pip install -e '.[test]'   # 仅在镜像没有项目测试依赖时执行
```

PyTorch、torch_npu、Triton 和 FlagTree 应由镜像提供。不要用普通 CUDA wheel 覆盖 NPU 镜像中的运行时。

## 4. CANN9、torch_npu 和 Triton 环境

以下是实际使用的环境变量：

```bash
export CANN_HOME=/usr/local/Ascend/cann-9.0.0
source "$CANN_HOME/set_env.sh"
export ASCEND_TOOLKIT_HOME="$CANN_HOME"
export TORCH_NPU_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_BACKEND=torch_npu
export FLAGTREE_BACKEND=ascend
export TRITON_JIT_BACKEND=NPU
export TRITON_ASCEND_ARCH=Ascend910B4
export TRITON_NPU_COMPILER_PATH="$ASCEND_TOOLKIT_HOME/tools/bishengir/bin"
export PYTHONNOUSERSITE=1
export ASCEND_OPS_FFT_ROOT="$OPSFFT"
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$BUILD/triton-cache"
mkdir -p "$TRITON_CACHE_DIR"
```

检查工具链和 Python 运行时：

```bash
test -f "$ASCEND_TOOLKIT_HOME/lib64/libascendcl.so"
test -f "$ASCEND_TOOLKIT_HOME/lib64/libruntime.so"
test -x "$TRITON_NPU_COMPILER_PATH/bishengir-compile"
python3 - <<'PY'
import torch
import torch_npu
import triton
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__file__)
print("triton:", triton.__version__)
PY
```

CANN9 镜像中的 Python 测试按 `torch -> torch_npu -> triton` 顺序导入，避免首次 pytest 导入时的循环初始化问题。

## 5. 准备独立的 ops-fft

ops-fft 是 CANN 的独立参考库，不包含在 FlagFFT 源码中，也不是通过 pip 安装。它需要单独准备或由验证镜像提供，并使用与 CANN9 兼容的构建版本。

```bash
export ASCEND_OPS_FFT_ROOT=/tmp/ops-fft
test -f "$ASCEND_OPS_FFT_ROOT/include/cann_ops_fft.h" || \
test -f "$ASCEND_OPS_FFT_ROOT/include/math_libs/cann_ops_fft.h" || \
test -f "$ASCEND_OPS_FFT_ROOT/src/include/cann_ops_fft.h"
find "$ASCEND_OPS_FFT_ROOT" \( -name 'libcann_ops_fft.so' -o -name 'libcann_ops_fft.a' \) -print
```

如果目录不存在，先按 `cann/ops-fft` 项目的 CANN9 构建说明完成它，再继续。没有 ops-fft 可以编译部分 FlagFFT 目标，但不能运行本项目的 Ascend 平台对标性能测试。

## 6. CMake 配置和编译

这是实际成功执行的配置。注意 `ASCEND_TOOLKIT_HOME` 通过环境变量提供，实际命令没有把它作为 `-D` 参数传入：

```bash
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBACKEND=NPU \
  -DFLAGFFT_TRITON_JIT_SOURCE_DIR="$TRITON_JIT" \
  -DASCEND_OPS_FFT_ROOT="$ASCEND_OPS_FFT_ROOT" \
  -DFLAGFFT_BUILD_CLI=ON \
  -DFLAGFFT_BUILD_TESTS=ON \
  -DTRITON_JIT_USE_EXTERNAL_JSON=OFF \
  -DTRITON_JIT_USE_EXTERNAL_FMTLIB=OFF \
  -DTRITON_JIT_USE_EXTERNAL_PYBIND11=ON \
  -DTRITON_JIT_BUILD_OPERATORS=OFF \
  -DBUILD_TESTING=OFF \
  -DFETCHCONTENT_SOURCE_DIR_JSON=/home/flagfft-npu-build/_deps/json-src \
  -DFETCHCONTENT_SOURCE_DIR_FMT=/home/flagfft-npu-build/_deps/fmt-src \
  -Dpybind11_DIR=/usr/local/python3.11.15/lib/python3.11/site-packages/pybind11/share/cmake/pybind11

cmake --build "$BUILD" --parallel 4
```

最后三个路径是本次容器的离线依赖缓存。没有这些缓存时，可以删除对应参数并确保网络或本地依赖可用。

构建成功后应存在：

```bash
test -f "$BUILD/libflagfft.so"
test -x "$BUILD/flagfft-cli"
test -x "$BUILD/ctest/numpy_fft_capture"
test -x "$BUILD/ctest/test_npu_real_edges"
```

## 7. 运行基础测试

```bash
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/ctest/test_npu_real_edges"
```

Python codegen 测试使用同一进程预先导入 NPU 运行时：

```bash
cd "$SRC"
python3 - <<'PY'
import pytest
import torch
import torch_npu
import triton
raise SystemExit(pytest.main(["-q", "tests/python/test_stockham_codegen.py",
                              "tests/python/test_stockham_vector_numpy.py",
                              "tests/python/test_performance_gate.py"]))
PY
```

## 8. 运行 Ascend 对标

每次使用新的空结果目录。单卡示例：

```bash
unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
export RESULTS=/workspace/results/$(date +%Y%m%d_%H%M%S)_ascend_ct_single

python3 "$SRC/tools/run_tests.py" \
  --ops 1d_ct_single_c2c,1d_ct_single_c2r,1d_ct_single_r2c \
  --gpus 1 \
  --build-dir "$BUILD" \
  --output-dir "$RESULTS" \
  --timeout 1200 --warmup 10 --iters 100 --color never -v
```

三个 CT single 算子的 44-case 门槛：

```bash
python3 "$SRC/tools/check_performance_gate.py" "$RESULTS" \
  --threshold 0.8 --expected-count 44 --ascend-ct-single
```

36-op 全量不传 `--ops`，使用 `--combination full`：

```bash
export RESULTS=/workspace/results/$(date +%Y%m%d_%H%M%S)_ascend_acceptance36
python3 "$SRC/tools/run_tests.py" \
  --combination full --gpus 1,2,3 \
  --build-dir "$BUILD" --output-dir "$RESULTS" \
  --timeout 1200 --warmup 10 --iters 100 --color never -v
```

多卡运行前必须用 `npu-smi info` 确认每张卡为空闲。不要使用 `--gpus all` 绕过设备检查；性能结果会受同卡任务、主机负载和 Triton cache 状态影响。

## 9. 结果和常见问题

结果写入工作区父目录的 `results/<timestamp>_...`，不要写入源码 worktree 内。重点文件是 `manifest.json`、`incremental.csv`、`summary.json`、`runner.log` 和设备监控日志。

`Passed` 只表示该阶段成功执行，不等于 speedup 达到 0.8。平台库不支持的 API/形状应记为 `Skipped`，不能当作性能通过。

- 找不到 AscendCL/runtime：确认执行了 CANN9 的 `set_env.sh`。
- 找不到 `cann_ops_fft.h` 或 `libcann_ops_fft`：检查 `ASCEND_OPS_FFT_ROOT`。
- torch/Triton 循环导入：按 `torch -> torch_npu -> triton` 顺序在同一 Python 进程中启动 pytest。
- CMake 下载依赖失败：准备 JSON、fmt、pybind11 离线缓存，或删除对应缓存参数并确保网络可用。
- 大 batch 超时：保留 Timeout 记录，不把未完成性能项当作通过。
- 结果目录非空：每次创建新的时间戳目录。

