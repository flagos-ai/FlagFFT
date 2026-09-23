# Ascend/CANN 9.0 构建与测试环境

本文是 `baai-ascend` 上 FlagFFT 的可复现 910B 测试流程。FlagFFT 源码、构建
目录、FlagTree 源码和测试结果均保存在宿主机；容器只用于安装外部依赖、编译和
运行。CANN 9.0、`torch_npu` 和 FlagTree 的 Ascend 3.5 版本必须保持一致。

FlagTree 的安装依据其官方
[Ascend 3.5 使用手册](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend)：
检出 `triton_v3.5.x` 分支，在 CANN 9.0 镜像中从源码安装；不要安装普通 CUDA
版 `triton`。

## 1. 宿主机准备工作区

以下命令在 `baai-ascend` 宿主机执行。当前主机已有一个 FlagFFT Git 工作区，
用它创建新的测试 worktree；如果使用其他仓库路径，只需替换 `REPO`。

```bash
ssh baai-ascend

HOST_ROOT=/root/gcx
REPO="$HOST_ROOT/FlagFFT-dev-ascend-dev-validation"
SRC_HOST="$HOST_ROOT/FlagFFT-dev-ascend"
BUILD_HOST="$HOST_ROOT/FlagFFT-build-ascend"
FLAGTREE_HOST="$HOST_ROOT/FlagTree-ascend"
RESULTS_HOST="$HOST_ROOT/results"

# SOURCE_REF 应是已经从本地主机同步的、逻辑完整的 FlagFFT 提交或分支。
SOURCE_REF=codex/ascend-env-setup
git -C "$REPO" fetch origin "$SOURCE_REF"
test ! -e "$SRC_HOST"
git -C "$REPO" worktree add --detach "$SRC_HOST" "origin/$SOURCE_REF"
git -C "$SRC_HOST" submodule update --init --recursive
mkdir -p "$BUILD_HOST" "$RESULTS_HOST"

# FlagTree 使用官方 Ascend 3.5 分支，独立放在宿主机，避免容器重建后丢失。
if test ! -d "$FLAGTREE_HOST/.git"; then
  git clone --branch triton_v3.5.x --single-branch \
    https://github.com/flagos-ai/FlagTree.git "$FLAGTREE_HOST"
fi
git -C "$FLAGTREE_HOST" status --short --branch
```

如果远端没有可用的 GitHub 远程，先把本地已提交的 FlagFFT 分支通过 Git
push/fetch 同步；不要在容器内直接修改 FlagFFT 源码。

## 2. 创建 `flagfft-ascend` 容器

当前 `baai-ascend` 使用的镜像是：

```text
harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base
```

这个基础镜像提供 CANN 9.0、PyTorch、`torch_npu` 和常用 Python 构建依赖，
但当前实际验证表明它不预装 FlagTree/Triton；下一节会从官方 FlagTree 源码安装。

```bash
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base

docker run -dit -u 0 --user=root \
  --network=host --pid=host --ipc=host --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /etc/localtime:/etc/localtime:ro \
  -v /data:/data \
  -v "$SRC_HOST:/home/FlagFFT-dev-ascend" \
  -v "$BUILD_HOST:/home/FlagFFT-build-ascend" \
  -v "$FLAGTREE_HOST:/home/FlagTree-ascend" \
  -v "$RESULTS_HOST:/home/results" \
  -w /home/FlagFFT-dev-ascend \
  --name flagfft-ascend "$IMAGE" bash

docker exec -it flagfft-ascend bash
```

检查容器和设备：

```bash
test -f /usr/local/Ascend/cann-9.0.0/set_env.sh
command -v npu-smi
python3 --version
npu-smi info
```

## 3. 设置路径和 CANN/NPU 环境

以下命令在 `flagfft-ascend` 容器内执行。主机 shell 的变量不会自动传入
`docker exec`。

```bash
export SRC=/home/FlagFFT-dev-ascend
export BUILD=/home/FlagFFT-build-ascend
export FLAGTREE=/home/FlagTree-ascend
export TRITON_JIT="$SRC/deps/libtriton_jit"
export OPSFFT="$SRC/deps/ops-fft"
export CANN_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME="$CANN_HOME"

# CANN 镜像有两个常见入口；优先使用镜像官方入口，缺失时回退到 CANN9。
if test -f /usr/local/Ascend/ascend-toolkit/set_env.sh; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  source "$CANN_HOME/set_env.sh"
fi

export TORCH_NPU_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRITON_BACKEND=torch_npu
export FLAGTREE_BACKEND=ascend
export TRITON_JIT_BACKEND=NPU
export TRITON_ASCEND_ARCH=Ascend910B4
export TRITON_NPU_COMPILER_PATH="$ASCEND_TOOLKIT_HOME/tools/bishengir/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SRC/python:$PYTHONPATH"
export TRITON_CACHE_DIR="$BUILD/triton-cache"
mkdir -p "$TRITON_CACHE_DIR"

test -f "$ASCEND_TOOLKIT_HOME/lib64/libascendcl.so"
test -f "$ASCEND_TOOLKIT_HOME/lib64/libruntime.so"
test -x "$TRITON_NPU_COMPILER_PATH/bishengir-compile"
```

## 4. 安装 FlagTree（官方源码）

官方 Ascend 3.5 流程要求先安装系统依赖和 FlagTree 的 Python 依赖，再从
`triton_v3.5.x` 根目录安装。网络可用时，FlagTree 的构建脚本会自动获取其
编译依赖。

```bash
apt-get update
apt-get install -y \
  git cmake ninja-build build-essential \
  zlib1g zlib1g-dev libxml2 libxml2-dev nlohmann-json3-dev \
  sqlite3 libsqlite3-dev pkg-config ca-certificates

cd "$FLAGTREE"
git status --short --branch
python3 -m pip install -r python/requirements.txt

export FLAGTREE_BACKEND=ascend
# baai-ascend 上 GitHub 访问不稳定时，使用宿主机预先放入 FlagTree
# 工作区的源码依赖；没有该目录时，删除这个 if 块，setup_helper 会按
# 官方源码中的版本约束从 GitHub 获取依赖。
if test -d "$FLAGTREE/offline-deps"; then
  git config --global "url.${FLAGTREE}/offline-deps/flir.insteadOf" \
    https://github.com/flagos-ai/flir.git
  git config --global "url.${FLAGTREE}/offline-deps/FlagPrism.insteadOf" \
    https://github.com/flagos-ai/FlagPrism.git
  git config --global "url.${FLAGTREE}/offline-deps/AscendNPU-IR.insteadOf" \
    https://github.com/flagos-ai/FlagTree-AscendNPU-IR.git
fi

# setup_helper 会准备 Ascend LLVM 缓存和第三方源码。必须先让它执行，
# 再把该缓存中的 clang/clang++ 放到 PATH；否则 CMake 可能报找不到 clang。
python3 -c 'import python.setup_tools.setup_helper'
LLVM_ROOT="$(find /root/.flagtree/ascend -mindepth 1 -maxdepth 1 \
  -type d -name "llvm-*" -print -quit)"
test -n "$LLVM_ROOT" && test -x "$LLVM_ROOT/bin/clang"
test -x "$LLVM_ROOT/bin/clang++"
export PATH="$LLVM_ROOT/bin:$PATH"
export CC="$LLVM_ROOT/bin/clang"
export CXX="$LLVM_ROOT/bin/clang++"

MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v

python3 -m pip show flagtree
cd /tmp
python3 -c 'import triton; print("triton:", triton.__path__)'
```

当前实测的源码依赖版本如下；如果没有使用 `offline-deps`，则让
`setup_helper` 从官方仓库自动拉取，不要随意替换为其他分支：

```bash
if test -d "$FLAGTREE/offline-deps"; then
  git -C "$FLAGTREE/offline-deps/flir" rev-parse HEAD
  # 516ad3110cbf6112bbed8f927ae606c627a193f2
  git -C "$FLAGTREE/offline-deps/FlagPrism" rev-parse HEAD
  # 8541d6761805bd9d3c54d1bc53da4a1939ffd6c2
  git -C "$FLAGTREE/offline-deps/AscendNPU-IR" rev-parse HEAD
  # a205c9574907907d608da6029403415ca2f98d3c
fi
```

如果构建中途失败，修正依赖或 LLVM 环境后，回到 `$FLAGTREE` 重跑
`MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v`。
不要执行 `pip install triton`；它会覆盖 Ascend 适配版 Triton。

## 5. 安装 FlagFFT 的 Python 依赖

```bash
cd "$SRC"
git submodule update --init --recursive
test -f "$TRITON_JIT/CMakeLists.txt"
# 基础镜像已经带有 numpy/PyYAML/pytest；关闭 build isolation，避免 pip
# 为 setuptools 创建临时环境时访问外部 PyPI。
python3 -m pip install --no-index --no-build-isolation -e "${SRC}[test]"
python3 - <<'PY'
import torch
import torch_npu
import triton
import flagtree

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__file__)
print("triton:", triton.__version__)
print("flagtree:", flagtree.__file__)
PY
```

必须按 `torch -> torch_npu -> triton` 的顺序在同一进程导入；
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 用于避免 PyTorch 自动加载导致的循环导入。

## 6. 在 `FlagFFT/deps` 下构建 ops-fft

`ops-fft` 是 CANN 独立参考库，不通过 pip 安装。这次固定放在
`FlagFFT` 工作区的 `deps/ops-fft`，该目录位于宿主机
`/root/gcx/FlagFFT-dev-ascend/deps/ops-fft`，容器重启后仍然保留。

```bash
mkdir -p "$SRC/deps"
if test ! -d "$OPSFFT/.git"; then
  git clone https://gitcode.com/cann/ops-fft.git "$OPSFFT"
else
  git -C "$OPSFFT" fetch --all --prune
fi
OPSFFT_COMMIT=f2ed13ec7dc9a5ee1d60bc307daf0b92062d7309
git -C "$OPSFFT" checkout --detach "$OPSFFT_COMMIT"

cd "$OPSFFT"
bash build.sh --soc=Ascend910B -j16
cmake --install build --prefix build_out

export ASCEND_OPS_FFT_ROOT="$OPSFFT/build_out/ops_fft"
test -f "$ASCEND_OPS_FFT_ROOT/include/cann_ops_fft.h"
test -f "$ASCEND_OPS_FFT_ROOT/lib64/libcann_ops_fft.so"
find "$ASCEND_OPS_FFT_ROOT" \( -name 'libcann_ops_fft.so' -o \
  -name 'libcann_ops_fft.a' \) -print
```

## 7. 配置和编译 FlagFFT

```bash
cd "$SRC"
PYBIND11_CMAKE_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')

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
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"

cmake --build "$BUILD" --parallel 4

test -f "$BUILD/libflagfft.so"
test -x "$BUILD/flagfft-cli"
test -x "$BUILD/ctest/numpy_fft_capture"
test -x "$BUILD/ctest/test_npu_real_edges"
```

## 8. 基础测试

```bash
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/ctest/test_npu_real_edges"

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

## 9. Ascend 对标测试

先用 `npu-smi info` 确认设备空闲。`--gpus` 使用宿主机物理 NPU 编号；下面以
空闲的 1 号卡为例，不要在有其他进程时照抄设备号。

```bash
unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
export RESULTS=/home/results/$(date +%Y%m%d_%H%M%S)_ascend_ct_single

python3 "$SRC/tools/run_tests.py" \
  --ops 1d_ct_single_c2c,1d_ct_single_c2r,1d_ct_single_r2c \
  --gpus 1 \
  --build-dir "$BUILD" \
  --output-dir "$RESULTS" \
  --timeout 1200 --warmup 10 --iters 100 --color never -v
```

三个 CT single 算子的门槛检查：

```bash
python3 "$SRC/tools/check_performance_gate.py" "$RESULTS" \
  --threshold 0.8 --expected-count 44 --ascend-ct-single
```

36-op 全量测试应使用新的结果目录，并只选择确认空闲的卡：

```bash
export RESULTS=/home/results/$(date +%Y%m%d_%H%M%S)_ascend_acceptance36
python3 "$SRC/tools/run_tests.py" \
  --combination full --gpus 1,2 \
  --build-dir "$BUILD" --output-dir "$RESULTS" \
  --timeout 1200 --warmup 10 --iters 100 --color never -v
```

结果实际落在宿主机 `/root/gcx/results/<timestamp>_*`，不要写入源码 worktree
内。`Passed` 只表示该阶段执行成功，不等于 speedup 达到 0.8；参考库不支持
的 API/形状应记录为 `Skipped`。

## 10. 常见问题

- 找不到 AscendCL/runtime：重新执行本节的 CANN `set_env.sh`，并检查
  `ASCEND_TOOLKIT_HOME` 是否为 `/usr/local/Ascend/cann-9.0.0`。
- 找不到 `triton` 或导入循环：确认从官方 FlagTree `triton_v3.5.x` 源码安装，
  设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不要安装普通 `triton`。
- FlagTree 构建找不到 `clang`：重新执行
  `python3 -c 'import python.setup_tools.setup_helper'`，从
  `/root/.flagtree/ascend/llvm-*` 找到 LLVM，并重新设置 `PATH`、`CC`、`CXX` 后再重试。
- FlagTree 构建缺少 GitHub 依赖：确认 `$FLAGTREE/offline-deps` 中的三个源码仓库
  存在，或恢复网络后删除本节的 Git URL 映射，让官方 `setup_helper` 自动获取。
- 找不到 `cann_ops_fft.h` 或 `libcann_ops_fft`：确认 `ops-fft` 已在
  `$SRC/deps/ops-fft` 中以 `Ascend910B` 构建，并重新设置
  `ASCEND_OPS_FFT_ROOT`。
- 结果目录非空：每次使用新的秒级时间戳目录；不要覆盖以前的结果。

## 11. 本次实测记录（2026-09-22）

本流程在 `baai-ascend` 的 `flagfft-ascend` 容器中完成验证。FlagTree 使用官方
仓库的 `triton_v3.5.x` 分支（提交 `7aa36854464b719cb3c8f097e8e9810fe9be4d0a`），
通过源码安装得到 `flagtree 0.7.0+ascend.git7aa36854`、Triton 3.5.1；没有安装
普通 Triton wheel。`ops-fft` 使用提交 `f2ed13ec7dc9a5ee1d60bc307daf0b92062d7309`，
安装根目录为 `$SRC/deps/ops-fft/build_out/ops_fft`。

验证结果：

- CMake Release 构建成功，`libflagfft.so`、CLI 和 NPU 测试目标均生成。
- `ctest --test-dir "$BUILD" --output-on-failure`：1/1 通过。
- 三个 Python 测试文件：53 passed。
- `1d_ct_single_c2c` Ascend smoke：精度 1/1、性能 1/1 通过，speedup 3.1468。

smoke 结果保存在宿主机 `/root/gcx/results/20260922_115027_ascend_smoke_c2c`，
不要把结果目录放入 FlagFFT worktree。
