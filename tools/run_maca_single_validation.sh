#!/usr/bin/env bash
# Run inside the MACA development container. Source trees are synchronized from
# committed host worktrees; this script only configures, builds and measures.
set -euo pipefail
if (( $# < 4 )); then
  echo "usage: $0 {build|run} SOURCE BUILD OUTPUT [run_tests.py arguments...]" >&2
  exit 2
fi
mode=$1; src=$(realpath "$2"); build=$(realpath -m "$3"); out=$(realpath -m "$4")
shift 4
mkdir -p "$out"
export CUCC_PATH=/opt/maca/tools/cu-bridge CUCC_CMAKE_ENTRY=2
export PATH="$CUCC_PATH/tools:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0 MACA_PATH=/opt/maca MACA_HOME=/opt/maca
export PYTHONPATH="$src/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=4 MACA_VISIBLE_DEVICES=4 MC_VISIBLE_DEVICES=4
export TRITON_CACHE_DIR="$build/triton-cache"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ $mode == build ]]; then
  cmake_maca -S "$src" -B "$build" -DBACKEND=MACA \
    -DCMAKE_BUILD_TYPE=Release -DFLAGFFT_BUILD_CLI=ON -DFLAGFFT_BUILD_TESTS=ON \
    -DSQLite3_INCLUDE_DIR=/opt/conda/include -DSQLite3_LIBRARY=/opt/conda/lib/libsqlite3.so \
    -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DMCFFT_INCLUDE_DIR=/opt/maca/include/mcfft -DMCFFT_LIB=/opt/maca/lib/libmcfft.so \
    -DMACA_PATH=/opt/maca \
    -DFETCHCONTENT_SOURCE_DIR_FMT=/workspace/FlagFFT-build/vendor/fmt-src \
    -DFETCHCONTENT_SOURCE_DIR_JSON=/workspace/FlagFFT-build/vendor/json-src \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/workspace/FlagFFT-build/vendor/googletest-src \
    > "$out/configure.log" 2>&1
  make_maca -C "$build" -j8 flagfft-cli numpy_fft_capture > "$out/build.log" 2>&1
elif [[ $mode == run ]]; then
  # Only one team member coordinates GPU 4; flock also prevents accidental overlap.
  exec 9>/tmp/flagfft-maca-gpu4.lock
  flock -x 9
  {
    date -u +%FT%TZ
    git -C "$src" rev-parse HEAD
    git -C "$src" status --short
    git -C "$src/deps/libtriton_jit" rev-parse HEAD
    python3 -c 'import torch,triton,flagfft_codegen; print("torch",torch.__version__); print("triton",triton.__version__,triton.__file__); print("codegen",flagfft_codegen.__file__)'
    env | sort | grep -E '^(FLAGFFT_|TRITON_|CUDA_VISIBLE|MACA_|MC_VISIBLE|PYTHONPATH|LD_LIBRARY_PATH)'
    /usr/bin/mx-smi
  } > "$out/environment.txt" 2>&1
  set +e
  python3 "$src/tools/run_tests.py" --gpus 4 --build-dir "$build" \
    --output-dir "$out" --warmup 5 --iters 50 --timeout 600 --color never "$@" \
    > "$out/runner.log" 2>&1
  result=$?
  /usr/bin/mx-smi > "$out/gpu-after.txt" 2>&1
  printf 'VALIDATION_EXIT=%s\n' "$result" | tee "$out/completion.txt"
  exit "$result"
else
  echo "unknown mode: $mode" >&2
  exit 2
fi
