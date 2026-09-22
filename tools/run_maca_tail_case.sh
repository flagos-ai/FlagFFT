#!/usr/bin/env bash
# Execute one frozen MACA variant; use a distinct executable/cache directory
# for each variant, and the same physical GPU for all members of an A/B pair.
set -euo pipefail
if (( $# < 6 )); then
  echo "usage: $0 SOURCE BUILD OUTPUT GPU OP SHAPE [runner arguments...]" >&2
  exit 2
fi
src=$(realpath "$1")
build=$(realpath "$2")
out=$(realpath -m "$3")
gpu=$4
op=$5
shape=$6
shift 6
[[ $gpu =~ ^[0-7]$ ]] || exit 2
test -x "$build/flagfft-cli"
test -x "$build/ctest/numpy_fft_capture"
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite existing result directory: $out" >&2
  exit 2
fi
mkdir -p "$out" "$build/.flagfft" "$build/ctest/.flagfft"
exec 9>"/tmp/flagfft-maca-gpu${gpu}.lock"
flock -x 9
export CUDA_VISIBLE_DEVICES=$gpu MACA_VISIBLE_DEVICES=$gpu MC_VISIBLE_DEVICES=$gpu
export FLAGFFT_PYTHON=/opt/conda/bin/python
export PYTHONPATH="$src/python"
export TRITON_CACHE_DIR="$build/triton-cache"
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_BENCH_SAMPLES=1
export MACA_PATH=/opt/maca MACA_HOME=/opt/maca
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
finish() {
  local result=$?
  printf 'VALIDATION_EXIT=%s\n' "$result" > "$out/completion.txt"
}
trap finish EXIT
{
  date -u +%FT%TZ
  git -C "$src" rev-parse HEAD
  git -C "$src" status --short
  sha256sum "$build/flagfft-cli" "$build/ctest/numpy_fft_capture"
  if [[ -f "$build/libflagfft.so" ]]; then sha256sum "$build/libflagfft.so"; fi
  env | sort | grep -E '^(FLAGFFT_|TRITON_|CUDA_VISIBLE|MACA_|MC_VISIBLE|PYTHONPATH|LD_LIBRARY_PATH)'
  printf 'GPU=%s OP=%s SHAPE=%s WARMUP=5 ITERS=30\n' "$gpu" "$op" "$shape"
} > "$out/environment.txt"
python3 "$src/tools/run_tests.py" --gpus "$gpu" --build-dir "$build" \
  --ops "$op" --shapes "$shape" --scales 1 --warmup 5 --iters 30 \
  --timeout 600 --color never --output-dir "$out/acceptance" "$@" \
  > "$out/runner.log" 2>&1
