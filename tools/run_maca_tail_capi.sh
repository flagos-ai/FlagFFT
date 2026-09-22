#!/usr/bin/env bash
# Correctness only: one isolated Python/C++ cache per real-DFT variant/mode.
set -euo pipefail
if (( $# != 6 )); then
  echo "usage: $0 SOURCE LIBRARY EXE_DIR OUTPUT GPU INPLACE_0_OR_1" >&2
  exit 2
fi
src=$(realpath "$1")
library=$(realpath "$2")
exe=$(realpath -m "$3")
out=$(realpath -m "$4")
gpu=$5
inplace=$6
[[ $gpu =~ ^[0-7]$ && $inplace =~ ^[01]$ ]] || exit 2
test -f "$library"
test ! -e "$exe"
test ! -e "$out"
mkdir -p "$exe" "$out"
cp /opt/conda/bin/python "$exe/python"
cp "$library" "$exe/libflagfft.so"
exec 9>"/tmp/flagfft-maca-gpu${gpu}.lock"
flock -x 9
trap 'result=$?; printf "CAPI_EXIT=%s\n" "$result" > "$out/completion.txt"' EXIT
export CUDA_VISIBLE_DEVICES=$gpu MACA_VISIBLE_DEVICES=$gpu MC_VISIBLE_DEVICES=$gpu
export PYTHONHOME=/opt/conda PYTHONPATH="$src/python" FLAGFFT_PYTHON=/opt/conda/bin/python
export MACA_HOME=/opt/maca MACA_PATH=/opt/maca FLAGFFT_TUNE_DISABLE=1
export LD_LIBRARY_PATH="$exe:/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="$exe/triton-cache"
export FLAGFFT_TEST_MACA=1 FLAGFFT_TEST_MACA_INPLACE=$inplace
export FLAGFFT_TEST_LIBRARY="$exe/libflagfft.so" FLAGFFT_TEST_MACA_EXE_DIR="$exe"
export FLAGFFT_TEST_MACA_OUTPUT_DIR="$out/cases"
export FLAGFFT_TEST_MACA_RESULTS_ROOT="$(dirname "$out")"
export FLAGFFT_TEST_MACA_SHAPES=${FLAGFFT_TEST_MACA_SHAPES:-23}
export FLAGFFT_TEST_MACA_APIS=${FLAGFFT_TEST_MACA_APIS:-r2c,c2r,d2z,z2d}
export FLAGFFT_TEST_MACA_SCALES=${FLAGFFT_TEST_MACA_SCALES:-all}
{
  date -u +%FT%TZ
  git -C "$src" rev-parse HEAD
  git -C "$src" status --short
  sha256sum "$exe/python" "$exe/libflagfft.so"
  env | sort | grep -E '^(FLAGFFT_|TRITON_|CUDA_VISIBLE|MACA_|MC_VISIBLE|PYTHONPATH|PYTHONHOME|LD_LIBRARY_PATH)'
} > "$out/environment.txt"
"$exe/python" -m pytest -q "$src/tests/python/test_maca_c_api.py" \
  > "$out/pytest.log" 2>&1
