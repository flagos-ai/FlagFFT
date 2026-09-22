#!/usr/bin/env bash
# Bounded end-to-end plan comparison, with independent executable/JIT caches.
set -euo pipefail
if (( $# < 7 )); then
  echo "usage: $0 SOURCE BUILD OUTPUT GPU OP SHAPE PLAN..." >&2
  exit 2
fi
src=$(realpath "$1"); build=$(realpath "$2"); out=$(realpath -m "$3")
gpu=$4; op=$5; shape=$6
shift 6
driver=$(dirname "$(realpath "$0")")/run_maca_tail_case.sh
test ! -e "$out"
mkdir -p "$out"
trap 'result=$?; printf "SWEEP_EXIT=%s\n" "$result" > "$out/completion.txt"' EXIT
for plan in "$@"; do
  [[ $plan =~ ^[a-z0-9]+$ ]] || exit 2
  variant="$out/build-$plan"
  mkdir -p "$variant/ctest"
  cp "$build/flagfft-cli" "$build/libflagfft.so" "$build/CMakeCache.txt" "$variant/"
  cp "$build/ctest/numpy_fft_capture" "$variant/ctest/"
  FLAGFFT_MACA_TAIL_PLAN=$plan FLAGFFT_PACKED_REAL=0 \
    bash "$driver" "$src" "$variant" "$out/$plan" "$gpu" "$op" "$shape"
done
