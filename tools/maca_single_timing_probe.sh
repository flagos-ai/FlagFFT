#!/usr/bin/env bash
# Diagnose cold-to-warm timing transitions without changing the bench loop.
set -euo pipefail
if (( $# < 3 )); then
  echo "usage: $0 SOURCE BUILD OUTPUT [shape (997)]" >&2
  exit 2
fi
src=$(realpath "$1"); build=$(realpath "$2"); out=$(realpath -m "$3")
shape=${4:-997}
mkdir -p "$out"
export PYTHONPATH="$src/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=4 MACA_VISIBLE_DEVICES=4 MC_VISIBLE_DEVICES=4
export TRITON_CACHE_DIR="$build/triton-cache"
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_BENCH_SAMPLES=1
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>/tmp/flagfft-maca-gpu4.lock
flock -x 9
{
  date -u +%FT%TZ
  git -C "$src" rev-parse HEAD
  git -C "$src" status --short
  sha256sum "$build/flagfft-cli"
  ldd "$build/flagfft-cli"
  env | sort | grep -E '^(FLAGFFT_|TRITON_|CUDA_VISIBLE|MACA_|MC_VISIBLE|PYTHONPATH)'
} > "$out/environment.txt"
round=0
for setting in 5:50 200:100 200:100 200:100; do
  round=$((round + 1))
  warmup=${setting%:*}; iters=${setting#*:}
  stem="$out/round${round}_w${warmup}_i${iters}"
  /usr/bin/mx-smi -i 4 --show-clock --show-dpm cur --show-clk-tr > "$stem.clock-before.txt"
  "$build/flagfft-cli" bench --api c2c --rank 1 --shape "$shape" --batch 1 \
    --direction forward --warmup "$warmup" --iters "$iters" --json --print-path \
    > "$stem.json" 2> "$stem.stderr"
  /usr/bin/mx-smi -i 4 --show-clock --show-dpm cur --show-clk-tr > "$stem.clock-after.txt"
done
printf 'TIMING_PROBE_EXIT=0\n' > "$out/completion.txt"
