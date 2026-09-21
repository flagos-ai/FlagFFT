#!/usr/bin/env bash
# Same CLI, source and benchmark loop; isolate generated kernels by variant.
set -euo pipefail
if (( $# != 3 )); then
  echo "usage: $0 SOURCE BUILD_PARENT OUTPUT" >&2
  exit 2
fi
src=$(realpath "$1"); parent=$(realpath "$2"); out=$(realpath -m "$3")
mkdir -p "$out"
# Validate isolated runtime directories before taking the GPU lock.
for variant in A C D; do
  runtime="$parent/maca-single-control-$variant-6ee6f03"
  test -x "$runtime/flagfft-cli"
  mkdir -p "$runtime/ctest/.flagfft" "$runtime/triton-cache"
  test -d "$runtime/.flagfft" && test -w "$runtime/.flagfft"
done
export PYTHONPATH="$src/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=4 MACA_VISIBLE_DEVICES=4 MC_VISIBLE_DEVICES=4
export FLAGFFT_TUNE_DISABLE=1 FLAGFFT_BENCH_SAMPLES=1
export FLAGFFT_MACA_SPLIT_ORDER=lsb FLAGFFT_MACA_BLUESTEIN_FOUR_STEP_FUSION=0
export FLAGFFT_MACA_VEC_IO=0
unset FLAGFFT_MACA_BATCH_PACK FLAGFFT_MACA_INNER_PACK FLAGFFT_MACA_MAX_WARPS FLAGFFT_MACA_MAX_PACK
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>/tmp/flagfft-maca-gpu4.lock
flock -x 9
{
  date -u +%FT%TZ
  git -C "$src" rev-parse HEAD
  git -C "$src" status --short
  for variant in A C D; do
    sha256sum "$parent/maca-single-control-$variant-6ee6f03/flagfft-cli"
  done
  env | sort | grep -E '^(FLAGFFT_|CUDA_VISIBLE|MACA_|MC_VISIBLE|PYTHONPATH)'
} > "$out/environment.txt"
for round in ${FLAGFFT_COMPARE_ROUNDS:-1 2 3}; do
  variants="A C D"
  if (( round == 2 )); then variants="D C A"; fi
  for variant in $variants; do
    build="$parent/maca-single-control-$variant-6ee6f03"
    export TRITON_CACHE_DIR="$build/triton-cache"
    export FLAGFFT_MACA_EXCHANGE=direct FLAGFFT_MACA_BLUESTEIN_LEAF_FUSION=0
    shapes=1024,2048,997
    if [[ $variant == A ]]; then export FLAGFFT_MACA_EXCHANGE=; fi
    if [[ $variant == D ]]; then
      export FLAGFFT_MACA_BLUESTEIN_LEAF_FUSION=1
      shapes=997
    fi
    # Repeated shapes share the in-process kernel cache, but retain independent
    # plans, warmup and timing loops. These are labelled steady-state repeats.
    base_shapes=$shapes
    for (( repeat=1; repeat<${FLAGFFT_COMPARE_REPEATS:-1}; ++repeat )); do
      shapes="$shapes,$base_shapes"
    done
    for direction in ${FLAGFFT_COMPARE_DIRECTIONS:-forward inverse}; do
      stem="$out/round${round}_${variant}_${direction}"
      /usr/bin/mx-smi -i 4 --show-clock --show-dpm cur > "$stem.clock-before.txt"
      "$build/flagfft-cli" bench --api c2c --rank 1 --shape "$shapes" --batch 1 \
        --direction "$direction" --warmup 200 --iters 100 --json --print-path \
        > "$stem.json" 2> "$stem.stderr"
      /usr/bin/mx-smi -i 4 --show-clock --show-dpm cur > "$stem.clock-after.txt"
    done
  done
done
printf 'COMPARE_EXIT=0\n' > "$out/completion.txt"
