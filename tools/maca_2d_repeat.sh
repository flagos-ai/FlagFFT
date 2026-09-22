#!/usr/bin/env bash
# Paired mcFFT timings, one physical GPU, isolated default/rollback caches.
# Run inside the MACA container; never edits the source worktree.
set -euo pipefail
if (( $# != 4 )); then
  echo "usage: $0 SOURCE BUILD OUTPUT PHYSICAL_GPU" >&2
  exit 2
fi
src=$(realpath "$1")
build=$(realpath "$2")
out=$(realpath -m "$3")
gpu=$4
[[ $gpu =~ ^[0-7]$ ]] || exit 2
test -x "$build/flagfft-cli"
test ! -e "$out"
mkdir -p "$out"
exec 9>"/tmp/flagfft-maca-gpu${gpu}.lock"
flock -x 9
for knob in $(compgen -e FLAGFFT_MACA_); do unset "$knob"; done
unset FLAGFFT_PROFILE_KERNELS FLAGFFT_PACKED_REAL
export CUDA_VISIBLE_DEVICES="$gpu" MACA_VISIBLE_DEVICES="$gpu" MC_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$src/python" FLAGFFT_TUNE_DISABLE=1 FLAGFFT_BENCH_SAMPLES=1
git -C "$src" rev-parse HEAD > "$out/source-commit.txt"
git -C "$src" status --short > "$out/source-status.txt"
sha256sum "$build/flagfft-cli" "$build/libflagfft.so" "$0" > "$out/sha256.txt"
env | sort | grep -E '^(FLAGFFT_|CUDA_VISIBLE|MACA_VISIBLE|MC_VISIBLE|PYTHONPATH)' > "$out/environment.txt"
/usr/bin/mx-smi > "$out/gpu-before.txt"
shapes=${FLAGFFT_QUALIFY_SHAPES:-64x64,128x128,2048x2048,46189x48,32x46189}
rounds=${FLAGFFT_QUALIFY_ROUNDS:-"1 2 3"}
for variant in rollback default; do
  variant_build="$build/repeat-$variant"
  mkdir -p "$variant_build"
  cp "$build/flagfft-cli" "$build/libflagfft.so" "$variant_build/"
done
for round in $rounds; do
  # Baseline once, followed by three independent-process candidate rounds.
  variants=default
  if [[ $round == 1 ]]; then variants="rollback default"; fi
  for variant in $variants; do
    variant_build="$build/repeat-$variant"
    export LD_LIBRARY_PATH="$variant_build:/opt/maca/lib:/opt/conda/lib"
    export TRITON_CACHE_DIR="$variant_build/triton-cache"
    if [[ $variant == rollback ]]; then
      export FLAGFFT_MACA_2D_SINGLE=0
    else
      unset FLAGFFT_MACA_2D_SINGLE
    fi
    for spec in c2c:forward c2c:inverse r2c:forward c2r:inverse; do
      api=${spec%:*}
      direction=${spec#*:}
      "$variant_build/flagfft-cli" bench --rank 2 --api "$api" --shape "$shapes" \
        --batch 1 --direction "$direction" --warmup 200 --iters 100 --print-path --json \
        > "$out/round${round}_${variant}_${api}_${direction}.json" \
        2> "$out/round${round}_${variant}_${api}_${direction}.stderr"
    done
  done
done
/usr/bin/mx-smi > "$out/gpu-after.txt"
printf 'VALIDATION_EXIT=0\n' > "$out/completion.txt"
