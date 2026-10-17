# Test matrices

`operators.yaml` keeps the same 30 logical operators for all matrices. Select
input shapes and numeric batches with `tools/run_tests.py --matrix-file PATH`.
Without that option, the runner continues to use `test_matrix.yaml`.

| Matrix | Purpose | Accuracy cases | Performance cases |
|---|---|---:|---:|
| `test_matrix.yaml` | Existing acceptance suite | 400 | 400 |
| `test_matrix_generalization.yaml` | Discovery baseline for policy design | 736 | 736 |
| `test_matrix_generalization_holdout.yaml` | Fixed validation after policy changes | 624 | 624 |

Counts include FP32/FP64 and all legal directions, before backend precision
omissions and reference-library skips. IX/NPU currently omit FP64, halving
these expanded counts. The manifest contains the full matrix, its source path
and SHA-256, selected cases and runtime environment; dry-run also prints the
matrix and its identity. Use separate timestamped output directories for each
matrix and repetition. Do not combine their scores with acceptance results.

The two generalization matrices use transform shapes absent from acceptance,
and have no exact transform shape in common. They sample mixed radices, primes,
odd/even real packing, boundary neighbors, non-cubes, rectangles, axis order
and batches outside the acceptance counts. They bound transform sizes to keep
the first survey manageable; they do not establish large-transform or
out-of-memory behavior. No GPU/NPU correctness or performance claims have been
made for these new inputs until the corresponding remote run completes.

`profile: generalization` allows distinct integer batches greater than one in
the batch groups; single groups remain batch one. In this profile CT/four-step/
prime are sampling labels, not forced algorithms: a legal plan without a
FourStep node does not fail just because it belongs to the four-step input
list. Actual plans and numerical correctness are still recorded and checked.
The default `acceptance` profile retains its batch constraints and FourStep
plan assertion. No optimization switch or planner default is changed here.

New full shapes can still select existing specialized child kernels. Inspect
the recorded runtime plans and provider choices before claiming a generic
path was measured. On Ascend, record whether a plan used SiP, Triton or AIV;
an automatic SiP route alone does not validate the Triton/AIV implementation.
These matrices do not force a provider or disable specializations. In-place,
custom strides/embeds, and independent kernel compilation timing are outside
the current runner's coverage.

Use discovery measurements to design rules. Freeze the holdout before tuning
and compare it after changes; it is a shape holdout, not an unseen-factor-family
guarantee. Once a holdout result is used to tune a rule, retire that holdout
and choose a new independent one. Keep unsupported inputs, timeouts and
failures visible rather than deleting them from the matrix. A platform
reference limitation remains separate from FlagFFT-vs-NumPy correctness.

From a source worktree, inspect the lists locally without a device:

```bash
python tools/run_tests.py --dry-run \
  --matrix-file conf/test_matrix_generalization.yaml
python tools/run_tests.py --dry-run \
  --matrix-file conf/test_matrix_generalization_holdout.yaml
```

On a remote selected and initialized using the workspace environment guide,
set `FLAGFFT_SURVEY_BUILD` to that backend's actual build directory, then run:

```bash
python tools/run_tests.py \
  --matrix-file conf/test_matrix_generalization.yaml \
  --build-dir "$FLAGFFT_SURVEY_BUILD" --gpus 0 \
  --warmup 3 --iters 30 --timeout 600 \
  --output-dir "../results/$(date +%Y%m%d_%H%M%S)_generalization_discovery"
```

Initialize CANN/SiP and library paths in the same shell **before** starting
Python on Ascend. Select verified idle devices rather than assuming device 0
is available. The example result path is outside the source tree; use the
actual external result root for a container, and archive it to the local
workspace `results/` after execution.

For the first baseline, run correctness and performance together (the default),
and repeat the discovery run at least three times with distinct result paths.
Run the same frozen matrix and settings on both code revisions with alternating
order. Compare matched case IDs, absolute FlagFFT/reference times, plan/provider
changes and repeat spread, then compute per-op speedup geometric means.
Warmup/iteration counts here match the previous acceptance measurements; the
matrix itself does not prescribe them. This config adds no automatic repeated
benchmark orchestration or policy tuning.
