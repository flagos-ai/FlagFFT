# MUSA FP64 mixed-radix codelets

The previous tuning (`a8ea518`, documented at `d06dd0e`) left Z2Z
46189 × 48 at approximately 5.04 ms, versus 1.12 ms for C2C. Both use
the same 46189 = 209 × 221 split, with radix 19/11 and 17/13 leaves.
The corresponding muFFT precision penalty was much smaller.

## Evidence and change

Device-event instrumentation around individual JIT launches located about
87% of the original latency in the two four-step kernels. The old row
kernel took about 2.42 ms and the column kernel about 2.03 ms. After
rewriting the codelets, these fell to approximately 0.25 and 0.32 ms;
transpose code was unchanged. Instrumentation was removed before all
final end-to-end comparisons.

The matching old FP64 row binary reports `private_memory_size=1600`
and `temp_reg_count=256`. The new binary reports zero private memory and
180 temporary registers. Both use eight warps and 32768 bytes of Triton
shared memory. This supports register spilling as a major source of the
old FP64 cost, alongside avoidable arithmetic. Raw binaries, IR, metadata
and profiling logs are preserved with the experiment. A preliminary
1564-byte figure belonged to a different FP32 cache and is not used in
the final comparison.

For odd radix n, define complex pair sums and differences
P_j = x_j + x_(n-j) and M_j = x_j - x_(n-j). Compute
C_k = x_0 + sum_j cos(2πkj/n) P_j and
S_k = sum_j sin(2πkj/n) M_j. Then the forward outputs are
X_k = C_k - i S_k and X_(n-k) = C_k + i S_k.

The new generator shares C/S between each output pair and finishes one
pair at a time, instead of maintaining all positive/negative output
accumulators throughout the butterfly. Radix-19 source multiplications
fall from 648 to 324 and additions/subtractions from 720 to 396. These
are static source operation counts, not measured machine instruction
counts. No arithmetic is demoted from FP64, and tolerance is unchanged.

The selection is restricted to MUSA `complex128`, leaf lengths 209/221,
and the six non-strided four-step row/column kinds (including real and
Hermitian boundary variants). The bundled codelets remain the default
for CUDA, FP32, other leaf lengths and other kernel kinds. Extending the
policy requires separate performance and correctness evidence.

## Validation and results

Results are stored outside this worktree in
`../results/20260913_025236_musa_fp64_2d/`. `REPORT.md` and
`comparison.csv` contain final paired measurements; `bench.sh` and
`verify.sh` preserve the commands. The baseline uses both a saved library
and a matching snapshot of the old Python generator. Final benchmarks
use MTT S5000 GPU 0, Release, batch one, out of place, 20 warmups and
100 timed iterations. The unchanged 128 × 128 shape is a control, and
its few-microsecond timing variation is not attributed to this change.

Validation includes 36 GPU correctness tests over all six APIs and
roundtrips at 46189 × 48 batch 1/3 and 32 × 46189 batch 1. Sixteen new
Python tests cover independent NumPy DFT agreement, inverse convention,
real/imaginary basis vectors, constant/random inputs at three scales,
and backend/precision selection. All 48 existing codegen tests pass.

Final paired results (milliseconds):

| API | Direction | Shape | Old | New | muFFT / new |
|---|---|---|---:|---:|---:|
| z2z | forward | 46189x48 | 5.026960 | 1.177760 | 1.367× |
| z2z | forward | 32x46189 | 3.125880 | 0.558920 | 1.939× |
| z2z | inverse | 46189x48 | 5.053880 | 1.194520 | 1.348× |
| z2z | inverse | 32x46189 | 3.047040 | 0.573080 | 1.894× |
| d2z | forward | 46189x48 | 2.863280 | 0.750680 | 1.222× |
| d2z | forward | 32x46189 | 3.144120 | 0.578320 | 1.929× |
| z2d | inverse | 46189x48 | 2.855960 | 0.735120 | 1.257× |
| z2d | inverse | 32x46189 | 3.127880 | 0.580600 | 1.946× |
