# MUSA FP32 paired codelets

Extending the d07f623 paired-output codelets to MUSA complex64 is effective
for the same 209/221-point leaves and six non-strided four-step kernel kinds.
The mathematical identity and operation order are shared with FP64; precision
and correctness tolerances are unchanged. CUDA selection remains separate
until the independent A100 experiment is evaluated.

The FP32 row kernel metadata changes from 244 bytes of private memory and
256 temporary registers to zero private memory and 76 temporary registers.
This corroborates reduced register pressure in addition to less arithmetic.

MTT S5000 GPU 0, Release, batch one, out of place, 20 warmups and 100 timed
iterations. Baseline d07f623 uses its own saved library and Python generator.
All 36 GPU correctness checks and 76 Python tests passed. GPU coverage is
all six APIs at 46189 × 48 batch 1/3 and 32 × 46189 batch 1.

| API | Direction | Shape | Old ms | New ms | muFFT / new |
|---|---|---|---:|---:|---:|
| c2c | forward | 46189x48 | 1.125120 | 0.539280 | 1.926× |
| c2c | forward | 32x46189 | 0.687160 | 0.309240 | 2.215× |
| c2c | inverse | 46189x48 | 1.119600 | 0.537080 | 1.936× |
| c2c | inverse | 32x46189 | 0.686520 | 0.319000 | 2.142× |
| r2c | forward | 46189x48 | 0.683040 | 0.377760 | 1.555× |
| r2c | forward | 32x46189 | 0.644880 | 0.255480 | 2.780× |
| c2r | inverse | 46189x48 | 0.696120 | 0.396200 | 1.476× |
| c2r | inverse | 32x46189 | 0.655200 | 0.266120 | 2.701× |

Raw data and reproduction scripts: `../results/20260913_102255_musa_fp32_paired/`.
