# A100 paired mixed-radix codelets

The output-pair sharing used by the MUSA optimization also improves the
same 209/221-point leaves on A100. The original d07f623 selector only
selected MUSA FP64, so simply cherry-picking it would not change CUDA
kernels. This experiment explicitly enabled paired codelets for CUDA
FP32 and FP64, keeping CUDA graph replay enabled.

The final selector enables the six measured non-strided four-step kernel
kinds for complex64/complex128 on MUSA or NVIDIA Triton, with PPU excluded.
Other leaf lengths and kernel kinds retain the bundled codelets. The CUDA
measurements below cover A100; other NVIDIA architectures have not been
benchmarked in this experiment.

A100-SXM4-40GB, SM80, Release, batch one, out of place, 20 warmups and
100 timed iterations. The old variant uses a saved old Python generator;
results and reproduction details are in
`../results/20260913_102255_a100_paired_codelets/REPORT.md`.

| API | Direction | Shape | Old ms | New ms | Improvement | cuFFT / new |
|---|---|---|---:|---:|---:|---:|
| c2c | forward | 46189x48 | 0.262144 | 0.239616 | 1.094× | 0.863× |
| c2c | forward | 32x46189 | 0.156672 | 0.140288 | 1.117× | 1.044× |
| c2c | inverse | 46189x48 | 0.268288 | 0.245760 | 1.092× | 0.842× |
| c2c | inverse | 32x46189 | 0.156672 | 0.148480 | 1.055× | 0.986× |
| z2z | forward | 46189x48 | 0.434176 | 0.384000 | 1.131× | 0.803× |
| z2z | forward | 32x46189 | 0.230400 | 0.197632 | 1.166× | 1.166× |
| z2z | inverse | 46189x48 | 0.446464 | 0.392192 | 1.138× | 0.786× |
| z2z | inverse | 32x46189 | 0.237568 | 0.201728 | 1.178× | 1.142× |
| r2c | forward | 46189x48 | 0.220160 | 0.209920 | 1.049× | 0.595× |
| r2c | forward | 32x46189 | 0.149504 | 0.134144 | 1.115× | 1.176× |
| c2r | inverse | 46189x48 | 0.204800 | 0.193536 | 1.058× | 0.656× |
| c2r | inverse | 32x46189 | 0.154624 | 0.142336 | 1.086× | 1.144× |
| d2z | forward | 46189x48 | 0.376832 | 0.340992 | 1.105× | 0.517× |
| d2z | forward | 32x46189 | 0.248832 | 0.216064 | 1.152× | 1.185× |
| z2d | inverse | 46189x48 | 0.372736 | 0.337920 | 1.103× | 0.536× |
| z2d | inverse | 32x46189 | 0.266240 | 0.230400 | 1.156× | 1.138× |

All 16 paired performance cases improved by 1.049–1.178×. Correctness
passed 24 actually executed tests across the two shapes, covering all
six APIs, forward/inverse reference comparisons and both real roundtrips.
The per-API real test invocations intentionally skip unrelated APIs;
these skipped entries are not counted as passed tests. The final backend,
precision, leaf-length and kernel-kind selector passed 90 Python tests
(including the existing codegen suite).
