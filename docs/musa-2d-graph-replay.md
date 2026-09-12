# MUSA S5000 complex 2D launch policy

The MUSA S5000 (capability `31`, driver/runtime 4.3) runs batch-1 complex
2D FFTs faster with direct kernel launches than with the internal graph
replay cache. The compiler disables that cache for this target and batch
size, for both the row/column (RC) and row/transpose/column/transpose paths.
Other targets and batch sizes retain their existing graph policy. Real
2D transforms are not changed.

## Controlled experiment

The baseline is commit `0e990e4`, with libtriton_jit at
`0271d3ccb0c031ea117235acaab96c7774fc34de`, built in Release mode on
`baai-mthread`. The baseline shared library was retained for A/B runs using
the same CLI, generated kernels, inputs, stream, and muFFT reference.
The optimization changes launch policy only; it does not change FFT
factorization, arithmetic, normalization, or error tolerances.

An initial C2C forward experiment (batch 1, out of place, 20 warmups,
200 timed iterations) measured:

| Shape | Graph baseline | Direct launches | Baseline/direct |
|---|---:|---:|---:|
| 128 × 128 | 0.125680 ms | 0.033920 ms | 3.705× |
| 46189 × 48 | 1.747240 ms | 1.644440 ms | 1.063× |

For 128 × 128, the execution plan already uses two kernels without
transposes: contiguous row FFT and strided column FFT. The difference
therefore isolates substantial overhead in the graph execution path;
adding another transpose optimization is not needed to explain this gain.
The experiment does not determine the internal driver cause of that
overhead, and should not be generalized to all MUSA releases or hardware.

For 46189 × 48, the plan runs a length-48 row FFT, transpose, a
46189 = 209 × 221 four-step column FFT, and transpose back. Temporary
per-kernel event timing measured approximately 0.783 ms and 0.564 ms for
the two four-step kernels. These remain the main optimization target.
Profiling synchronized after each kernel and includes instrumentation
overhead; its timings are diagnostic and must not replace the uninstrumented
end-to-end benchmark.

## Reproduction and records

All experiment scripts, JSON reports, correctness logs, and the temporary
profiling patch are stored outside the worktree in:

`../results/20260913_013456_musa_2d_opt/`

`bench.sh` reproduces the representative benchmark. `verify.sh` runs C2C
and Z2Z forward/inverse/reference/roundtrip validation at 128 × 128,
46189 × 48, 23 × 30, and 8191 × 1009, followed by paired baseline/optimized
benchmarks of all nine CT/BS matrix shapes in both directions (20 warmups,
100 timed iterations). Its paths refer to the independent remote build.
The correctness tests use the existing three input scales and tolerances.

All 24 correctness tests passed, with no skips. The paired 36-case matrix
measured 34 improvements; the two remaining measurements were within 0.5%
of the baseline (Z2Z 32 × 46189). The geometric mean of baseline/direct
timing ratios was 1.844×; this is a case-weighted aggregate, not a workload
throughput prediction. In this second measurement, C2C forward 128 × 128
went from 0.133440 to 0.027960 ms (4.773×), and 46189 × 48 went from
1.748600 to 1.642840 ms (1.064×). Their optimized muFFT/FlagFFT ratios were
0.901× and 0.631× respectively, so both still trail muFFT.

`recheck.sh` separately repeats Z2Z forward 32 × 46189 in reverse A/B
order with 50 warmups and 300 timed iterations. Retain those measurements
alongside the original matrix; tiny differences on compute-heavy cases
should not be presented as established gains or regressions.

Kernel timing instrumentation is not part of the final implementation.
