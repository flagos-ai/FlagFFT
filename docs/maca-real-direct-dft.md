# MACA small real DirectDFT experiment

Set `FLAGFFT_MACA_REAL_DIRECT_DFT=1` before plan creation to fuse real
boundaries into a DirectDFT kernel. Unset it or set it to `0` for the baseline.
This switch is independent of `FLAGFFT_MACA_1D_SINGLE` and defaults to off.

The compiler selects it only for MACA, rank 1, batch 1, FP32/FP64, and an
existing DirectDFT plan of the requested length (1 through 128). It does not
force the planner to choose DirectDFT. R2C/D2Z and C2R/Z2D use separate kernel
and cache identities. Plan descriptions contain `direct_dft_r2c_kernel` or
`direct_dft_c2r_kernel` when the experiment is active.

The kernels reuse the existing full DFT tables and launch ABI. R2C reads real
input and writes only the half spectrum, with exact zero imaginary DC and
even-length Nyquist endpoints. C2R restores Hermitian input and ignores those
endpoint imaginary components, including nonfinite values. Inverse output is
unnormalised, as in the C API. FP64 retains compensated summation. Out-of-place
execution launches one compute kernel; in-place execution also retains the
existing device copy for alias protection, sized to the actual input extent.

## Optional FP64 tree reduction

For MACA FP64 lengths 1 through 32, opt into the tree variant with both:

```sh
export FLAGFFT_MACA_REAL_DIRECT_DFT=1
export FLAGFFT_MACA_REAL_DFT_REDUCTION=tree
```

The reduction switch is read at code generation. Only the exact value `tree`
enables it; unset it or use `kahan` for the original compensated accumulation.
FP32, lengths above 32, and non-MACA targets keep their original source and
kernel names. The small-real runtime gate still applies, so this does not
force a planner algorithm or affect C2C.

The tree computes independent products using the existing FP64 input and DFT
tables, then emits explicit adjacent pair additions, carrying an unmatched
last term to the next level. N=23 has at most five addition levels. Every
product and addition remains FP64; there is no TF32, dot product, or precision
conversion. Real/Hermitian boundary rules and the launch ABI are unchanged.
Tree kernel names end in `_tree`, for example
`direct_dft_c2r_kernel_n23_f64_b32_tree`.

Only the tree variant reads table entries at `j*N+k`, making accesses along
output lane `k` contiguous. The production DFT-table builder forms the integer
product `j*k` before evaluating the angle, so its FP64 real and imaginary
tables are bitwise symmetric. This preserves the coefficients while changing
their access pattern; the frozen Kahan indexing remains `k*N+j`.

Tree summation is not compensated and need not match Kahan bit for bit.
Cancellation and mixed-magnitude inputs can lose more low-order bits. CPU
oracle tests check these cases as well as DC/Hermitian, odd/even endpoints,
and input scales; they do not establish MACA numerical or performance results.

C++ is unchanged: the in-process cache key does not include this switch,
and the generated module basename is shared by the variants. Use separate
processes, executable/cache directories, and codegen output directories for
Kahan/tree A/B, and verify the `_tree` name in recorded plans. Do not switch
the variable on an existing plan or reuse one variant's generated files for
the other. No GPU tests were run for this optional change.

Tree handoff validation: 260 Python tests and 8 subtests passed, including
the original regression suites. The tree tests verify addition depth,
FP64 output, cancellation error against `math.fsum` of rounded terms, table
symmetry, contiguous table addresses, metadata, and opt-in scope. Default and
fallback source/name/ABI also matched commit `6383e0f` byte for byte across
202 combinations. Results are in the parent workspace's
`results/20260922_150928_maca_real_tree/`.

## Local CPU and source checks

Always pin the worktree package; the container's installed package may point
to another checkout:

```sh
docker exec \
  -e PYTHONPATH=/workspace/FlagFFT-dev-maca-tail-real/python \
  -w /workspace/FlagFFT-dev-maca-tail-real flagtree-dev3 \
  /root/.pyenv/versions/3.12.13/bin/python -m pytest -q \
  tests/python/test_maca_real_direct_dft.py tests/python/test_codegen.py \
  tests/python/test_backend_profile.py tests/python/test_maca_single_policy.py
```

The new CPU tests execute generated source with a NumPy model of Triton
arithmetic and masked pointers. They cover lengths 1, 2, 22, 23, 24, 33, 128;
FP32/64; input scales 1e-6, 1, 1e6; DC, impulses, random input, Hermitian
endpoints, odd final bins, and unnormalised roundtrips. These do not validate
MACA lowering or the device runtime dispatch.

## Device validation handoff

Use the existing isolated MACA executable/cache/library setup described in
`tests/python/test_maca_c_api.py`, with the intended worktree in `PYTHONPATH`
and the JIT interpreter in `FLAGFFT_PYTHON`. For an already configured MACA
test environment, the variant-specific command is:

```sh
env FLAGFFT_MACA_REAL_DIRECT_DFT=1 \
  FLAGFFT_TEST_MACA=1 FLAGFFT_TEST_MACA_SHAPES=23 \
  FLAGFFT_TEST_MACA_APIS=r2c,c2r,d2z,z2d FLAGFFT_TEST_MACA_SCALES=all \
  "$FLAGFFT_TEST_MACA_EXE_DIR/python" -m pytest -q tests/python/test_maca_c_api.py
```

Use a fresh isolated executable/cache and results directory for each variant;
do not toggle the switch on a previously compiled plan. Check recorded plan
descriptions for the fused kernel names. The switch alone is not evidence of
selection if a tuned plan uses a different algorithm. Device scheduling and
performance measurements belong to the main validation agent.

Validation at handoff: 192 Python tests and 8 subtests passed; CUDA-backend
`flagfft` and `test_kernel_key` built, and the CTest kernel-key suite passed.
The four N23 generated kernels also compiled offline for CUDA sm80 with GPUs
hidden. MACA target source/metadata generation passed, but MACA device
correctness, compiler lowering, and performance have not been measured here.
Logs and generated artifacts are under the parent workspace's
`results/20260922_145332_maca_real_direct_dft/`.
