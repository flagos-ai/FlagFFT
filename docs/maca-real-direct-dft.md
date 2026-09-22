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
