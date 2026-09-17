# MACA / MetaX C550 adaptation

The MACA build uses native `mcruntime` for memory, streams, events, module
loading and graph operations, and `mcFFT` for the CLI/test reference. It keeps
the existing FFT planner, radix codelets, Four-Step, Rader and Bluestein paths.

## Environment and device selection

The development environment is Python 3.12, Torch `2.8.0+metax3.7.2.0`,
FlagTree `0.6.1+metax3.6` (Triton 3.6), and MACA SDK 3.7.2, in
`flagrand-metax:py312-flagtree0.6.1-metax3.6`. Build and run against the same
SDK headers and libraries. The host's default CUDA/MThreads Python is not
the MACA environment.

Set the device filters before starting Python or a native process:

```bash
export CUDA_VISIBLE_DEVICES=4
export MACA_VISIBLE_DEVICES=4
export MC_VISIBLE_DEVICES=4
export FLAGFFT_PYTHON=/opt/conda/bin/python
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

On the development machine, physical card 4 has PCI address `0000:aa:00.0`.
With these filters, both MACA and Torch expose exactly one logical device,
numbered 0. The dedicated container `flagfft-maca-card4-adapt` fixes these
filters at creation. Test runners must use `--gpus 4` / `--gpu 4`, because
their arguments are physical IDs used to set the child process filters.

Native C550 capability is 10.2; this MetaX Torch/Triton environment exposes
the compatibility target `maca:80:64`. The MACA adaptor reports native device
capability for planning and uses the latter identity for kernel cache keys.
The backend registration name is `metax`; its runtime target name is `maca`.

## Build

Execute build commands in the container; edit the mounted sources on the host.
The fixed libtriton_jit revision `f3382f9` includes the MACA consumer CMake fixes:
public `USE_MACA`, a globally visible `MACA::mcruntime` imported target, and
respect for an explicit `MACA_PATH`. Its isolated compilation helper also
accepts a known target and binds FlagTree's hint manager to MACA without
querying a Torch CUDA compatibility device.

This dependency commit is on the local `codex/maca-consumer` branch. The
development results contain `libtriton_jit-maca.bundle` with its history.
For a checkout that already has the original submodule, restore the revision
before building:

```bash
git -C deps/libtriton_jit fetch /workspace/FlagFFT-results/libtriton_jit-maca.bundle \
    refs/heads/codex/maca-consumer
git -C deps/libtriton_jit checkout f3382f9
```

```bash
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH="$CUCC_PATH/tools:$PATH"
export CUCC_CMAKE_ENTRY=2
cmake_maca -S . -B /workspace/FlagFFT-build/release \
    -DBACKEND=MACA -DMACA_PATH=/opt/maca -DCMAKE_BUILD_TYPE=Release \
    -DFLAGFFT_BUILD_CLI=ON -DFLAGFFT_BUILD_TESTS=ON -DBUILD_TESTING=ON \
    -DPython_EXECUTABLE=/opt/conda/bin/python \
    -DTorch_DIR=/opt/conda/lib/python3.12/site-packages/torch/share/cmake/Torch \
    -DSQLite3_INCLUDE_DIR=/opt/conda/include \
    -DSQLite3_LIBRARY=/opt/conda/lib/libsqlite3.so \
    -DTRITON_JIT_USE_EXTERNAL_PYBIND11=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_STANDARD=17
make_maca -C /workspace/FlagFFT-build/release -j8
```

The CUDA architecture/standard flags configure the SDK's cu-bridge support
for the vendor Torch package; the FFT adaptor and binary launcher use native
MACA APIs. When GitHub downloads are unavailable, prepopulate dependency
sources on the host and pass `FETCHCONTENT_SOURCE_DIR_JSON`,
`FETCHCONTENT_SOURCE_DIR_FMT` and `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`.

## Compiler compatibility

Codegen receives the explicit target from C++. MACA uses ordinary Triton
loads/stores, precomputed outer twiddles, and tiled transpose. It disables
the PTX I/O and approximate-twiddle variants.

The installed plugin's TLE shared-pointer lowering fails with an invalid
cross-address-space `llvm.bitcast`. The portable leaf instead evaluates all
butterflies of each stage in register tensors and inverts the existing routing
to gather the next stage. It uses no TLE local pointers or global scratch
buffer. Small supported FFTs use one existing codelet. Multi-stage exchange
uses tensors of at least 128 elements and avoids `tl.join`, because the SDK
cannot parse the plugin's `maca.shfl.sync` for warp layout conversions.

The planner and codegen use 64 threads per warp, at least two warps and one transform per
multi-stage leaf block. Small single-codelet leaves retain batch packing.
MACA-specific settings do not change the other backends' FFT decomposition
or data exchange.

MACA does not apply the existing small-batch radix/lane heuristic or the
measured thread-local Four-Step preference. Its cost estimate uses all stage
butterflies, matching the portable kernel. Four-Step inner packing is one in
both planning and codegen. Rader and Bluestein remain available to the tuner;
MACA does not inherit the FP32 preference based on fused Bluestein.

Bluestein uses the existing separate prepare, FFT, pointwise, inverse FFT and
finish pipeline. The double-FFT fused leaf is disabled for MACA: compilation
of a 257-point transform with a 1024-point convolution did not finish after
several minutes during validation. The split pipeline avoids that optimization
problem. MACA has separate planner/codegen/runtime tuning fingerprints.

The two-warp floor passed both precisions and directions for single codelets
and 64/256/780/1024-point multi-stage leaves. One-warp multi-stage kernels
still emit the unsupported shuffle operation. Larger leaves increase the
warp count from the full stage lane count, up to eight.

MACA compiles in the existing codegen Python process using libtriton_jit's
`standalone_compile.py`, then loads `.mcfatbin` and launches raw arguments in
C++ through `TritonKernelImpl<MacaBackend>`. Compilation and native module
loading finish before graph capture. This process boundary avoids the SDK's
`libmcFlashAttn.so` teardown crash observed with embedded Torch in native
executables. Kernels requiring nonzero global/profile scratch are rejected;
the supported kernels use the backend ABI's two trailing null scratch pointers.

The compiler receives `GPUTarget("maca", 80, 64)` directly. Generated MACA
modules omit unused TLE imports, and the isolated helper binds FlagTree's
hint manager to MACA rather than detecting Torch's `cuda` compatibility type.
This avoids device initialization in the compiler process. A cached 16-point
leaf probe took 2.25 seconds and verified `torch.cuda.is_initialized() == False`;
the earlier device-properties query alone took 19.47 seconds. These are startup
measurements, separate from FFT event timings.

## Validation

The implementation checkpoint passed these checks:

| Check | Coverage | Result |
|---|---|---|
| Portable leaf against NumPy | FP32/FP64, both directions, lengths 8/15/16/32/64/256/780, batches 1/2/7/16/256, two warps | 140 passed |
| Final native C API against NumPy | Six APIs; 1D/2D/3D, 257/8191-point primes, odd/prime axes, nondefault stream, batch 7/256 | 120 passed; mcFFT checked independently |
| Forced prime runtime against CPU direct DFT | Rader and split Bluestein, N=257, batch=7, FP32/FP64, both directions; stream, graph replay and in-place | Passed |
| Packed real batch runtime | N=1024, batch=7, D2Z/Z2D, dense/padded/in-place layouts | Passed |
| libtriton_jit runtime | Hooks, C++/Python arguments, tuple signatures | 5 passed |
| Planner structural regression | Existing CUDA/MUSA synthetic-context cases | 6 passed |
| Python CPU regression | Complete existing Python test directory | 89 passed, 140 GPU tests skipped |
| CUDA compile regression | Core library, CLI and complete C++ test targets; no GPU execution | Passed |

The native C API NumPy matrix and serial mcFFT performance measurements are
recorded separately in the development results. The numerical harness uses
native MACA allocations and a nondefault stream without importing Torch;
FlagFFT and mcFFT each execute three times before comparison with NumPy.

The public PlanMany API keeps the existing layout limits: unit element strides,
dense batch distances, and the standard padded 1D real layout for in-place
transforms. Arbitrary element strides or extra batch padding, including padded
2D/3D layouts, return `FLAGFFT_NOT_SUPPORTED`.

Run the opt-in portable-leaf numerical tests on the selected card:

```bash
FLAGFFT_TEST_MACA=1 /opt/conda/bin/python -m pytest \
    tests/python/test_maca_codegen.py -q
```

Build the independent NumPy capture executable against the same library:

```bash
cmake_maca -S tools/numpy_fft_validation \
    -B /workspace/FlagFFT-build/numpy-capture \
    -DBACKEND=MACA -DMACA_PATH=/opt/maca \
    -DFLAGFFT_BUILD_DIR=/workspace/FlagFFT-build/release
make_maca -C /workspace/FlagFFT-build/numpy-capture -j4
```

The development result mount `/workspace/FlagFFT-results` mirrors the host's
workspace-level `results/20260916_154838_maca_implementation/`. Keep all test
and benchmark output under that mount, including intermediate CSV files:

```bash
/opt/conda/bin/python tools/run_tests.py \
    --build-dir /workspace/FlagFFT-build/release \
    --combination 1d_ct_single --ops 1d_ct_single_c2c \
    --gpus 4 --accuracy-only \
    --output-dir /workspace/FlagFFT-results/numpy_smoke
/opt/conda/bin/python tools/run_tests.py \
    --build-dir /workspace/FlagFFT-build/release --gpus 4 \
    --combination 1d_ct_single --accuracy-only \
    --output-dir /workspace/FlagFFT-results/accuracy
```

NumPy validation records FlagFFT and mcFFT independently. A vendor-library
failure or a process error remains a failure even if another implementation
passes. Benchmark compilation time is separate from event timings; measure
performance after correctness validation finishes on card 4.

## C550 performance baseline

The 28-case baseline used physical card 4 alone, 10 warmups and 100 timed
iterations with native MACA events. FlagFFT kept its default API execution
(2D/3D graph paths enabled); mcFFT used native direct Exec. Compilation
was completed before event timing. Five cases were faster than mcFFT;
the geometric mean speedup was **0.487x**, with a range of **0.103–1.358x**.
This implementation establishes functional compatibility; performance parity
requires further C550 tuning.

| API / shape / batch | FlagFFT median ms | mcFFT median ms | Speedup |
|---|---:|---:|---:|
| C2C / 4096 / 1 | 0.061440 | 0.022016 | 0.358x |
| Z2Z / 4096 / 1 | 0.030720 | 0.041728 | 1.358x |
| C2C / 65536 / 256 | 3.703552 | 0.574976 | 0.155x |
| Z2Z / 65536 / 256 | 14.690816 | 1.513728 | 0.103x |
| Z2Z / 8191 / 1 | 0.085760 | 0.097792 | 1.140x |
| Z2Z / 16x32x64 / 1 | 0.084736 | 0.035584 | 0.420x |

Full JSON reports, paths, process wall times and incremental CSV are under
`/workspace/FlagFFT-results/performance_baseline/`. The validated native
NumPy suites are stored next to that directory: 384 cases for the initial
four-warp compatibility checkpoint and 120 cases for the final settings.
Across these suites, FlagFFT's maximum relative L2 errors were 2.627e-7
for FP32 and 2.303e-15 for FP64.

The next performance work should compare large FP32 leaves with Four-Step
alternatives, calibrate the portable register-exchange cost model, and compare
Four-Step fused I/O with explicit tiled transpose and batch packing. The
4096-point batch-1 FP32 baseline chose a 4096-point leaf; FP64 chose 64x64
Four-Step. These measurements identify candidates for comparison rather than
proving the cause of the performance gap. A working mctle lowering would
also allow the existing TLE path to be evaluated separately.
