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
The fixed libtriton_jit revision includes the MACA consumer CMake fixes:
public `USE_MACA`, a globally visible `MACA::mcruntime` imported target, and
respect for an explicit `MACA_PATH`.

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

The planner and codegen use 64 threads per warp, at least four warps and one transform per
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

MACA compiles in the existing codegen Python process using libtriton_jit's
`standalone_compile.py`, then loads `.mcfatbin` and launches raw arguments in
C++ through `TritonKernelImpl<MacaBackend>`. Compilation and native module
loading finish before graph capture. This process boundary avoids the SDK's
`libmcFlashAttn.so` teardown crash observed with embedded Torch in native
executables. Kernels requiring nonzero global/profile scratch are rejected;
the supported kernels use the backend ABI's two trailing null scratch pointers.

## Validation

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
/opt/conda/bin/python tools/numpy_fft_validation/validate.py \
    --capture-bin /workspace/FlagFFT-build/numpy-capture/numpy_fft_capture \
    --backend MACA --combination 1d_ct_single --shapes 256 \
    --gpu 4 --output-dir /workspace/FlagFFT-results/numpy_smoke
/opt/conda/bin/python tools/run_tests.py \
    --build-dir /workspace/FlagFFT-build/release --gpus 4 \
    --combination 1d_ct_single --accuracy-only \
    --output-dir /workspace/FlagFFT-results/accuracy
```

NumPy validation records FlagFFT and mcFFT independently. A vendor-library
failure or a process error remains a failure even if another implementation
passes. Benchmark compilation time is separate from event timings; measure
performance after correctness validation finishes on card 4.
