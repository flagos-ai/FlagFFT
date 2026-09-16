"""C API regressions for the Ascend backend, using an independent CPU oracle.

Requires a BACKEND=NPU build in FLAGFFT_TEST_LIBRARY and the matching CANN
environment. No reference GPU FFT library is used.
"""

import ctypes
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

HANDLE = ctypes.c_void_p
INT = ctypes.c_int
PTR = ctypes.c_void_p
SHAPES = [(1,), (2,), (4,), (8,), (30,), (256,), (257,), (65536,),
          (8, 16), (7, 9), (4, 5, 8), (3, 7, 9)]
IDS = ["x".join(map(str, shape)) for shape in SHAPES]
TYPES = {"c2c": 0x29, "r2c": 0x2a, "c2r": 0x2c,
         "z2z": 0x69, "d2z": 0x6a, "z2d": 0x6c}


@pytest.fixture(scope="module")
def api():
    library = os.environ.get("FLAGFFT_TEST_LIBRARY")
    if not library or not torch.npu.is_available():
        pytest.skip("set FLAGFFT_TEST_LIBRARY to an NPU libflagfft.so on an Ascend host")
    torch.npu.set_device(int(os.environ.get("FLAGFFT_TEST_DEVICE", "0")))
    api = ctypes.CDLL(library)
    dims = ctypes.POINTER(INT)
    api.flagfftPlanMany.argtypes = [ctypes.POINTER(HANDLE), INT, dims, dims, INT,
                                   INT, dims, INT, INT, INT, INT]
    api.flagfftPlanMany.restype = INT
    api.flagfftSetStream.argtypes = [HANDLE, PTR]
    api.flagfftSetStream.restype = INT
    api.flagfftDestroy.argtypes = [HANDLE]
    api.flagfftDestroy.restype = INT
    for name in TYPES:
        fun = getattr(api, "flagfftExec" + name.upper())
        fun.argtypes = [HANDLE, PTR, PTR] + ([INT] if name in ("c2c", "z2z") else [])
        fun.restype = INT
    return api


def run_fft(api, shape, batch, operation, user_stream, inplace=False):
    double = operation in ("z2z", "d2z", "z2d")
    real_dtype = np.float64 if double else np.float32
    complex_dtype = np.complex128 if double else np.complex64
    real_forward = operation in ("r2c", "d2z")
    real_inverse = operation in ("c2r", "z2d")
    compact = (*shape[:-1], shape[-1] // 2 + 1)
    input_shape = compact if real_inverse else shape
    output_shape = compact if real_forward else shape
    plan = HANDLE()
    dim_array = (INT * len(shape))(*shape)
    assert api.flagfftPlanMany(ctypes.byref(plan), len(shape), dim_array, None, 1,
                              int(np.prod(input_shape)), None, 1,
                              int(np.prod(output_shape)), TYPES[operation], batch) == 0
    stream = torch.npu.Stream() if user_stream else None
    try:
        assert api.flagfftSetStream(plan, stream.npu_stream if stream else None) == 0
        rng = np.random.default_rng(20260916)
        n = int(np.prod(shape))
        axes = tuple(range(1, len(shape) + 1))
        for pattern in ("random", "impulse", "constant", "tone"):
            source = rng.standard_normal((batch, *shape))
            if not (real_forward or real_inverse):
                source = source + 1j * rng.standard_normal(source.shape)
            if pattern == "impulse":
                source.fill(0)
                source.reshape(batch, -1)[:, 0] = 1
            elif pattern == "constant":
                source.fill(1)
            elif pattern == "tone":
                source = np.broadcast_to(np.cos(2 * np.pi * np.arange(n) / n),
                                         (batch, n)).reshape(batch, *shape).copy()
            source = source.astype(real_dtype if real_forward or real_inverse else complex_dtype)
            if real_inverse:
                source = np.fft.rfftn(source, axes=axes).astype(complex_dtype)
            directions = (-1, 1) if not (real_forward or real_inverse) else (1 if real_inverse else -1,)
            for direction in directions:
                if real_forward:
                    expected = np.fft.rfftn(source.astype(np.float64), axes=axes)
                elif real_inverse:
                    expected = n * np.fft.irfftn(source.astype(np.complex128), s=shape, axes=axes)
                elif direction == -1:
                    expected = np.fft.fftn(source.astype(np.complex128), axes=axes)
                else:
                    expected = n * np.fft.ifftn(source.astype(np.complex128), axes=axes)
                raw = np.ascontiguousarray(source).view(real_dtype).reshape(-1)
                # Byte tensors preserve FP64 bit patterns even when torch_npu
                # cannot construct float64 tensors. The C API sees raw pointers.
                source_tensor = torch.from_numpy(raw.view(np.uint8).copy()).to("npu")
                output_bytes = expected.size * np.dtype(real_dtype).itemsize * (1 if real_inverse else 2)
                storage = torch.full((output_bytes + 128,), 0xA5, dtype=torch.uint8, device="npu")
                output = source_tensor if inplace else storage[64:-64]
                torch.npu.synchronize()
                call = getattr(api, "flagfftExec" + operation.upper())
                args = [plan, source_tensor.data_ptr(), output.data_ptr()]
                if not (real_forward or real_inverse):
                    args.append(direction)
                # Repeated out-of-place executions exercise scratch reuse. For
                # in-place transforms each input is transformed exactly once.
                for _ in range(1 if inplace else 3):
                    assert call(*args) == 0
                stream.synchronize() if stream else torch.npu.synchronize()
                actual = output.cpu().numpy().copy().view(real_dtype if real_inverse else complex_dtype)
                actual = actual.reshape(expected.shape)
                tolerance = 2e-11 if double else 3e-5
                scale = max(1.0, float(np.linalg.norm(source.reshape(batch, -1), axis=1).max()))
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)
                relative = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
                assert np.isfinite(actual).all() and relative < tolerance
                if not inplace:
                    guard = storage.cpu().numpy()
                    assert (guard[:64] == 0xA5).all() and (guard[-64:] == 0xA5).all()
                    np.testing.assert_array_equal(source_tensor.cpu().numpy(), raw.view(np.uint8))
    finally:
        torch.npu.synchronize()
        assert api.flagfftDestroy(plan) == 0


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
@pytest.mark.parametrize("batch,user_stream", [(1, False), (3, True)])
@pytest.mark.parametrize("operation", ["c2c", "r2c", "c2r"])
def test_fp32(api, shape, batch, user_stream, operation):
    run_fft(api, shape, batch, operation, user_stream)


@pytest.mark.parametrize("shape", [(8,), (256,), (8, 16), (4, 5, 8)])
def test_c2c_inplace(api, shape):
    run_fft(api, shape, 3, "c2c", True, inplace=True)


@pytest.mark.parametrize("batch", [65, 257])
def test_large_batch(api, batch):
    run_fft(api, (8,), batch, "c2c", True)
