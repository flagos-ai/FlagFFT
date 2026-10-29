"""Execute generated pair boundaries on NumPy buffers, including masked padding."""
from types import SimpleNamespace
import numpy as np
import pytest
from flagfft_codegen.kernels_real import _build_real_pair_kernel_source


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


def run_boundary(n, dtype, mode, source, output, input_distance, output_distance, batch):
    ids = [0, 0]

    def load(pointer, mask, other):
        offset = np.asarray(pointer.offset)
        mask = np.broadcast_to(mask, offset.shape)
        result = np.full(offset.shape, other, dtype=pointer.data.dtype)
        result[mask] = pointer.data[offset[mask]]
        return result

    def store(pointer, value, mask):
        offset = np.asarray(pointer.offset)
        mask = np.broadcast_to(mask, offset.shape)
        values = np.broadcast_to(value, offset.shape)
        pointer.data[offset[mask]] = values[mask]

    tl = SimpleNamespace(program_id=lambda axis: ids[axis], arange=np.arange,
                         where=np.where, minimum=np.minimum, load=load, store=store)
    name, generated, _, _ = _build_real_pair_kernel_source(n, dtype, mode)
    scope = {'triton': SimpleNamespace(jit=lambda fn: fn), 'tl': tl}
    exec(generated, scope)
    extent = n // 2 + 1 if mode == 'r2c_pair_unpack' else n
    for pair in range((batch + 1) // 2):
        for block in range((extent + 255) // 256):
            ids[:] = [block, pair]
            scope[name](Pointer(source), Pointer(output), input_distance, output_distance, batch)


@pytest.mark.parametrize('n', [23, 76, 997])
@pytest.mark.parametrize('batch', [3, 8])
@pytest.mark.parametrize('precision', [32, 64])
@pytest.mark.parametrize('inverse', [False, True])
def test_real_pair_generated_boundaries_match_numpy(n, batch, precision, inverse):
    real_dtype = np.float32 if precision == 32 else np.float64
    complex_dtype = np.complex64 if precision == 32 else np.complex128
    dtype = 'complex64' if precision == 32 else 'complex128'
    rng = np.random.default_rng(27)
    original = rng.standard_normal((batch, n)).astype(real_dtype)
    half, pairs = n // 2 + 1, (batch + 1) // 2
    values = np.fft.rfft(original).astype(complex_dtype) if inverse else original
    input_distance = values.shape[1] + 3
    output_distance = (n if inverse else half) + 5
    input_values = np.zeros((batch, input_distance), dtype=complex_dtype if inverse else real_dtype)
    input_values[:, :values.shape[1]] = values
    packed = np.zeros(pairs * n * 2, dtype=real_dtype)
    mode = 'c2r' if inverse else 'r2c'
    run_boundary(n, dtype, mode + '_pair_pack', input_values.view(real_dtype).ravel(), packed,
                 input_distance, output_distance, batch)
    spectrum = packed.view(complex_dtype).reshape(pairs, n)
    transformed = (np.fft.ifft(spectrum) * n if inverse else np.fft.fft(spectrum)).astype(complex_dtype)
    output = np.full((batch, output_distance), 123456,
                     dtype=real_dtype if inverse else complex_dtype)
    run_boundary(n, dtype, mode + '_pair_unpack', transformed.view(real_dtype).ravel(),
                 output.view(real_dtype).ravel(), input_distance, output_distance, batch)
    expected = np.fft.irfft(values, n=n) * n if inverse else np.fft.rfft(values)
    tolerance = 3e-6 if precision == 32 else 3e-13
    np.testing.assert_allclose(output[:, :expected.shape[1]], expected,
                               rtol=tolerance, atol=tolerance * n)
    assert np.all(output[:, expected.shape[1]:] == 123456)
