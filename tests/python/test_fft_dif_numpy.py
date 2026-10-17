"""Check DIF autosort stages, paired prime codelets and real IO composition."""

import ast
from pathlib import Path

import numpy as np
import pytest

from flagfft_codegen.kernels_fft_stages import FFTStageConfig, emit_fft_stage, stage_twiddle_values
from test_tensor_fft_numpy import Language, Pointer


@pytest.mark.parametrize("factors", [(4, 4), (8, 8), (2, 3, 5, 7), (11, 13), (13, 17),
    (17, 19), (23,), (29,), (3, 23)])
@pytest.mark.parametrize("mode,direction", [("complex", "forward"), ("complex", "inverse"),
    ("r2c", "forward"), ("c2r", "inverse")])
@pytest.mark.parametrize("block,pack,stride,permuted", [(8, 1, 1, 0), (16, 4, 3, 0), (8, 2, 1, 7)])
def test_dif_sequence(tmp_path, factors, mode, direction, block, pack, stride, permuted):
    n, batch = int(np.prod(factors)), 5
    rng = np.random.default_rng(20261017)
    real = rng.normal(size=(batch, n)).astype(np.float32)
    if mode == "complex":
        logical = (real + 1j * rng.normal(size=real.shape)).astype(np.complex64)
        expected = np.fft.fft(logical) if direction == "forward" else np.fft.ifft(logical) * n
    elif mode == "r2c":
        logical, expected = real, np.fft.rfft(real)
    else:
        logical = np.fft.rfft(real).astype(np.complex64)
        expected = np.fft.irfft(logical, n=n) * n
    ilen, olen = logical.shape[1], expected.shape[1]
    source = np.zeros(((batch + stride - 1) // stride) * ilen * stride, logical.dtype)
    for row in range(batch):
        base = row // stride * ilen * stride + row % stride
        source[base:base + ilen * stride:stride] = logical[row]
    span = n
    for stage, radix in enumerate(factors):
        last = span == radix
        output_stride = stride if last and not permuted else 1
        extent = permuted if last else 0
        config = FFTStageConfig(n, radix, span, mode=mode, direction=direction, block=block,
            rows_per_program=pack, input_stride=stride if stage == 0 else 1,
            output_stride=output_stride, output_rows_per_batch=extent)
        meta = emit_fft_stage(config, tmp_path / f"stage{stage}")
        tree = ast.parse(Path(meta["module_path"]).read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            function.decorator_list = []
        language = Language()
        namespace = {"tl": language}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "generated_dif", "exec"), namespace)
        size = (olen if last else n) * (extent or output_stride) * (
            (batch + (extent or output_stride) - 1) // (extent or output_stride))
        destination = np.full(size, np.nan, np.float32 if last and mode == "c2r" else np.complex64)
        twiddle = np.asarray(stage_twiddle_values(config), np.float32)
        count = meta["programs_per_row_group"] * ((batch + pack - 1) // pack)
        for program in range(count):
            language.pid = 0
            namespace[meta["kernel_name"]](Pointer(source.view(np.float32)),
                Pointer(destination.view(np.float32)), Pointer(twiddle), batch, program)
        source = destination
        span //= radix
    assert span == 1
    actual = np.empty_like(expected)
    step = permuted or stride
    for row in range(batch):
        base = row // step * olen * step + row % step
        actual[row] = source[base:base + olen * step:step]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-7
