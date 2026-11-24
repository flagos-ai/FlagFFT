"""Check generated prime-convolution boundaries over multiple launch tiles."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flagfft_codegen import emit
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from test_npu_structured_fft import Pointer, execute_module


@pytest.fixture(params=("cuda", "npu"))
def backend(request):
    token = set_profile(BackendProfile.from_device({
        "backend": request.param,
        "device_arch": "Ascend910B4" if request.param == "npu" else "sm_80",
        "warp_size": 1 if request.param == "npu" else 32,
        "max_threads_per_block": 1024,
        "max_dynamic_shared_memory": 192 * 1024,
    }, policy="legacy"))
    yield request.param
    reset_profile(token)


def _run(metadata, arguments, columns, batch):
    ns, pid = execute_module(metadata["module_path"])
    ns["tle"] = SimpleNamespace(dsa=ns["al"])
    ns["_cmul"] = lambda ar, ai, br, bi: (ar * br - ai * bi, ar * bi + ai * br)
    for b in range(batch):
        pid[1] = b
        for block in range(math.ceil(columns / metadata["block"])):
            pid[0] = block
            ns[metadata["kernel_name"]](*arguments)


def _guarded(shape):
    count = math.prod(shape)
    guard = torch.full((count + 128,), 31415.25)
    return guard, Pointer(guard, 64), guard[64:-64].reshape(shape)


def _check_guard(guard):
    assert (guard[:64] == 31415.25).all() and (guard[-64:] == 31415.25).all()


def test_bluestein_boundary_covers_all_tiles(backend, tmp_path):
    torch.manual_seed(113)
    n, m, batch = 701, 2048, 2
    x = torch.randn(batch, n, 2)
    chirp = torch.randn(n, 2)
    a = torch.randn(batch, m, 2)
    b = torch.randn(m, 2)
    multiply = lambda x, y: torch.view_as_real(torch.view_as_complex(x) * torch.view_as_complex(y))

    for kind in ("bluestein_prepare", "bluestein_pointwise", "bluestein_finalize"):
        meta = emit._emit_bluestein_jit_kernel(kernel=kind, n=n, m=m, out_dir=tmp_path)
        width = n if kind == "bluestein_finalize" else m
        guard, y, output = _guarded((batch, width, 2))
        output.fill_(float("nan"))
        if kind == "bluestein_prepare":
            args = [Pointer(x), Pointer(chirp), y, n, m, batch]
            expected = torch.zeros_like(output)
            expected[:, :n] = multiply(x, chirp)
        elif kind == "bluestein_pointwise":
            args = [Pointer(a), Pointer(b), y, m, batch]
            expected = multiply(a, b)
            expected[..., 1] *= -1
        else:
            args = [Pointer(a), Pointer(chirp), y, n, m, batch]
            value = a[:, :n].clone() / m
            value[..., 1] *= -1
            expected = multiply(value, chirp)
        _run(meta, args, width, batch)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        _check_guard(guard)


def test_rader_boundary_covers_all_tiles(backend, tmp_path):
    torch.manual_seed(114)
    n, m, batch = 1009, 1008, 2
    x = torch.randn(batch, n, 2)
    a = torch.randn(batch, m, 2)
    b = torch.randn(m, 2)
    indices = torch.arange(1, n, dtype=torch.int64)

    prepare = emit._emit_rader_jit_kernel(kernel="rader_prepare", n=n, m=m, out_dir=tmp_path)
    guard, output, prepared = _guarded((batch, m, 2))
    prepared.fill_(float("nan"))
    _run(prepare, [Pointer(x), Pointer(indices), output, n, m, batch], m, batch)
    order = indices[(-torch.arange(m)) % m]
    torch.testing.assert_close(prepared, x[:, order])
    _check_guard(guard)

    pointwise = emit._emit_rader_jit_kernel(kernel="rader_pointwise", n=n, m=m, out_dir=tmp_path)
    guard, output, product = _guarded((batch, m, 2))
    dc_guard, dc_output, dc = _guarded((batch, n, 2))
    product.fill_(float("nan"))
    _run(pointwise, [Pointer(a), Pointer(b), output, Pointer(x), dc_output, n, m, batch], m, batch)
    expected = torch.view_as_real(torch.view_as_complex(a) * torch.view_as_complex(b))
    expected[..., 1] *= -1
    torch.testing.assert_close(product, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(dc[:, 0], a[:, 0] + x[:, 0])
    _check_guard(guard)
    _check_guard(dc_guard)

    finalize = emit._emit_rader_jit_kernel(kernel="rader_finalize", n=n, m=m, out_dir=tmp_path)
    guard, output, final = _guarded((batch, n, 2))
    final.fill_(float("nan"))
    _run(finalize, [Pointer(x), Pointer(a), Pointer(indices), output, n, m, batch], m, batch)
    expected = a / m
    expected[..., 1] *= -1
    expected = expected + x[:, :1]
    torch.testing.assert_close(final[:, indices], expected, atol=1e-6, rtol=1e-5)
    _check_guard(guard)
