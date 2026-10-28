"""Execute the NPU half-spectrum expansion, including odd highest bins."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flagfft_codegen import kernels_real
from test_npu_structured_fft import Pointer, execute_module


@pytest.mark.parametrize("n", [16, 17, 1009, 1010])
@pytest.mark.parametrize("batch", [1, 3])
def test_half_spectrum_expansion(tmp_path, monkeypatch, n, batch):
    monkeypatch.setattr(kernels_real, "current_profile", lambda: SimpleNamespace(backend="npu"))
    name, source, _, rows = kernels_real._build_compact_to_hermitian_full_kernel_source(n, "complex64")
    path = tmp_path / "expand.py"
    path.write_text(source)
    ns, pid = execute_module(path)
    ns["tle"] = SimpleNamespace(dsa=ns["al"])
    half = n // 2 + 1
    # A nonzero highest-bin imaginary part is valid for odd lengths only.
    torch.manual_seed(1009)
    compact = torch.randn(batch, half, 2)
    compact[:, -1, 1] = 7.25
    saved = compact.clone()
    expected = compact.clone()
    expected[:, 0, 1] = 0
    if n % 2 == 0:
        expected[:, -1, 1] = 0
    mirror = expected[:, 1:half - (n % 2 == 0)].flip(1).clone()
    mirror[:, :, 1] *= -1
    expected = torch.cat((expected, mirror), dim=1)
    guarded = torch.full((batch * n * 2 + 128,), 31415.25)
    block_cols, _ = kernels_real._packed_layout(n, 256)
    for by in range((batch + rows - 1) // rows):
        pid[1] = by
        for bx in range((n + block_cols - 1) // block_cols):
            pid[0] = bx
            ns[name](Pointer(compact), Pointer(guarded, 64), half, batch)
    assert torch.equal(guarded[64:-64].reshape(batch, n, 2), expected)
    assert torch.equal(compact, saved)
    assert (guarded[:64] == 31415.25).all()
    assert (guarded[-64:] == 31415.25).all()
