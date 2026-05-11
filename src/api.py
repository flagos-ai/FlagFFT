from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

import torch

try:
    import _flagfft_core
except ImportError:  # pragma: no cover - extension may be absent before build
    _flagfft_core = None


def _require_core():
    if _flagfft_core is None:
        raise ImportError("_flagfft_core is not built; install FlagFFT before calling the API")
    return _flagfft_core


def _not_implemented(name: str) -> NoReturn:
    raise NotImplementedError(f"flagfft.{name} is not implemented yet")


def fft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a 1-D complex FFT using the C++ FlagFFT backend."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"flagfft.fft expected a torch.Tensor, got {type(input).__name__}")

    result = _require_core().fft(input, n, dim, norm)
    if out is not None:
        out.copy_(result)
        return out
    return result


def ifft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a 1-D inverse complex FFT using the C++ FlagFFT backend."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"flagfft.ifft expected a torch.Tensor, got {type(input).__name__}")

    result = _require_core().ifft(input, n, dim, norm)
    if out is not None:
        out.copy_(result)
        return out
    return result


def fft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("fft2")


def ifft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("ifft2")


def fftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("fftn")


def ifftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("ifftn")


def rfft(input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("rfft")


def irfft(input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("irfft")


def rfft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("rfft2")


def irfft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("irfft2")


def rfftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("rfftn")


def irfftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("irfftn")


def hfft(input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("hfft")


def ihfft(input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("ihfft")


def hfft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("hfft2")


def ihfft2(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] = (-2, -1), norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("ihfft2")


def hfftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("hfftn")


def ihfftn(input: torch.Tensor, s: Sequence[int] | None = None, dim: Sequence[int] | None = None, norm: str | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    _not_implemented("ihfftn")


def fftfreq(n: int, d: float = 1.0, *, out: torch.Tensor | None = None, dtype: torch.dtype | None = None, layout: torch.layout = torch.strided, device: torch.device | str | None = None, requires_grad: bool = False) -> torch.Tensor:
    _not_implemented("fftfreq")


def rfftfreq(n: int, d: float = 1.0, *, out: torch.Tensor | None = None, dtype: torch.dtype | None = None, layout: torch.layout = torch.strided, device: torch.device | str | None = None, requires_grad: bool = False) -> torch.Tensor:
    _not_implemented("rfftfreq")


def fftshift(input: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    _not_implemented("fftshift")


def ifftshift(input: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    _not_implemented("ifftshift")


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
