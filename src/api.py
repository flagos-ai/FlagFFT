from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NoReturn

import torch

from .exec import fft_mixed_radix_triton


def _not_implemented(name: str) -> NoReturn:
    raise NotImplementedError(f"flagfft.{name} is not implemented yet")


def _normalize_dim(dim: int, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def fft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a 1-D complex FFT using the FlagFFT backend."""
    if input.ndim == 0:
        raise ValueError("flagfft.fft expected at least a 1-D tensor")

    resolved_dim = _normalize_dim(dim, input.ndim)
    if resolved_dim != input.ndim - 1:
        raise NotImplementedError("flagfft.fft currently supports only the last dimension")

    if n is not None and n != input.shape[resolved_dim]:
        raise NotImplementedError("flagfft.fft currently does not support padding or trimming with n")

    if norm not in (None, "backward"):
        raise NotImplementedError("flagfft.fft currently supports only norm=None or norm='backward'")

    result = fft_mixed_radix_triton(input)
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
    _not_implemented("ifft")


def fft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("fft2")


def ifft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("ifft2")


def fftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("fftn")


def ifftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("ifftn")


def rfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("rfft")


def irfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("irfft")


def rfft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("rfft2")


def irfft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("irfft2")


def rfftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("rfftn")


def irfftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("irfftn")


def hfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("hfft")


def ihfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("ihfft")


def hfft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("hfft2")


def ihfft2(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("ihfft2")


def hfftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("hfftn")


def ihfftn(
    input: torch.Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _not_implemented("ihfftn")


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    _not_implemented("fftfreq")


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    _not_implemented("rfftfreq")


def fftshift(input: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    _not_implemented("fftshift")


def ifftshift(input: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    _not_implemented("ifftshift")


_FFT_API_NAMES = (
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
)


def get_api_functions() -> list[Callable[..., torch.Tensor]]:
    """Return the public torch.fft-compatible FlagFFT API functions."""
    return [globals()[name] for name in _FFT_API_NAMES]


__all__ = [*_FFT_API_NAMES, "get_api_functions"]
