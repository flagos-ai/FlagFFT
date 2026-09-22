"""Closed, native-request-scoped MACA tail policy (never mutates os.environ)."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os

from .target import backend_name

_root_mode = ContextVar("maca_tail_root_mode", default="off")
_kernel_mode = ContextVar("maca_tail_kernel_mode", default="off")
ENV_NAMES = (
    "FLAGFFT_MACA_REAL_DFT_REDUCTION", "FLAGFFT_MACA_EXCHANGE",
    "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
    "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
    "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
    "FLAGFFT_MACA_MIXED_EXCHANGE",
)


def set_maca_tail_mode(mode):
    if mode not in {"off", "p4w4", "real23"}:
        raise ValueError(f"Invalid MACA tail mode: {mode!r}")
    return _root_mode.set(mode)


def reset_maca_tail_mode(token):
    _root_mode.reset(token)


def eligible_kernel_mode(kernel, length, dtype, n1=0, n2=0):
    if backend_name() != "maca":
        return "off"
    if (_root_mode.get() == "p4w4" and dtype == "complex128"
            and length == n1 == n2 == 1024
            and kernel in {"four_step_row", "four_step_col"}):
        return "p4w4"
    if (_root_mode.get() == "real23" and length == 23
            and dtype in {"complex64", "complex128"}
            and kernel in {"direct_dft_r2c", "direct_dft_c2r"}):
        return "real23"
    return "off"


@contextmanager
def maca_tail_kernel_scope(kernel, length, dtype, n1=0, n2=0):
    token = _kernel_mode.set(eligible_kernel_mode(kernel, length, dtype, n1, n2))
    try:
        yield
    finally:
        _kernel_mode.reset(token)


def resource_default(name):
    if backend_name() == "maca" and _kernel_mode.get() == "p4w4":
        return {"FP64_REGISTER_PACK": "1", "INNER_PACK": "4", "MAX_WARPS": "4"}.get(name)
    return None


def real_tree_default(n):
    return backend_name() == "maca" and n == 23 and _kernel_mode.get() == "real23"


def variant_suffix(*, root=False):
    """Separate module/filesystem identities, including standalone legacy tree."""
    if backend_name() != "maca":
        return ""
    mode = _root_mode.get() if root else _kernel_mode.get()
    env = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    if mode == "off" and not env:
        return ""
    payload = json.dumps({"version": 1, "mode": mode, "env": env}, sort_keys=True)
    return "_tail_" + hashlib.sha256(payload.encode()).hexdigest()[:20]
