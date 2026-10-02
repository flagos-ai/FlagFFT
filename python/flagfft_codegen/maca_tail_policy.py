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
    "FLAGFFT_MACA_BATCH_PACK",
    "FLAGFFT_MACA_3D_N64_PACK",
    "FLAGFFT_MACA_3D_PERMSTORE_PACK",
    "FLAGFFT_MACA_3D_MIDDLE_STORE",
    "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
    "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
    "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
    "FLAGFFT_MACA_MIXED_EXCHANGE", "FLAGFFT_MACA_RADER_FULL_LEAF",
    "FLAGFFT_MACA_RADER_BOUNDARY_LEAF", "FLAGFFT_MACA_RADER_VEC_IO",
)


def set_maca_tail_mode(mode):
    if mode not in {"off", "p4w4", "real-direct", "batch-vecio",
                    "batch-real-pack2", "batch-c2c-pack2", "batch-prime-vecio",
                    "batch-c2c-tree", "batch-prime-rader-vecio"}:
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
    if (_root_mode.get() == "real-direct" and 2 <= length <= 37
            and dtype in {"complex64", "complex128"}
            and kernel in {"direct_dft_r2c", "direct_dft_c2r"}):
        return "real-direct"
    if (_root_mode.get() == "batch-c2c-tree" and kernel == "direct_dft"
            and length == 23 and dtype == "complex128"):
        return "batch-c2c-tree"
    if kernel in {"four_step_row", "four_step_col"} and length in {n1, n2}:
        if (_root_mode.get() == "batch-vecio" and dtype in {"complex64", "complex128"}
                and (n1, n2) in {(256, 64), (128, 128), (64, 128), (390, 476),
                                 (512, 1024), (512, 512)}):
            return "batch-vecio"
        if (_root_mode.get() == "batch-real-pack2" and dtype == "complex128"
                and n1 * n2 == 92820):
            return "batch-real-pack2"
        if (_root_mode.get() == "batch-c2c-pack2" and dtype == "complex128"
                and (n1, n2) == (390, 476)):
            return "batch-c2c-pack2"
        if (_root_mode.get() == "batch-prime-vecio"
                and dtype in {"complex64", "complex128"}
                and (n1, n2) in {(90, 91), (91, 180)}):
            return "batch-prime-vecio"
    return "off"


@contextmanager
def maca_tail_kernel_scope(kernel, length, dtype, n1=0, n2=0):
    token = _kernel_mode.set(eligible_kernel_mode(kernel, length, dtype, n1, n2))
    try:
        yield
    finally:
        _kernel_mode.reset(token)


def resource_default(name):
    if backend_name() != "maca":
        return None
    if _kernel_mode.get() == "p4w4":
        return {"FP64_REGISTER_PACK": "1", "INNER_PACK": "4", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "batch-c2c-tree":
        return {"REAL_DFT_REDUCTION": "tree"}.get(name)
    if _kernel_mode.get() in {"batch-vecio", "batch-prime-vecio"}:
        return {"VEC_IO": "1"}.get(name)
    if _kernel_mode.get() in {"batch-real-pack2", "batch-c2c-pack2"}:
        return {"INNER_PACK": "2"}.get(name)
    return None


def real_tree_default(n):
    return backend_name() == "maca" and n == 23 and _kernel_mode.get() == "real-direct"


def rader_vector_io_default():
    return backend_name() == "maca" and _root_mode.get() == "batch-prime-rader-vecio"


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
