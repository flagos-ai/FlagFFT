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
    "FLAGFFT_MACA_3D_N32_FP64_PACK",
    "FLAGFFT_MACA_3D_PERMSTORE_PACK",
    "FLAGFFT_MACA_3D_FINAL_STORE",
    "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
    "FLAGFFT_MACA_FOURSTEP_ROW_PACK", "FLAGFFT_MACA_FOURSTEP_COL_PACK",
    "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
    "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
    "FLAGFFT_MACA_MIXED_EXCHANGE", "FLAGFFT_MACA_COMPACT_STAGES",
    "FLAGFFT_MACA_RADER_CODELET", "FLAGFFT_MACA_RADER_FULL_LEAF",
    "FLAGFFT_MACA_RADER_BOUNDARY_LEAF", "FLAGFFT_MACA_RADER_VEC_IO",
    "FLAGFFT_MACA_3D_C2C32_MORE_LANES",
)


def set_maca_tail_mode(mode):
    if mode not in {"off", "p4w4", "real-direct", "batch-vecio",
                    "batch-real-pack2", "batch-c2c-pack2", "batch-prime-vecio",
                    "batch-c2c-tree", "batch-prime-rader-vecio", "c2c663000", "c2c663000-batch",
                    "r2c663000", "batch-c2r663000", "batch-c2r663000-split", "c2c340200-split",
                    "c2c340200-single",
                    "z2z340200-split", "batch-real340200-split", "batch-real16384-split",
                    "batch-real524288-row4", "batch-real524288-split",
                    "batch-real185640-p2", "batch-real185640-p4",
                    "z2z16-pack16", "z2z16384-split", "z2z524288-pack4"}:
        raise ValueError(f"Invalid MACA tail mode: {mode!r}")
    return _root_mode.set(mode)


def reset_maca_tail_mode(token):
    _root_mode.reset(token)


def eligible_kernel_mode(kernel, length, dtype, n1=0, n2=0):
    if backend_name() != "maca":
        return "off"
    if (dtype == "complex128" and n2 == 1024
            and (_root_mode.get(), n1) in {("p4w4", 1024), ("z2z524288-pack4", 512)}
            and ((kernel == "four_step_row" and length == n1)
                 or (kernel == "four_step_col" and length == n2))):
        return _root_mode.get()
    if (_root_mode.get() == "real-direct" and 2 <= length <= 37
            and dtype in {"complex64", "complex128"}
            and kernel in {"direct_dft_r2c", "direct_dft_c2r"}):
        return "real-direct"
    if (_root_mode.get() == "batch-c2c-tree" and kernel == "direct_dft"
            and length == 23 and dtype == "complex128"):
        return "batch-c2c-tree"
    if (_root_mode.get() == "z2z16-pack16" and kernel == "leaf"
            and length == 16 and dtype == "complex128"):
        return "z2z16-pack16"
    if (_root_mode.get() == "c2c340200-single" and (dtype, n1, n2) == ("complex64", 900, 378)
            and (kernel, length) in {("four_step_row", n1), ("four_step_col", n2)}):
        return "c2c340200-single"
    if kernel in {"four_step_row", "four_step_col"} and length in {n1, n2}:
        if dtype == "complex64":
            if _root_mode.get() == "c2c663000" and (n1, n2) == (375, 1768):
                return "c2c663000"
            if _root_mode.get() == "c2c663000-batch" and (n1, n2) == (780, 850):
                return "c2c663000-batch"
            if _root_mode.get() == "r2c663000" and (n1, n2) == (300, 1105):
                return "r2c663000"
            if _root_mode.get() == "batch-c2r663000-split" and (n1, n2) == (340, 975):
                return "batch-c2r663000-split"
        if _root_mode.get() == "c2c340200-split" and (dtype, n1, n2) == ("complex64", 567, 600):
            return "c2c340200-split"
        if _root_mode.get() == "z2z340200-split" and (dtype, n1, n2) == ("complex128", 504, 675):
            return "z2z340200-split"
        if _root_mode.get() == "batch-real340200-split" and (dtype, n1, n2) == ("complex64", 405, 420):
            return "batch-real340200-split"
        if (_root_mode.get() == "batch-real16384-split" and dtype in {"complex64", "complex128"}
                and (n1, n2) == (128, 64)):
            return "batch-real16384-split"
        if _root_mode.get() == "batch-real524288-row4" and (dtype, n1, n2) == ("complex64", 512, 512):
            return "batch-real524288-row4" if kernel == "four_step_row" else "batch-vecio"
        if _root_mode.get() == "batch-real524288-split" and (dtype, n1, n2) == ("complex64", 256, 1024):
            return "batch-real524288-split"
        if (n1, n2) == (420, 221) and (_root_mode.get(), dtype) in {
            ("batch-real185640-p2", "complex64"), ("batch-real185640-p4", "complex128"),
        }:
            return _root_mode.get()
        if _root_mode.get() == "z2z16384-split" and (dtype, n1, n2) == ("complex128", 256, 64):
            return "z2z16384-split"
        if (_root_mode.get() == "batch-c2r663000" and dtype in {"complex64", "complex128"}
                and (n1, n2) == (300, 1105)):
            return "batch-c2r663000"
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
def maca_tail_kernel_scope(kernel, length, dtype, n1=0, n2=0, *, perm_form="outer"):
    mode = eligible_kernel_mode(kernel, length, dtype, n1, n2)
    if backend_name() == "maca" and perm_form == "maca_packed_c2r_row":
        mode = perm_form
    token = _kernel_mode.set(mode)
    try:
        yield
    finally:
        _kernel_mode.reset(token)


def resource_default(name):
    if backend_name() != "maca":
        return None
    if _kernel_mode.get() in {"p4w4", "z2z524288-pack4"}:
        return {"FP64_REGISTER_PACK": "1", "INNER_PACK": "4", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "batch-c2c-tree":
        return {"REAL_DFT_REDUCTION": "tree"}.get(name)
    if _kernel_mode.get() == "z2z16-pack16":
        return {"BATCH_PACK": "16"}.get(name)
    if _kernel_mode.get() in {"c2c663000", "r2c663000"}:
        return {"RADER_CODELET": "1", "VEC_IO": "packed"}.get(name)
    if _kernel_mode.get() == "batch-c2r663000-split":
        return {"RADER_CODELET": "1", "VEC_IO": "packed",
                "FOURSTEP_ROW_PACK": "4", "FOURSTEP_COL_PACK": "2"}.get(name)
    if _kernel_mode.get() == "c2c340200-split":
        return {"VEC_IO": "packed", "INNER_PACK": "2", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "c2c340200-single":
        return {"VEC_IO": "packed", "MAX_WARPS": "8"}.get(name)
    if _kernel_mode.get() == "z2z340200-split":
        return {"VEC_IO": "packed", "FOURSTEP_ROW_PACK": "2", "FOURSTEP_COL_PACK": "1",
                "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "batch-real340200-split":
        return {"VEC_IO": "packed", "FOURSTEP_ROW_PACK": "2", "FOURSTEP_COL_PACK": "4",
                "MAX_WARPS": "8"}.get(name)
    if _kernel_mode.get() == "batch-real16384-split":
        return {"VEC_IO": "1", "EXCHANGE": "direct_all", "INNER_PACK": "8", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "batch-real524288-row4":
        return {"VEC_IO": "1", "FOURSTEP_ROW_PACK": "4"}.get(name)
    if _kernel_mode.get() == "batch-real524288-split":
        return {"VEC_IO": "1", "FOURSTEP_ROW_PACK": "4", "FOURSTEP_COL_PACK": "8"}.get(name)
    if _kernel_mode.get() == "batch-real185640-p2":
        return {"VEC_IO": "packed", "INNER_PACK": "2", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "batch-real185640-p4":
        return {"VEC_IO": "packed", "INNER_PACK": "4", "MAX_WARPS": "8"}.get(name)
    if _kernel_mode.get() == "z2z16384-split":
        return {"VEC_IO": "1", "EXCHANGE": "direct_all", "FOURSTEP_ROW_PACK": "4",
                "FOURSTEP_COL_PACK": "8", "MAX_WARPS": "4"}.get(name)
    if _kernel_mode.get() == "maca_packed_c2r_row":
        return {"VEC_IO": "packed", "EXCHANGE": "direct_all", "INNER_PACK": "4"}.get(name)
    if _kernel_mode.get() == "batch-c2r663000":
        return {"VEC_IO": "packed"}.get(name)
    if _kernel_mode.get() == "c2c663000-batch":
        return {"RADER_CODELET": "1", "VEC_IO": "packed",
                "FOURSTEP_ROW_PACK": "2", "FOURSTEP_COL_PACK": "4"}.get(name)
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
