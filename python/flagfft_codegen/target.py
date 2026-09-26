# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target capabilities used while emitting a module, without GPU initialization."""

from contextvars import ContextVar

_target = ContextVar("flagfft_codegen_target", default="")
_MACA_1D_SINGLE_DEFAULT = ContextVar("flagfft_maca_1d_single_default", default=False)
_IX_CT_SINGLE_DEFAULT = ContextVar("flagfft_ix_ct_single_default", default=False)
_IX_CT_BATCH_DEFAULT = ContextVar("flagfft_ix_ct_batch_default", default=False)
_IX_CT_SINGLE_TLE_DEFAULT = ContextVar("flagfft_ix_ct_single_tle_default", default=0)
_MACA_2D_SINGLE_DEFAULT = ContextVar("flagfft_maca_2d_single_default", default=False)


def set_ix_ct_single_tle_default(policy: int):
    if policy not in (0, 1, 2):
        raise ValueError("IX TLE single policy must be 0, 1 or 2")
    return _IX_CT_SINGLE_TLE_DEFAULT.set(policy)


def reset_ix_ct_single_tle_default(token):
    _IX_CT_SINGLE_TLE_DEFAULT.reset(token)


def ix_ct_single_tle_default() -> int:
    return _IX_CT_SINGLE_TLE_DEFAULT.get()


def set_ix_ct_single_default(enabled: bool):
    return _IX_CT_SINGLE_DEFAULT.set(bool(enabled))


def reset_ix_ct_single_default(token):
    _IX_CT_SINGLE_DEFAULT.reset(token)


def ix_ct_single_default_enabled() -> bool:
    return _IX_CT_SINGLE_DEFAULT.get()


def set_ix_ct_batch_default(enabled: bool):
    return _IX_CT_BATCH_DEFAULT.set(bool(enabled))


def reset_ix_ct_batch_default(token):
    _IX_CT_BATCH_DEFAULT.reset(token)


def ix_ct_batch_default_enabled() -> bool:
    return _IX_CT_BATCH_DEFAULT.get()


def set_codegen_target(target: str) -> None:
    if target and len(target.split(":")) != 3:
        raise ValueError(f"Invalid Triton target: {target!r}")
    if target and int(target.rsplit(":", 1)[1]) <= 0:
        raise ValueError(f"Invalid warp size in target: {target!r}")
    _target.set(target)


def set_maca_1d_single_default(enabled: bool):
    """Set the scoped MACA 1D single-transform code-generation policy."""
    return _MACA_1D_SINGLE_DEFAULT.set(bool(enabled))


def reset_maca_1d_single_default(token) -> None:
    _MACA_1D_SINGLE_DEFAULT.reset(token)


def maca_1d_single_default_enabled() -> bool:
    """Whether the native compiler requested the MACA 1D single defaults."""
    return _MACA_1D_SINGLE_DEFAULT.get()


def set_maca_2d_single_default(enabled: bool):
    return _MACA_2D_SINGLE_DEFAULT.set(bool(enabled))


def reset_maca_2d_single_default(token) -> None:
    _MACA_2D_SINGLE_DEFAULT.reset(token)


def maca_2d_single_default_enabled() -> bool:
    """Root 2D FP32 batch-1 policy, including its batched axis kernels."""
    return _MACA_2D_SINGLE_DEFAULT.get()


def backend_name() -> str:
    target = _target.get()
    return target.split(":", 1)[0] if target else ""


def warp_size() -> int:
    target = _target.get()
    if target:
        return int(target.rsplit(":", 1)[1])
    # HCU exposes a HIP-compatible runtime but uses 64-lane wavefronts.
    # Keep direct codegen invocations correct even before a device profile has
    # been installed in the context.
    import os

    declared = (
        os.environ.get("TRITON_JIT_BACKEND", "")
        or os.environ.get("FLAGTREE_BACKEND", "")
    ).lower()
    if declared in {"hcu", "hygon"}:
        return 64
    try:
        from triton._C import libtriton

        if hasattr(libtriton, "hcu"):
            return 64
        if hasattr(libtriton, "metax"):
            return 64
    except ImportError:
        pass
    return 32
