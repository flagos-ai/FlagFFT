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

"""Single source of truth for generated JIT kernel kinds.

Every ``--kernel`` value accepted by the codegen CLI is declared here with its
family, leaf I/O mode, required CLI flags, and generated module-name pattern.
Keeping this table centralized prevents the kernel-name strings from drifting
between ``jit_source.py``, ``kernels.py`` and the C++ runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CT_LEAF = "ct_leaf"
FOUR_STEP = "four_step"
BLUESTEIN_LEAF = "bluestein_leaf"
BLUESTEIN_FOUR_STEP = "bluestein_four_step"
DIRECT_DFT = "direct_dft"
BLUESTEIN = "bluestein"
RADER = "rader"
RESHAPE = "reshape"
TRANSPOSE = "transpose"
TRANSPOSE3D = "transpose3d"
REAL_POINTWISE = "real_pointwise"

LEAF_LIKE_FAMILIES = frozenset(
    {CT_LEAF, FOUR_STEP, BLUESTEIN_LEAF, BLUESTEIN_FOUR_STEP}
)
FOUR_STEP_FAMILIES = frozenset({FOUR_STEP, BLUESTEIN_FOUR_STEP})
BLUESTEIN_FAMILIES = frozenset({BLUESTEIN_LEAF, BLUESTEIN_FOUR_STEP, BLUESTEIN})


@dataclass(frozen=True)
class KernelSpec:
    name: str
    family: str
    io_mode: str | None = None
    requires: tuple[str, ...] = ()

    @property
    def is_leaf_like(self) -> bool:
        return self.family in LEAF_LIKE_FAMILIES

    @property
    def is_four_step(self) -> bool:
        return self.family in FOUR_STEP_FAMILIES


_BASE_LEAF_FLAGS = ("length", "factors", "lanes", "num_warps", "smem_size")
_FOUR_STEP_FLAGS = ("four_step_n1", "four_step_n2")
_BLUESTEIN_FLAGS = ("bluestein_n", "bluestein_m")
_RADER_FLAGS = ("rader_n", "rader_m")
_RESHAPE_FLAGS = ("reshape_n1", "reshape_n2")
_TRANSPOSE3D_FLAGS = (
    "transpose3d_n0",
    "transpose3d_n1",
    "transpose3d_n2",
    "transpose3d_order",
)

_SPECS: tuple[KernelSpec, ...] = (
    KernelSpec("leaf", CT_LEAF, io_mode="contiguous", requires=_BASE_LEAF_FLAGS),
    KernelSpec("leaf_strided", CT_LEAF, io_mode="strided", requires=_BASE_LEAF_FLAGS),
    KernelSpec(
        "leaf_r2c", CT_LEAF, io_mode="contiguous_r2c", requires=_BASE_LEAF_FLAGS
    ),
    KernelSpec(
        "leaf_c2r", CT_LEAF, io_mode="contiguous_c2r", requires=_BASE_LEAF_FLAGS
    ),
    KernelSpec(
        "leaf_bluestein",
        BLUESTEIN_LEAF,
        io_mode="bluestein_full_leaf",
        requires=_BASE_LEAF_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "leaf_bluestein_prepare",
        BLUESTEIN_LEAF,
        io_mode="bluestein_prepare_leaf",
        requires=_BASE_LEAF_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "leaf_bluestein_finish",
        BLUESTEIN_LEAF,
        io_mode="bluestein_finish_leaf",
        requires=_BASE_LEAF_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "bluestein_four_step_prepare_row",
        BLUESTEIN_FOUR_STEP,
        io_mode="bluestein_four_step_prepare_row",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "bluestein_four_step_pointwise_row",
        BLUESTEIN_FOUR_STEP,
        io_mode="bluestein_four_step_pointwise_row",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "bluestein_four_step_finish_col",
        BLUESTEIN_FOUR_STEP,
        io_mode="bluestein_four_step_finish_col",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS + ("bluestein_n",),
    ),
    KernelSpec(
        "direct_dft", DIRECT_DFT, requires=("length",)
    ),
    KernelSpec(
        "direct_dft_strided", DIRECT_DFT, requires=("length",)
    ),
    KernelSpec(
        "four_step_row",
        FOUR_STEP,
        io_mode="four_step_row",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_row_strided",
        FOUR_STEP,
        io_mode="four_step_row_strided",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_real_row",
        FOUR_STEP,
        io_mode="four_step_real_row",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_hermitian_row",
        FOUR_STEP,
        io_mode="four_step_hermitian_row",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_col",
        FOUR_STEP,
        io_mode="four_step_col",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_col_strided",
        FOUR_STEP,
        io_mode="four_step_col_strided",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_r2c_col",
        FOUR_STEP,
        io_mode="four_step_r2c_col",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "four_step_c2r_col",
        FOUR_STEP,
        io_mode="four_step_c2r_col",
        requires=_BASE_LEAF_FLAGS + _FOUR_STEP_FLAGS,
    ),
    KernelSpec(
        "bluestein_prepare", BLUESTEIN, requires=_BLUESTEIN_FLAGS
    ),
    KernelSpec(
        "bluestein_pointwise", BLUESTEIN, requires=_BLUESTEIN_FLAGS
    ),
    KernelSpec(
        "bluestein_finalize", BLUESTEIN, requires=_BLUESTEIN_FLAGS
    ),
    KernelSpec("rader_prepare", RADER, requires=_RADER_FLAGS),
    KernelSpec("rader_pointwise", RADER, requires=_RADER_FLAGS),
    KernelSpec("rader_finalize", RADER, requires=_RADER_FLAGS),
    KernelSpec("reshape_pack", RESHAPE, requires=_RESHAPE_FLAGS),
    KernelSpec("twiddle_reshape_pack", RESHAPE, requires=_RESHAPE_FLAGS),
    KernelSpec("tiled_transpose", TRANSPOSE, requires=_RESHAPE_FLAGS),
    KernelSpec("transpose3d", TRANSPOSE3D, requires=_TRANSPOSE3D_FLAGS),
    KernelSpec("real_to_complex", REAL_POINTWISE, requires=("length",)),
    KernelSpec("r2c_half_pack", REAL_POINTWISE, requires=("length",)),
    KernelSpec(
        "compact_to_hermitian_full", REAL_POINTWISE, requires=("length",)
    ),
    KernelSpec("complex_to_real", REAL_POINTWISE, requires=("length",)),
)

KERNEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)
KERNEL_SPECS: dict[str, KernelSpec] = {spec.name: spec for spec in _SPECS}

# Four-step row/column kernels that must validate ``plan.length`` against
# ``four_step_n1`` / ``four_step_n2`` before emitting.
FOUR_STEP_ROW_NAMES = frozenset(
    {
        "four_step_row",
        "four_step_row_strided",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
)
FOUR_STEP_COL_NAMES = frozenset(
    {
        "four_step_col",
        "four_step_col_strided",
        "four_step_r2c_col",
        "four_step_c2r_col",
    }
)

# Subsets used by metadata generation. These intentionally mirror the
# historical ``kernel_type`` sets in ``jit_source._metadata``.
CONTIGUOUS_BATCH_PACK_KERNELS = frozenset(
    {
        "leaf",
        "leaf_strided",
        "leaf_r2c",
        "leaf_c2r",
        "leaf_bluestein",
    }
)
INNER_PACK_ROW_KERNELS = frozenset(
    {
        "four_step_row",
        "four_step_real_row",
        "four_step_hermitian_row",
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_pointwise_row",
        "four_step_row_strided",
    }
)
INNER_PACK_COL_KERNELS = frozenset(
    {
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
        "bluestein_four_step_finish_col",
        "four_step_col_strided",
    }
)
STRIDED_FOUR_STEP_KERNELS = frozenset(
    {"four_step_row_strided", "four_step_col_strided"}
)


def kernel_spec(name: str) -> KernelSpec:
    try:
        return KERNEL_SPECS[name]
    except KeyError:
        raise ValueError(f"unsupported JIT kernel kind: {name}") from None


def module_name_for(
    spec: KernelSpec,
    *,
    direction_tag: str,
    factor_tag: str,
    lanes: int,
    lane_block: int,
    dtype_tag: str,
    length: int,
    prime_n: int,
    four_step_n1: int,
    four_step_n2: int,
) -> str:
    """Build the on-disk generated module name for a kernel spec."""
    if spec.family == BLUESTEIN_LEAF:
        return (
            f"flagfft_jit_{spec.name}_{direction_tag}_{factor_tag}"
            f"_n{prime_n}_m{length}_l{lanes}_b{lane_block}_{dtype_tag}"
        )
    if spec.family == BLUESTEIN_FOUR_STEP:
        return (
            f"flagfft_jit_{spec.name}_{direction_tag}_{factor_tag}_p{prime_n}"
            f"_n{four_step_n1}_{four_step_n2}_l{lanes}_b{lane_block}_{dtype_tag}"
        )
    if spec.name == "direct_dft":
        return f"flagfft_jit_direct_dft_{direction_tag}_n{length}_{dtype_tag}"
    if spec.name == "direct_dft_strided":
        return (
            f"flagfft_jit_direct_dft_strided_{direction_tag}_n{length}_{dtype_tag}"
        )
    if spec.name == "leaf":
        return (
            f"flagfft_jit_{direction_tag}_{factor_tag}"
            f"_l{lanes}_b{lane_block}_{dtype_tag}"
        )
    if spec.name in {"leaf_r2c", "leaf_c2r"}:
        return (
            f"flagfft_jit_{spec.name}_{direction_tag}_{factor_tag}"
            f"_l{lanes}_b{lane_block}_{dtype_tag}"
        )
    return (
        f"flagfft_jit_{spec.name}_{direction_tag}_{factor_tag}"
        f"_n{four_step_n1}_{four_step_n2}_l{lanes}_b{lane_block}_{dtype_tag}"
    )


def family_of(name: str) -> str:
    return kernel_spec(name).family


def is_four_step_twiddle_eligible(name: str) -> bool:
    """Mirror the historical tle-fused-twiddle eligibility check."""
    return (
        name.startswith("four_step_") or name.startswith("bluestein_four_step_")
    ) and not name.endswith("_strided")


__all__ = [
    "BLUESTEIN",
    "BLUESTEIN_FAMILIES",
    "BLUESTEIN_FOUR_STEP",
    "BLUESTEIN_LEAF",
    "CONTIGUOUS_BATCH_PACK_KERNELS",
    "CT_LEAF",
    "DIRECT_DFT",
    "FOUR_STEP",
    "FOUR_STEP_COL_NAMES",
    "FOUR_STEP_FAMILIES",
    "FOUR_STEP_ROW_NAMES",
    "INNER_PACK_COL_KERNELS",
    "INNER_PACK_ROW_KERNELS",
    "KERNEL_NAMES",
    "KERNEL_SPECS",
    "KernelSpec",
    "LEAF_LIKE_FAMILIES",
    "RADER",
    "REAL_POINTWISE",
    "RESHAPE",
    "STRIDED_FOUR_STEP_KERNELS",
    "TRANSPOSE",
    "TRANSPOSE3D",
    "family_of",
    "is_four_step_twiddle_eligible",
    "kernel_spec",
    "module_name_for",
]
