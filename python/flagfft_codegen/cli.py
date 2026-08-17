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

from __future__ import annotations

"""Command-line entry point for ``python -m flagfft_codegen.jit_source``."""

import argparse
import json
from pathlib import Path

from .emit import (
    _emit_bluestein_jit_kernel,
    _emit_rader_jit_kernel,
    _emit_reshape_jit_kernel,
    _emit_r2c_pointwise_jit_kernel,
    _emit_tiled_transpose3d_jit_kernel,
    _emit_tiled_transpose_jit_kernel,
    emit_jit_kernel,
)
from .metadata import _csv_ints
from .registry import (
    BLUESTEIN,
    BLUESTEIN_FOUR_STEP,
    BLUESTEIN_LEAF,
    DIRECT_DFT,
    KERNEL_NAMES,
    RADER,
    REAL_POINTWISE,
    RESHAPE,
    TRANSPOSE,
    TRANSPOSE3D,
    kernel_spec,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FlagFFT libtriton_jit kernel sources"
    )
    parser.add_argument(
        "--kernel",
        choices=KERNEL_NAMES,
        required=True,
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--factors", type=_csv_ints)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--generic-radices", type=_csv_ints, default=())
    parser.add_argument("--smem-size", type=int)
    parser.add_argument(
        "--direction", choices=("forward", "inverse"), default="forward"
    )
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--four-step-n1", type=int, default=0)
    parser.add_argument("--four-step-n2", type=int, default=0)
    parser.add_argument("--bluestein-n", type=int)
    parser.add_argument("--bluestein-m", type=int)
    parser.add_argument("--rader-n", type=int)
    parser.add_argument("--rader-m", type=int)
    parser.add_argument("--reshape-n1", type=int, default=0)
    parser.add_argument("--reshape-n2", type=int, default=0)
    parser.add_argument("--transpose3d-n0", type=int, default=0)
    parser.add_argument("--transpose3d-n1", type=int, default=0)
    parser.add_argument("--transpose3d-n2", type=int, default=0)
    parser.add_argument("--transpose3d-order", choices=("021", "210", "201", "120"))
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = kernel_spec(args.kernel)
    missing = [
        flag
        for flag in spec.requires
        if getattr(args, flag) is None
    ]
    if missing:
        parser.error(
            f"--kernel {args.kernel} requires "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
        )

    if spec.family == BLUESTEIN:
        metadata = _emit_bluestein_jit_kernel(
            kernel=args.kernel,
            n=args.bluestein_n,
            m=args.bluestein_m,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
    elif spec.family == RADER:
        metadata = _emit_rader_jit_kernel(
            kernel=args.kernel,
            n=args.rader_n,
            m=args.rader_m,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
    elif spec.family == RESHAPE:
        if args.reshape_n1 <= 0 or args.reshape_n2 <= 0:
            parser.error(
                f"--kernel {args.kernel} requires --reshape-n1 and --reshape-n2"
            )
        metadata = _emit_reshape_jit_kernel(
            kernel=args.kernel,
            n1=args.reshape_n1,
            n2=args.reshape_n2,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
    elif spec.family == TRANSPOSE:
        if args.reshape_n1 <= 0 or args.reshape_n2 <= 0:
            parser.error(
                "--kernel tiled_transpose requires --reshape-n1 and --reshape-n2"
            )
        metadata = _emit_tiled_transpose_jit_kernel(
            n0=args.reshape_n1,
            n1=args.reshape_n2,
            dtype=args.dtype,
            tile_size=args.tile_size,
            out_dir=args.out_dir,
        )
    elif spec.family == TRANSPOSE3D:
        if (
            args.transpose3d_n0 <= 0
            or args.transpose3d_n1 <= 0
            or args.transpose3d_n2 <= 0
            or args.transpose3d_order is None
        ):
            parser.error(
                "--kernel transpose3d requires --transpose3d-n0, --transpose3d-n1, "
                "--transpose3d-n2 and --transpose3d-order"
            )
        metadata = _emit_tiled_transpose3d_jit_kernel(
            n0=args.transpose3d_n0,
            n1=args.transpose3d_n1,
            n2=args.transpose3d_n2,
            order=args.transpose3d_order,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
    elif spec.family == REAL_POINTWISE:
        if args.length is None or args.length <= 0:
            parser.error(f"--kernel {args.kernel} requires --length")
        metadata = _emit_r2c_pointwise_jit_kernel(
            kernel=args.kernel,
            n=args.length,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
    elif spec.family == DIRECT_DFT:
        if args.length is None or args.length <= 0:
            parser.error("--kernel direct_dft requires --length")
        metadata = emit_jit_kernel(
            kernel=args.kernel,
            length=args.length,
            factors=(),
            lanes=1,
            num_warps=1,
            generic_radices=(),
            smem_size=0,
            direction=args.direction,
            dtype=args.dtype,
            prime_n=0,
            four_step_n1=0,
            four_step_n2=0,
            out_dir=args.out_dir,
        )
    elif spec.is_leaf_like:
        if spec.family in {BLUESTEIN_LEAF, BLUESTEIN_FOUR_STEP} and (
            args.bluestein_n is None or args.bluestein_n <= 0
        ):
            parser.error(f"--kernel {args.kernel} requires --bluestein-n")
        if spec.is_four_step and (
            args.four_step_n1 <= 0 or args.four_step_n2 <= 0
        ):
            parser.error(
                f"--kernel {args.kernel} requires --four-step-n1 and --four-step-n2"
            )
        metadata = emit_jit_kernel(
            kernel=args.kernel,
            length=args.length,
            factors=args.factors,
            lanes=args.lanes,
            num_warps=args.num_warps,
            generic_radices=args.generic_radices,
            smem_size=args.smem_size,
            direction=args.direction,
            dtype=args.dtype,
            prime_n=args.bluestein_n or 0,
            four_step_n1=args.four_step_n1,
            four_step_n2=args.four_step_n2,
            out_dir=args.out_dir,
        )
    else:
        raise AssertionError(f"unreachable kernel spec: {args.kernel}")

    print(json.dumps(metadata, sort_keys=True))


__all__ = [
    "main",
]
