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
import importlib.util
import json
import os
import sys
from pathlib import Path

from .emit import (
    _emit_bluestein_jit_kernel,
    _emit_r2c_pointwise_jit_kernel,
    _emit_rader_jit_kernel,
    _emit_reshape_jit_kernel,
    _emit_tiled_transpose3d_jit_kernel,
    _emit_tiled_transpose_jit_kernel,
    emit_jit_kernel,
)
from .metadata import _csv_ints
from .kernels_small_3d import (
    emit_fused_32_column_kernel,
    emit_fused_16_cube_kernel,
    emit_fused_16_plane_kernel,
    emit_fused_32_real_plane_kernel,
    emit_fused_rect_plane_kernel,
    emit_fused_16_real_plane_kernel,
    emit_fused_32_plane_kernel,
)
from .kernels_small_2d import emit_fused_2d_kernel
from .artifacts import write_text_atomic
from .registry import (
    BLUESTEIN,
    BLUESTEIN_FOUR_STEP,
    BLUESTEIN_LEAF,
    DIRECT_DFT,
    KERNEL_NAMES,
    RADER,
    REAL_POINTWISE,
    RESHAPE,
    SMALL_3D,
    SMALL_2D,
    STOCKHAM,
    TRANSPOSE,
    TRANSPOSE3D,
    kernel_spec,
)
from .target import (
    set_codegen_target,
    set_maca_1d_single_default,
    set_maca_1d_batch_default,
    set_maca_2d_single_default,
    set_maca_3d_default,
    set_maca_3d_c2c32_single_cube,
)
from .maca_tail_policy import set_maca_tail_mode, variant_suffix


def _toolchain_version() -> str:
    """Toolchain identity for the cache fingerprint, without importing triton.

    The standalone codegen process cannot always import triton: on Ascend the
    FlagTree plugin expects torch_npu to be initialised first, and the import
    aborts code generation. Distribution metadata carries the same identity
    without executing the package.
    """
    from importlib import metadata

    for distribution in ("flagtree", "triton"):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return "unspecified"


def _maca_environment_fingerprint(environment: dict[str, str] | None = None) -> str:
    """Separate generated MACA artifacts for explicit resource-policy overrides."""
    import hashlib

    values = os.environ if environment is None else environment
    overrides = {key: value for key, value in values.items() if key.startswith("FLAGFFT_MACA_")}
    if not overrides:
        return ""
    payload = json.dumps(overrides, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    import hashlib
    from dataclasses import asdict, replace

    from .backend_profile import BackendProfile, current_profile, set_profile

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
    parser.add_argument(
        "--perm-form",
        choices=(
            "outer",
            "inner",
            "outer_first",
            "inner_middle",
            "inner_middle_c2r_cube",
            "outer_last",
        ),
        default="outer",
        help="axis placement for the permuted store's fused permutation",
    )
    parser.add_argument("--hcu-3d-full-smem", action="store_true")
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
    parser.add_argument("--fused-plane-n0", type=int, default=0)
    parser.add_argument("--fused-plane-n1", type=int, default=0)
    parser.add_argument("--fused-plane-middle", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--fused-2d-transpose", action="store_true")
    parser.add_argument(
        "--npu-portable-leaf",
        action="store_true",
        help="use portable exchange for scoped Ascend leaf kernels",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target", default="", help="Triton backend:architecture:warp_size"
    )
    parser.add_argument(
        "--maca-1d-single",
        action="store_true",
        help="enable the measured MACA rank-1 batch-1 code-generation defaults",
    )
    parser.add_argument("--maca-1d-batch", action="store_true",
                        help="enable measured MACA rank-1 batch-64 code-generation defaults")
    parser.add_argument("--ix-ct-single", action="store_true",
                        help="enable the scoped IX FP32 portable single policy")
    parser.add_argument("--ix-ct-batch", action="store_true",
                        help="enable the scoped IX FP32 batch-64 CT leaf policy")
    parser.add_argument("--ix-real-single-pack", action="store_true",
                        help="use one transform per block for IX N=210 real single")
    parser.add_argument("--ix-ct-single-tle", type=int, choices=(0, 1, 2), default=0,
                        help="IX TLE single preset: 0=off, 1=mixed, 2=1048576")
    parser.add_argument("--maca-tail-mode",
                        choices=("off", "p4w4", "real-direct", "batch-vecio",
                                 "batch-real-pack2", "batch-c2c-pack2", "batch-prime-vecio",
                                 "batch-c2c-tree", "batch-prime-rader-vecio"),
                        default="off")
    parser.add_argument(
        "--maca-2d-single",
        action="store_true",
        help="enable the MACA rank-2 FP32 batch-1 code-generation policy",
    )
    parser.add_argument("--maca-3d", action="store_true",
                        help="enable scoped MACA rank-3 axis packing")
    parser.add_argument("--maca-3d-c2c32-single-cube", action="store_true",
                        help="enable the MACA single-cube FP64 C2C 32^3 packing policy")
    parser.add_argument(
        "--compile-script",
        type=Path,
        help="Compile in this process using libtriton_jit's standalone helper",
    )
    parser.add_argument(
        "--device-profile", help="JSON device capabilities from the adaptor"
    )
    parser.add_argument(
        "--execution-policy", choices=("legacy", "native", "packed", "balanced")
    )
    args = parser.parse_args()
    if args.npu_portable_leaf:
        os.environ["FLAGFFT_NPU_FOURSTEP_LEAF"] = "1"
    set_codegen_target(args.target)
    set_maca_1d_single_default(args.maca_1d_single)
    set_maca_1d_batch_default(args.maca_1d_batch)
    from .target import (set_ix_ct_single_default, set_ix_ct_batch_default,
                         set_ix_ct_single_tle_default, set_ix_real_single_pack)
    set_ix_ct_single_default(args.ix_ct_single)
    set_ix_ct_batch_default(args.ix_ct_batch)
    set_ix_real_single_pack(args.ix_real_single_pack)
    set_ix_ct_single_tle_default(args.ix_ct_single_tle)
    set_maca_tail_mode(args.maca_tail_mode)
    set_maca_2d_single_default(args.maca_2d_single)
    set_maca_3d_default(args.maca_3d)
    set_maca_3d_c2c32_single_cube(args.maca_3d_c2c32_single_cube)
    if args.device_profile:
        device = json.loads(args.device_profile)
        default_policies = {"ix": "balanced", "hcu": "native"}
        policy = args.execution_policy or default_policies.get(
            device.get("backend"), "legacy"
        )
        set_profile(BackendProfile.from_device(device, policy))
    source_hash = hashlib.sha256()
    for source_path in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(
            source_path.relative_to(Path(__file__).parent).as_posix().encode()
        )
        source_hash.update(source_path.read_bytes())
    profile = replace(
        current_profile(),
        toolchain=_toolchain_version(),
        source_fingerprint=source_hash.hexdigest(),
    )
    set_profile(profile)
    profile_dir = profile.fingerprint
    if profile.backend == "ix":
        overrides = {k: v for k, v in os.environ.items() if k.startswith("FLAGFFT_IX_")}
        profile_dir += "-" + hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:12]
        profile_dir += "-ct-single" if args.ix_ct_single else "-ct-single-off"
        if args.ix_ct_batch:
            profile_dir += "-ct-batch"
        if args.ix_real_single_pack:
            profile_dir += "-real-single-pack"
        if args.ix_ct_single_tle:
            profile_dir += f"-tle{args.ix_ct_single_tle}"
    if profile.backend == "maca":
        # Keep default-on and explicit opt-out artifacts separate.  The native
        # compiler also includes this bit in its in-process cache key, but the
        # filesystem cache must not let one policy overwrite the other.
        profile_dir += (
            "-maca-1d-single" if args.maca_1d_single else "-maca-1d-single-off"
        )
        profile_dir += (
            "-maca-1d-batch" if args.maca_1d_batch else "-maca-1d-batch-off"
        )
        profile_dir += (
            "-maca-2d-single" if args.maca_2d_single else "-maca-2d-single-off"
        )
        profile_dir += "-maca-3d" if args.maca_3d else "-maca-3d-off"
        maca_env_fingerprint = _maca_environment_fingerprint()
        if maca_env_fingerprint:
            profile_dir += f"-maca-env-{maca_env_fingerprint}"
    if profile.backend == "musa":
        pair_store = os.getenv("FLAGFFT_MUSA_3D_PAIR_STORE", "1")
        permuted_pack = os.getenv("FLAGFFT_MUSA_3D_PACK", "auto")
        if pair_store not in {"0", "1"}:
            parser.error("FLAGFFT_MUSA_3D_PAIR_STORE must be 0 or 1")
        if permuted_pack not in {"auto", "1", "2", "4", "8"}:
            parser.error("FLAGFFT_MUSA_3D_PACK must be auto, 1, 2, 4 or 8")
        profile_dir += f"-musa-3d-pair-store-{pair_store}-pack-{permuted_pack}"
    if profile.backend == "hcu":
        pair_store = os.getenv("FLAGFFT_HCU_3D_PAIR_STORE", "1")
        permuted_pack = os.getenv("FLAGFFT_HCU_3D_PACK", "auto")
        fp64_tile = os.getenv("FLAGFFT_HCU_3D_FP64_TILE", "auto")
        fused_warps = os.getenv("FLAGFFT_HCU_3D_FUSED_WARPS", "auto")
        transpose_pair = os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_PAIR", "0")
        transpose_tile = os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_TILE", "auto")
        transpose_slice_group = os.getenv(
            "FLAGFFT_HCU_3D_TRANSPOSE_SLICE_GROUP", "auto"
        )
        full_smem = os.getenv("FLAGFFT_HCU_3D_FULL_SMEM", "0")
        u64_load = os.getenv("FLAGFFT_HCU_3D_U64_LOAD", "1")
        first_pack = os.getenv("FLAGFFT_HCU_3D_FIRST_PACK", "auto")
        middle_pack = os.getenv("FLAGFFT_HCU_3D_MIDDLE_PACK", "auto")
        middle_batch_pack = os.getenv("FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK", "auto")
        final_warps = os.getenv("FLAGFFT_HCU_3D_FINAL_WARPS", "auto")
        final_pack = os.getenv("FLAGFFT_HCU_3D_FINAL_PACK", "auto")
        smem_swizzle = os.getenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE", "auto")
        swizzle_shift = os.getenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE_SHIFT", "5")
        r2c_leaf_swizzle = os.getenv("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE", "auto")
        factors_256 = os.getenv("FLAGFFT_HCU_3D_256_FACTORS", "auto")
        factors_2048 = os.getenv("FLAGFFT_HCU_3D_2048_FACTORS", "auto")
        r2c_fp64_factors_2048 = os.getenv("FLAGFFT_HCU_3D_R2C_FP64_2048_FACTORS", "auto")
        if pair_store not in {"0", "1"}:
            parser.error("FLAGFFT_HCU_3D_PAIR_STORE must be 0 or 1")
        if permuted_pack not in {"auto", "1", "2", "4", "8", "16", "32"}:
            parser.error("FLAGFFT_HCU_3D_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        if fp64_tile not in {"auto", "0", "1"}:
            parser.error("FLAGFFT_HCU_3D_FP64_TILE must be auto, 0 or 1")
        if fused_warps not in {"auto", "1", "2", "4", "8"}:
            parser.error("FLAGFFT_HCU_3D_FUSED_WARPS must be 1, 2, 4 or 8")
        if transpose_pair not in {"0", "1"}:
            parser.error("FLAGFFT_HCU_3D_TRANSPOSE_PAIR must be 0 or 1")
        if transpose_tile not in {"auto", "16", "32", "64"}:
            parser.error("FLAGFFT_HCU_3D_TRANSPOSE_TILE must be auto, 16, 32 or 64")
        if transpose_slice_group not in {"auto", "1", "2"}:
            parser.error(
                "FLAGFFT_HCU_3D_TRANSPOSE_SLICE_GROUP must be auto, 1 or 2"
            )
        if full_smem not in {"0", "1"}:
            parser.error("FLAGFFT_HCU_3D_FULL_SMEM must be 0 or 1")
        if u64_load not in {"0", "1"}:
            parser.error("FLAGFFT_HCU_3D_U64_LOAD must be 0 or 1")
        valid_packs = {"auto", "1", "2", "4", "8", "16", "32"}
        if first_pack not in valid_packs:
            parser.error("FLAGFFT_HCU_3D_FIRST_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        if middle_pack not in valid_packs:
            parser.error("FLAGFFT_HCU_3D_MIDDLE_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        if middle_batch_pack not in {"auto", "1", "2", "4", "8", "16", "32"}:
            parser.error("FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        if final_warps not in {"auto", "1", "2", "4", "8"}:
            parser.error("FLAGFFT_HCU_3D_FINAL_WARPS must be auto, 1, 2, 4 or 8")
        if final_pack not in {"auto", "1", "2", "4", "8", "16", "32"}:
            parser.error("FLAGFFT_HCU_3D_FINAL_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        if smem_swizzle not in {"auto", "0", "1"}:
            parser.error("FLAGFFT_HCU_3D_SMEM_SWIZZLE must be auto, 0 or 1")
        if swizzle_shift not in {str(x) for x in range(1, 9)}:
            parser.error("FLAGFFT_HCU_3D_SMEM_SWIZZLE_SHIFT must be in [1, 8]")
        if r2c_leaf_swizzle not in {"auto", "0", "1"}:
            parser.error("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE must be auto, 0 or 1")
        valid_factors_256 = {
            "auto",
            "16,16",
            "8,8,4",
            "4,8,8",
            "8,4,8",
            "4,16,4",
            "4,4,4,4",
        }
        if factors_256 not in valid_factors_256:
            parser.error(
                "FLAGFFT_HCU_3D_256_FACTORS must be auto, 16,16, "
                "8,8,4, 4,8,8, 8,4,8, 4,16,4 or 4,4,4,4"
            )
        if factors_2048 not in {"auto", "16,16,8", "8,16,16", "16,8,16"}:
            parser.error(
                "FLAGFFT_HCU_3D_2048_FACTORS must be auto, 16,16,8, 8,16,16 or 16,8,16"
            )
        if r2c_fp64_factors_2048 not in {"auto", "16,16,8", "8,16,16", "16,8,16"}:
            parser.error(
                "FLAGFFT_HCU_3D_R2C_FP64_2048_FACTORS must be auto, 16,16,8, "
                "8,16,16 or 16,8,16"
            )
        profile_dir += (f"-hcu-3d-pair-store-{pair_store}-pack-{permuted_pack}"
                        f"-fp64-tile-{fp64_tile}-warps-{fused_warps}"
                        f"-transpose-pair-{transpose_pair}-transpose-tile-{transpose_tile}"
                        f"-full-{full_smem}-key-{int(args.hcu_3d_full_smem)}"
                        f"-u64-load-{u64_load}-first-pack-{first_pack}-middle-pack-{middle_pack}"
                        f"-mbp-{middle_batch_pack}"
                        f"-final-warps-{final_warps}-final-pack-{final_pack}"
                        f"-swz-{smem_swizzle}-{swizzle_shift}"
                        f"-r2c-leaf-swizzle-{r2c_leaf_swizzle}")
        if transpose_slice_group != "auto":
            profile_dir += f"-sg{transpose_slice_group}"
        if factors_256 != "auto":
            profile_dir += f"-f{factors_256.replace(',', '')}"
    args.out_dir = args.out_dir / profile_dir
    # Legacy tree and explicit resource overrides must not overwrite a module
    # emitted earlier by the same executable, even when tail mode is off.
    if variant_suffix(root=True):
        args.out_dir = args.out_dir / variant_suffix(root=True)

    spec = kernel_spec(args.kernel)
    missing = [flag for flag in spec.requires if getattr(args, flag) is None]
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
    elif spec.family == SMALL_2D:
        metadata = emit_fused_2d_kernel(
            n=args.length,
            dtype=args.dtype,
            direction=args.direction,
            transpose_output=args.fused_2d_transpose,
            out_dir=args.out_dir,
        )
    elif spec.family == SMALL_3D:
        if args.kernel == "fused_32_column":
            metadata = emit_fused_32_column_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
            )
        elif args.kernel == "fused_16_cube":
            metadata = emit_fused_16_cube_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
            )
        elif args.kernel == "fused_32_real_plane":
            metadata = emit_fused_32_real_plane_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
            )
        elif args.kernel == "fused_32_plane":
            metadata = emit_fused_32_plane_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
            )
        elif args.kernel == "fused_16_real_plane":
            metadata = emit_fused_16_real_plane_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
            )
        elif args.kernel == "fused_rect_plane":
            metadata = emit_fused_rect_plane_kernel(
                dtype=args.dtype,
                direction=args.direction,
                out_dir=args.out_dir,
                plane_n0=args.fused_plane_n0,
                plane_n1=args.fused_plane_n1,
                middle_size=args.fused_plane_middle,
            )
        else:
            metadata = emit_fused_16_plane_kernel(
                dtype=args.dtype, direction=args.direction, out_dir=args.out_dir,
                plane_size=args.length or 16,
            )
    elif spec.family == REAL_POINTWISE:
        if args.length is None or args.length <= 0:
            parser.error(f"--kernel {args.kernel} requires --length")
        metadata = _emit_r2c_pointwise_jit_kernel(
            kernel=args.kernel,
            n=args.length,
            dtype=args.dtype,
            out_dir=args.out_dir,
            target=args.target,
        )
    elif spec.family in {DIRECT_DFT, STOCKHAM}:
        if args.length is None or args.length <= 0:
            parser.error("--kernel direct_dft requires --length")
        metadata = emit_jit_kernel(
            kernel=args.kernel,
            length=args.length,
            factors=args.factors if spec.family == STOCKHAM else (),
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
        if spec.is_four_step and (args.four_step_n1 <= 0 or args.four_step_n2 <= 0):
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
            prime_n=(args.rader_n if args.kernel in {"leaf_rader_full",
                                                    "leaf_rader_prepare",
                                                    "leaf_rader_finish"}
                     else args.bluestein_n) or 0,
            four_step_n1=args.four_step_n1,
            four_step_n2=args.four_step_n2,
            perm_form=args.perm_form,
            hcu_full_smem=args.hcu_3d_full_smem,
            out_dir=args.out_dir,
        )
    else:
        raise AssertionError(f"unreachable kernel spec: {args.kernel}")

    if args.compile_script:
        if not args.target.startswith("maca:"):
            parser.error("--compile-script currently requires a MACA target")
        os.environ["TRITON_JIT_BACKEND"] = "MACA"
        compile_spec = importlib.util.spec_from_file_location(
            "standalone_compile", args.compile_script
        )
        if compile_spec is None or compile_spec.loader is None:
            raise RuntimeError(f"Cannot load compilation helper: {args.compile_script}")
        compiler = importlib.util.module_from_spec(compile_spec)
        sys.modules[compile_spec.name] = compiler
        compile_spec.loader.exec_module(compiler)
        from triton.backends.compiler import GPUTarget

        backend, arch, warp = args.target.split(":")
        metadata["binary_dir"] = compiler.compile_a_kernel(
            metadata["module_path"],
            metadata["kernel_name"],
            metadata["signature"],
            metadata["num_warps"],
            metadata["num_stages"],
            0,
            {},
            compile_target=GPUTarget(backend, int(arch), int(warp)),
        )
        kernel_metadata = Path(metadata["binary_dir"]) / (
            metadata["kernel_name"] + ".json"
        )
        compiled = json.loads(kernel_metadata.read_text())
        if compiled.get("global_scratch_size", 0) or compiled.get(
            "profile_scratch_size", 0
        ):
            raise RuntimeError(
                "MACA kernel requires scratch allocation unsupported by raw launch"
            )
    profile.validate(metadata["num_warps"])
    metadata.update(
        {
            "hardware_profile": asdict(profile),
            "profile_id": profile.fingerprint,
            "maca_1d_single_default": args.maca_1d_single,
            "ix_ct_single_default": args.ix_ct_single,
            "ix_ct_batch_default": args.ix_ct_batch,
            "ix_real_single_pack": args.ix_real_single_pack,
            "ix_ct_single_tle_default": args.ix_ct_single_tle,
            "maca_tail_mode": args.maca_tail_mode,
            "maca_2d_single_default": args.maca_2d_single,
            "warp_size": profile.warp_size,
            "block_threads": metadata["num_warps"] * profile.warp_size,
        }
    )
    write_text_atomic(
        Path(metadata["module_path"]).with_suffix(".json"),
        json.dumps(metadata, sort_keys=True)
    )
    print(json.dumps(metadata, sort_keys=True))


__all__ = [
    "main",
]
