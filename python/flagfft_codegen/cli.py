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
from .registry import (
    BLUESTEIN,
    BLUESTEIN_FOUR_STEP,
    BLUESTEIN_LEAF,
    DIRECT_DFT,
    KERNEL_NAMES,
    RADER,
    REAL_POINTWISE,
    RESHAPE,
    STOCKHAM,
    TRANSPOSE,
    TRANSPOSE3D,
    kernel_spec,
)
from .target import set_codegen_target


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
        choices=("outer", "inner"),
        default="outer",
        help="axis placement for the permuted store's fused permutation",
    )
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
    parser.add_argument(
        "--target", default="", help="Triton backend:architecture:warp_size"
    )
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
    set_codegen_target(args.target)
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
    args.out_dir = args.out_dir / profile.fingerprint

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
    elif spec.family == REAL_POINTWISE:
        if args.length is None or args.length <= 0:
            parser.error(f"--kernel {args.kernel} requires --length")
        metadata = _emit_r2c_pointwise_jit_kernel(
            kernel=args.kernel,
            n=args.length,
            dtype=args.dtype,
            out_dir=args.out_dir,
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
            prime_n=args.bluestein_n or 0,
            four_step_n1=args.four_step_n1,
            four_step_n2=args.four_step_n2,
            perm_form=args.perm_form,
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
            "warp_size": profile.warp_size,
            "block_threads": metadata["num_warps"] * profile.warp_size,
        }
    )
    Path(metadata["module_path"]).with_suffix(".json").write_text(
        json.dumps(metadata, sort_keys=True)
    )
    print(json.dumps(metadata, sort_keys=True))


__all__ = [
    "main",
]
