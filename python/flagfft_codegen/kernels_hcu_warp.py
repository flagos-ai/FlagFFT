# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HCU wave-local complex FFT stores, retaining the native leaf ABI."""

import os

from .backend_profile import current_profile


def hcu_warp_store_pack(plan, io_mode, perm_form, span, batch):
    profile = current_profile()
    if (profile.backend != "hcu" or profile.warp_size != 64
            or os.getenv("FLAGFFT_HCU_3D_WARP_FFT", "0") != "1"
            or plan.dtype != "complex64"
            or not 64 <= plan.length <= 2048 or plan.length & (plan.length - 1)
            or io_mode != "permuted_store" or perm_form.endswith("_batched")
            or span <= 0 or batch <= 0):
        return 0
    inner = perm_form.startswith("inner")
    target = os.getenv("FLAGFFT_HCU_3D_WARP_PACK", "auto")
    if target not in {"auto", "1", "2", "4", "8", "16", "32"}:
        raise ValueError("FLAGFFT_HCU_3D_WARP_PACK must be auto, 1, 2, 4, 8, 16 or 32")
    pack = 32 if target == "auto" else int(target)
    # Limit the live input/output payload to 64 FP32 registers per thread.
    # Logical FFT rows can share a wave; the physical block stays <= 512 lanes.
    while pack > 1 and plan.length * pack * 2 > min(pack, 8) * 64 * 64:
        pack //= 2
    while pack > 1 and (batch % span or (batch // span) % pack
                       if inner else span % pack):
        pack //= 2
    return pack


def build_hcu_warp_store(plan, perm_form, params, pack, span, batch):
    # The wave-local radix decomposition is independent of the IX launcher.
    # Use ordinary Triton math and keep HCU's existing argument signature.
    from .kernels_ix_warp import build_ix_warp_source

    mode = "inner" if perm_form.startswith("inner") else "outer"
    radix = plan.length // 64
    chunk = min(radix, 8)
    shared_limit = min(current_profile().max_dynamic_shared_memory // 2, 32768)
    while chunk > 1 and pack * 64 * chunk * 8 > shared_limit:
        chunk //= 2
    name, source = build_ix_warp_source(
        plan.length, pack, mode, plan.direction == "inverse",
        output_radix_chunk=chunk if chunk < radix else 0,
    )
    leaf_name = "hcu_" + name + "_leaf"
    aliases = [
        "    inp=in_ptr", "    out=out_ptr",
        f"    span=tl.full((),{span},tl.int64)",
        f"    nbatch=tl.full((),{batch},tl.int64)",
        f"    input_distance=tl.full((),{plan.length},tl.int64)",
        f"    output_distance=tl.full((),{plan.length},tl.int64)",
    ]
    header = f"def {name}(inp,out,span,nbatch,input_distance,output_distance):"
    assert header in source
    source = source.replace(
        header, f"def {leaf_name}(" + ",".join(params) + "):\n" + "\n".join(aliases), 1,
    )
    return leaf_name, source
