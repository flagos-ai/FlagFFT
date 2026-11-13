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
    pack = 4 if plan.length > 1024 else 8
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
