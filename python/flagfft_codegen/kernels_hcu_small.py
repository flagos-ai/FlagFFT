# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""FFT rows fitting in one HCU wave, with the existing dense leaf ABI."""

import os

from .backend_profile import current_profile


def small_wave_pack(plan, io_mode):
    profile = current_profile()
    if profile.backend != "hcu" or profile.warp_size != 64:
        return 0
    setting = os.getenv("FLAGFFT_HCU_SMALL_WAVE", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_SMALL_WAVE must be 0 or 1")
    if (setting != "1" or plan.dtype != "complex64" or
            not 16 <= plan.length <= profile.warp_size or plan.length & (plan.length - 1) or
            io_mode not in {"contiguous", "contiguous_r2c", "contiguous_c2r"}):
        return 0
    return profile.warp_size // plan.length


def build_small_wave(plan, io_mode, params, pack):
    from .kernels_ix_warp import build_ix_warp_source

    mode = {"contiguous_r2c": "contiguous_r2c", "contiguous_c2r": "c2r"}.get(io_mode, "contiguous")
    name, source = build_ix_warp_source(plan.length, pack, mode, plan.direction == "inverse",
                                       lane_width=plan.length)
    leaf_name = "hcu_small_wave_" + name
    aliases = ["    inp=in_ptr", "    out=out_ptr", "    span=1"]
    for distance in ("input_distance", "output_distance"):
        if distance not in params:
            aliases.append(f"    {distance}={plan.length}")
    header = f"def {name}(inp,out,span,nbatch,input_distance,output_distance):"
    if header not in source:
        raise ValueError("small-wave kernel ABI changed")
    source = source.replace(header, f"def {leaf_name}(" + ",".join(params) + "):\n" + "\n".join(aliases), 1)
    source = source.replace("pid=tl.program_id(0)", "pid=tl.program_id(0).to(tl.int64)", 1)
    return leaf_name, source
