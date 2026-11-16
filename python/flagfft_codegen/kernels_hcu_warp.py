# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HCU wave-local complex FFT stores, retaining the native leaf ABI."""

import os

from .backend_profile import current_profile


_FOURSTEP_ROWS = {"four_step_row", "four_step_real_row", "four_step_hermitian_row"}
_FOURSTEP_COLS = {"four_step_col", "four_step_r2c_col", "four_step_c2r_col"}


def _root_table_enabled():
    setting = os.getenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_WAVE_ROOT_TABLE must be 0 or 1")
    return setting == "1"


def _scalar_wave_budget(plan):
    setting = os.getenv("FLAGFFT_HCU_WAVE_SCALAR_BUDGET", "64")
    if setting not in {"64", "128"}:
        raise ValueError("FLAGFFT_HCU_WAVE_SCALAR_BUDGET must be 64 or 128")
    return 64 if plan.dtype == "complex128" else int(setting)


def hcu_dense_wave_lanes(plan, io_mode):
    setting = os.getenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "64")
    if setting not in {"64", "auto"}:
        raise ValueError("FLAGFFT_HCU_WIDE_WAVE_LANES must be 64 or auto")
    if (setting == "auto" and plan.dtype == "complex128"
            and io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}):
        # Spread a wide scalar payload over two waves before increasing the
        # rows per CTA. A single-wave 2048-point FP64 FFT spills on gfx936.
        return 128 if plan.length * 8 > 64 * 128 else 64
    return 64


def hcu_dense_wave_pack(plan, io_mode):
    profile = current_profile()
    if profile.backend != "hcu" or profile.warp_size != 64:
        return 0
    setting = os.getenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_DENSE_WAVE must be 0 or 1")
    wide_setting = os.getenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "0")
    if wide_setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_DENSE_WAVE_FP64 must be 0 or 1")
    wide = plan.dtype == "complex128" and len(plan.factors) >= 2 and wide_setting == "1"
    packed = io_mode in {"packed_r2c", "packed_c2r"}
    lanes = hcu_dense_wave_lanes(plan, io_mode)
    if wide and not packed and plan.length * 8 > lanes * 128:
        # Account for input/output redistribution and 64-bit scalar width.
        # Larger tiles exceeded 256 VGPRs and spilled on the queried target.
        return 0
    if packed and plan.length * 4 > profile.warp_size * 64:
        # Reconstruction keeps both partner FFT values and output registers
        # live. Bound that doubled payload before selecting the wave path.
        return 0
    if (setting != "1" or (plan.dtype != "complex64" and not wide) or
            not 64 <= plan.length <= 2048 or plan.length & (plan.length - 1) or
            io_mode not in {"contiguous", "contiguous_r2c", "contiguous_c2r", "packed_r2c", "packed_c2r"} | _FOURSTEP_ROWS | _FOURSTEP_COLS):
        return 0
    target = os.getenv("FLAGFFT_HCU_DENSE_WAVE_PACK", "auto")
    if target not in {"auto", "1", "2", "4", "8", "16"}:
        raise ValueError("FLAGFFT_HCU_DENSE_WAVE_PACK must be auto, 1, 2, 4, 8 or 16")
    pack = (4 if wide else 8) if target == "auto" else int(target)
    # The supported lengths bound each lane to 64 scalar payload values
    # (128 FP32 register slots in FP64). Output redistribution is separately
    # chunked to half the queried LDS budget; FP64 starts with fewer waves.
    live_scalars = 4 if packed else 2
    budget = _scalar_wave_budget(plan)
    while pack > 1 and plan.length * pack * live_scalars > min(pack * lanes // 64, 8) * profile.warp_size * budget:
        pack //= 2
    return pack


def build_hcu_dense_wave(plan, io_mode, params, pack, n1=0, n2=0):
    from .kernels_ix_warp import build_ix_warp_source

    row = io_mode in _FOURSTEP_ROWS
    fourstep = row or io_mode in _FOURSTEP_COLS
    other = n2 if row else n1
    mode = {"contiguous_r2c": "contiguous_r2c", "contiguous_c2r": "c2r"}.get(io_mode, io_mode)
    lanes = hcu_dense_wave_lanes(plan, io_mode)
    radix = plan.length // lanes
    chunk = radix
    shared_limit = current_profile().max_dynamic_shared_memory // 2
    wide = plan.dtype == "complex128"
    root_table = _root_table_enabled()
    complex_bytes = 16 if wide else 8
    while chunk > 1 and pack * lanes * chunk * complex_bytes > shared_limit:
        chunk //= 2
    name, source = build_ix_warp_source(
        plan.length, pack, mode, plan.direction == "inverse",
        output_radix_chunk=chunk if chunk < radix else 0,
        four_step_other=other if fourstep else 0,
        wide=wide,
        table_roots=root_table and not wide,
        lane_width=lanes,
    )
    leaf_name = "hcu_" + name + "_dense"
    aliases = ["    inp=in_ptr", "    out=out_ptr", "    span=1"]
    if wide or root_table:
        last = len(plan.factors) - 1
        aliases.extend([f"    wave_tw_r_ptr=tw{last}_r_ptr+{plan.length}",
                        f"    wave_tw_i_ptr=tw{last}_i_ptr+{plan.length}"])
    for distance in ("input_distance", "output_distance"):
        if distance not in params:
            aliases.append(f"    {distance}={plan.length}")
    header = f"def {name}(inp,out,span,nbatch,input_distance,output_distance):"
    if header not in source:
        raise ValueError("HCU dense wave FFT ABI changed")
    source = source.replace(header, f"def {leaf_name}(" + ",".join(params) + "):\n" + "\n".join(aliases), 1)
    source = source.replace("pid=tl.program_id(0)", "pid=tl.program_id(0).to(tl.int64)", 1)
    source = source.replace("tl.program_id(1)*", "tl.program_id(1).to(tl.int64)*")
    return leaf_name, source


def hcu_warp_store_pack(plan, io_mode, perm_form, span, batch):
    profile = current_profile()
    if (profile.backend != "hcu" or profile.warp_size != 64
            or os.getenv("FLAGFFT_HCU_3D_WARP_FFT", "0") != "1"
            or plan.dtype != "complex64"
            or not 64 <= plan.length <= 2048 or plan.length & (plan.length - 1)
            or io_mode != "permuted_store"
            or span <= 0 or batch <= 0):
        return 0
    inner = perm_form.startswith("inner")
    if inner and batch % span:
        return 0
    target = os.getenv("FLAGFFT_HCU_3D_WARP_PACK", "auto")
    if target not in {"auto", "1", "2", "4", "8", "16", "32"}:
        raise ValueError("FLAGFFT_HCU_3D_WARP_PACK must be auto, 1, 2, 4, 8, 16 or 32")
    pack = 32 if target == "auto" else int(target)
    # Limit the live payload per physical thread, with a bounded larger
    # experimental budget for wider coalesced output tiles.
    # Logical FFT rows can share a wave; the physical block stays <= 512 lanes.
    budget = _scalar_wave_budget(plan)
    while pack > 1 and plan.length * pack * 2 > min(pack, 8) * 64 * budget:
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
        table_roots=_root_table_enabled(),
    )
    batched = perm_form.endswith("_batched")
    leaf_name = "hcu_" + name + ("_batched" if batched else "") + "_leaf"
    if batched:
        if "rows_per_transform" not in params or mode != "inner":
            raise ValueError("batched HCU wave stores require the per-cube row count")
        source = source.replace(
            f"    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+slot",
            f"    cube=row//rows_per_transform\n"
            f"    local_row=row%rows_per_transform\n"
            f"    gbase=cube*({plan.length}*rows_per_transform)+(local_row%span)*(rows_per_transform//span)+local_row//span", 1,
        )
        source = source.replace("    stride=nbatch", "    stride=rows_per_transform")
        hint = f"    gbase=tl.max_contiguous(tl.multiple_of(gbase,{pack}),{pack})"
        source = source.replace(hint, f"    if (rows_per_transform//span)%{pack}==0:\n    " + hint)
    aliases = [
        "    inp=in_ptr", "    out=out_ptr",
        f"    span=tl.full((),{span},tl.int64)",
        f"    nbatch=tl.full((),{batch},tl.int64)",
        f"    input_distance=tl.full((),{plan.length},tl.int64)",
        f"    output_distance=tl.full((),{plan.length},tl.int64)",
    ]
    if _root_table_enabled():
        last = len(plan.factors) - 1
        aliases.extend([f"    wave_tw_r_ptr=tw{last}_r_ptr+{plan.length}",
                        f"    wave_tw_i_ptr=tw{last}_i_ptr+{plan.length}"])
    header = f"def {name}(inp,out,span,nbatch,input_distance,output_distance):"
    assert header in source
    source = source.replace(
        header, f"def {leaf_name}(" + ",".join(params) + "):\n" + "\n".join(aliases), 1,
    )
    source = source.replace("pid=tl.program_id(0)", "pid=tl.program_id(0).to(tl.int64)", 1)
    return leaf_name, source
