# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HCU cooperative complex FFT stores, retaining the native leaf ABI."""

import os

from .backend_profile import current_profile


_FOURSTEP_ROWS = {"four_step_row", "four_step_real_row", "four_step_hermitian_row"}
_FOURSTEP_COLS = {"four_step_col", "four_step_r2c_col", "four_step_c2r_col"}


def _root_table_enabled():
    setting = os.getenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_WAVE_ROOT_TABLE must be 0 or 1")
    return setting == "1"


def _native_wave_codelet():
    setting = os.getenv("FLAGFFT_HCU_WAVE_CODELET", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_WAVE_CODELET must be 0 or 1")
    return setting == "1"


def _scalar_wave_budget(plan):
    setting = os.getenv("FLAGFFT_HCU_WAVE_SCALAR_BUDGET", "64")
    if setting not in {"64", "128"}:
        raise ValueError("FLAGFFT_HCU_WAVE_SCALAR_BUDGET must be 64 or 128")
    return 64 if plan.dtype == "complex128" else int(setting)


def _wide_root_period(plan):
    setting = os.getenv("FLAGFFT_HCU_WAVE_ROOT_PERIOD", "0")
    if setting not in {"0", "4", "8"}:
        raise ValueError("FLAGFFT_HCU_WAVE_ROOT_PERIOD must be 0, 4 or 8")
    return int(setting) if plan.dtype == "complex128" else 0


def hcu_dense_wave_lanes(plan, io_mode):
    packed_layout = os.getenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "wave")
    if packed_layout not in {"wave", "compact"}:
        raise ValueError("FLAGFFT_HCU_PACKED_WAVE_LAYOUT must be wave or compact")
    if packed_layout == "compact" and io_mode in {"packed_r2c", "packed_c2r"}:
        # Bound the register butterfly by scalar width, then let independent
        # short FFTs share one queried physical wave instead of dedicating
        # 64 threads to every half-length boundary.
        register_radix = 16 if plan.dtype == "complex128" else 32
        return min(current_profile().warp_size,
                   max(current_profile().warp_size // 4, plan.length // register_radix))
    setting = os.getenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "64")
    if setting not in {"64", "auto"}:
        raise ValueError("FLAGFFT_HCU_WIDE_WAVE_LANES must be 64 or auto")
    if (setting == "auto" and plan.dtype == "complex128"
            and io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}):
        # Spread a wide scalar payload over two waves before increasing the
        # rows per CTA. A single-wave 2048-point FP64 FFT spills on gfx936.
        return 128 if plan.length * 8 > 64 * 128 else 64
    return 64


def hcu_dense_wave_pack(plan, io_mode, n1=0, n2=0):
    profile = current_profile()
    if profile.backend != "hcu" or profile.warp_size != 64 or len(plan.factors) < 2:
        return 0
    setting = os.getenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    if setting not in {"0", "1", "leaves", "packed", "fourstep_codelets"}:
        raise ValueError("FLAGFFT_HCU_DENSE_WAVE must be 0, 1, leaves, packed or fourstep_codelets")
    wide_setting = os.getenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "0")
    if wide_setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_DENSE_WAVE_FP64 must be 0 or 1")
    wide = plan.dtype == "complex128" and len(plan.factors) >= 2 and wide_setting == "1"
    packed = io_mode in {"packed_r2c", "packed_c2r"}
    if setting == "packed" and not packed:
        return 0
    lanes = hcu_dense_wave_lanes(plan, io_mode)
    fourstep = io_mode in _FOURSTEP_ROWS | _FOURSTEP_COLS
    if fourstep:
        # Wave rows apply the cross-axis twiddle at their output, while wave
        # columns consume it there. Replacing just one pass would either omit
        # or repeat that twiddle. Admit the entire pair before either emitter.
        axis_limit = 1024 if plan.dtype == "complex128" else 2048
        if not all(64 <= axis <= axis_limit and not axis & (axis - 1) for axis in (n1, n2)):
            return 0
    if setting == "fourstep_codelets" and (not fourstep or not _native_wave_codelet()
                                           or max(n1, n2) // 64 not in {16, 32}):
        return 0
    byte_budget = 256 if (_wide_root_period(plan) and lanes == 64 and
        io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}) else 128
    if wide and not packed and plan.length * 8 > lanes * byte_budget:
        # Account for input/output redistribution and 64-bit scalar width.
        # Larger tiles exceeded 256 VGPRs and spilled on the queried target.
        return 0
    if packed and plan.length * 4 > profile.warp_size * 64:
        # Reconstruction keeps both partner FFT values and output registers
        # live. Bound that doubled payload before selecting the wave path.
        return 0
    minimum = 32 if packed and os.getenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "wave") == "compact" else 64
    if (setting == "0" or (setting == "leaves" and io_mode in _FOURSTEP_ROWS | _FOURSTEP_COLS)
            or (plan.dtype != "complex64" and not wide) or
            not minimum <= plan.length <= 2048 or plan.length & (plan.length - 1) or
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
    if packed and os.getenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "wave") == "compact":
        # FFT values and reconstructed partner outputs coexist. Use an equal
        # byte budget in both precisions, leaving the output LDS bound intact.
        budget = 64 if plan.dtype == "complex128" else 128
    while pack > 1 and plan.length * pack * live_scalars > max(1, min(pack * lanes // 64, 8)) * profile.warp_size * budget:
        pack //= 2
    return pack


def hcu_packed_transposed_pack(plan):
    profile = current_profile()
    compact = os.getenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "wave") == "compact"
    minimum = 32 if compact else 64
    if (profile.backend != "hcu" or profile.warp_size != 64 or
            os.getenv("FLAGFFT_HCU_PACKED_REAL_STORE", "0") != "1" or
            len(plan.factors) < 2 or
            not _root_table_enabled() or plan.dtype not in {"complex64", "complex128"} or
            not minimum <= plan.length <= 1024 or plan.length & (plan.length - 1)):
        return 0
    pack = 8 if plan.dtype == "complex128" else 16
    lanes = hcu_dense_wave_lanes(plan, "packed_r2c")
    budget = (64 if plan.dtype == "complex128" else 128) if compact else (32 if plan.dtype == "complex128" else 64)
    while pack > 1 and plan.length * pack * 4 > max(1, min(pack * lanes // 64, 8)) * 64 * budget:
        pack //= 2
    return pack


def build_hcu_dense_wave(plan, io_mode, params, pack, n1=0, n2=0, *, packed_transpose=False):
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
        root_period=_wide_root_period(plan),
        local_codelet=_native_wave_codelet(),
        packed_output_transpose=packed_transpose,
    )
    leaf_name = "hcu_" + name + "_dense"
    aliases = ["    inp=in_ptr", "    out=out_ptr",
               "    span=output_distance" if packed_transpose else "    span=1"]
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
    strided = io_mode == "strided_permuted_store"
    strided_setting = os.getenv("FLAGFFT_HCU_WAVE_STRIDED_INPUT", "0")
    if profile.backend == "hcu" and strided_setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_WAVE_STRIDED_INPUT must be 0 or 1")
    strided_enabled = (strided and strided_setting == "1" and plan.dtype == "complex64"
                       and perm_form.startswith("inner") and _root_table_enabled())
    wide_setting = os.getenv("FLAGFFT_HCU_3D_WARP_FP64", "0")
    if profile.backend == "hcu" and wide_setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_3D_WARP_FP64 must be 0 or 1")
    wide = plan.dtype == "complex128" and wide_setting == "1"
    if (profile.backend != "hcu" or profile.warp_size != 64
            or os.getenv("FLAGFFT_HCU_3D_WARP_FFT", "0") != "1"
            or (plan.dtype != "complex64" and not wide)
            or not 64 <= plan.length <= 2048 or plan.length & (plan.length - 1)
            or (wide and plan.length * 8 > profile.warp_size *
                (256 if _wide_root_period(plan) else 128))
            or (io_mode != "permuted_store" and not (wide and io_mode == "permuted_store_r2c")
                and not strided_enabled)
            or span <= 0 or batch <= 0):
        return 0
    inner = perm_form.startswith("inner")
    if batch % span:
        return 0
    target = os.getenv("FLAGFFT_HCU_3D_WARP_PACK", "auto")
    if target not in {"auto", "1", "2", "4", "8", "16", "32"}:
        raise ValueError("FLAGFFT_HCU_3D_WARP_PACK must be auto, 1, 2, 4, 8, 16 or 32")
    pack = (16 if wide else 32) if target == "auto" else int(target)
    # Limit the live payload per physical thread, with a bounded larger
    # experimental budget for wider coalesced output tiles.
    # Logical FFT rows can share a wave; the physical block stays <= 512 lanes.
    lanes = 64
    budget = (64 if plan.length * 8 > 64 * 128 else 32) if wide else _scalar_wave_budget(plan)
    while pack > 1 and plan.length * pack * 2 > min(pack * lanes // 64, 8) * 64 * budget:
        pack //= 2
    if wide and target == "auto":
        # Bound the complete CTA's live FP64 payload as well as each lane's
        # share. Leave register space for coefficients and layout temporaries.
        live_bytes = 2 * (profile.max_dynamic_shared_memory or 65536)
        while pack > 1 and plan.length * pack * 16 > live_bytes:
            pack //= 2
    while not strided and pack > 1 and ((batch // span) % pack
                       if inner else span % pack if io_mode == "permuted_store_r2c" else False):
        pack //= 2
    if wide and target == "auto" and pack < 4:
        # Retain the native store when row alignment prevents a coalesced
        # FP64 real-input tile. Complex outer stores use padded plane grids.
        return 0
    return pack


def build_hcu_warp_store(plan, perm_form, params, pack, span, batch, io_mode="permuted_store"):
    # The wave-local radix decomposition is independent of the IX launcher.
    # Use ordinary Triton math and keep HCU's existing argument signature.
    from .kernels_ix_warp import build_ix_warp_source

    mode = "r2c" if io_mode == "permuted_store_r2c" else "inner" if perm_form.startswith("inner") else "outer"
    wide = plan.dtype == "complex128"
    strided = io_mode == "strided_permuted_store"
    strided_tile = os.getenv("FLAGFFT_HCU_WAVE_STRIDED_TILE", "balanced")
    if strided_tile not in {"balanced", "output"}:
        raise ValueError("FLAGFFT_HCU_WAVE_STRIDED_TILE must be balanced or output")
    lanes = 64
    radix = plan.length // lanes
    chunk = min(radix, 8)
    shared_limit = min(current_profile().max_dynamic_shared_memory // 2, 32768)
    complex_bytes = 16 if wide else 8
    while chunk > 1 and pack * lanes * chunk * complex_bytes > shared_limit:
        chunk //= 2
    name, source = build_ix_warp_source(
        plan.length, pack, mode, plan.direction == "inverse",
        output_radix_chunk=chunk if chunk < radix else 0,
        table_roots=_root_table_enabled() and not wide,
        wide=wide, lane_width=lanes,
        root_period=_wide_root_period(plan),
        strided_input=strided,
        strided_output_major=strided and strided_tile == "output",
        local_codelet=_native_wave_codelet(),
    )
    batched = perm_form.endswith("_batched")
    leaf_name = "hcu_" + name + ("_batched" if batched else "") + "_leaf"
    if batched:
        if "rows_per_transform" not in params or mode != "inner":
            raise ValueError("batched HCU wave stores require the per-cube row count")
        source = source.replace(
            ("    gbase=(row%span)*(nbatch//span)+row//span" if strided else
             f"    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+slot"),
            f"    cube=row//rows_per_transform\n"
            f"    local_row=row%rows_per_transform\n"
            f"    gbase=cube*({plan.length}*rows_per_transform)+(local_row%span)*(rows_per_transform//span)+local_row//span", 1,
        )
        source = source.replace("    stride=nbatch", "    stride=rows_per_transform")
        if strided:
            source = source.replace(f"if (nbatch//span)%{pack}==0:",
                                    f"if (rows_per_transform//span)%{pack}==0:")
        else:
            hint = f"    gbase=tl.max_contiguous(tl.multiple_of(gbase,{pack}),{pack})"
            source = source.replace(hint, f"    if (rows_per_transform//span)%{pack}==0:\n    " + hint)
    aliases = [
        "    inp=in_ptr", "    out=out_ptr",
        f"    span=tl.full((),{span},tl.int64)",
        f"    nbatch=tl.full((),{batch},tl.int64)",
        f"    input_distance=tl.full((),{plan.length},tl.int64)",
        f"    output_distance=tl.full((),{plan.length},tl.int64)",
    ]
    if wide or _root_table_enabled():
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
