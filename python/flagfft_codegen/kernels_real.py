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

"""Real-transform pointwise kernels, including packed even-length transforms."""

import os
from textwrap import dedent

from .kernels_common import (
    _dtype_suffix,
    _maca_backend_active,
    _next_power_of_two,
    _zero_other,
    direct_dft_batch_tile,
)
from .maca_tail_policy import real_tree_default
from .backend_profile import current_profile


def _build_real_direct_dft_kernel_source(
    n: int, dtype: str, *, inverse: bool = False
) -> tuple[str, str, list[str]]:
    """Direct DFT with real/compact boundaries and the existing DFT-table ABI.

    Like the complex DirectDFT, FP32 reduces 32 input rows at once and FP64
    uses compensated accumulation. Tables carry the transform sign; C2R is
    unnormalised, matching the public C API for every input magnitude.
    MACA FP64 lengths <=32 can opt into an explicit balanced addition tree.
    """
    if not 1 <= n <= 128:
        raise ValueError("real DirectDFT requires 1 <= length <= 128")
    if dtype not in {"complex64", "complex128"}:
        raise ValueError("real DirectDFT requires complex64 or complex128")
    if (dtype == "complex64" and 2 <= n <= 32 and current_profile().backend == "npu"
            and os.environ.get("FLAGFFT_NPU_DIRECT_DFT_PREFETCH") == "1"
            and direct_dft_batch_tile(n, dtype=dtype) in (8, 16, 32)):
        from .kernels_npu_direct_prefetch import build_prefetched_direct_dft
        return build_prefetched_direct_dft(n, "c2r" if inverse else "r2c",
            "inverse" if inverse else "forward", direct_dft_batch_tile(n, dtype=dtype),
            pair=os.environ.get("FLAGFFT_NPU_DIRECT_DFT_PAIR") == "1")
    half = n // 2 + 1
    outputs = n if inverse else half
    block = _next_power_of_two(outputs)
    acc_dtype = "tl.float64" if dtype == "complex128" else "tl.float32"
    kind = "c2r" if inverse else "r2c"
    name = f"direct_dft_{kind}_kernel_n{n}_{_dtype_suffix(dtype)}_b{block}"
    tree = (
        dtype == "complex128" and n <= 32
        and os.environ.get("FLAGFFT_MACA_REAL_DFT_REDUCTION", "tree" if real_tree_default(n) else "kahan") == "tree"
        and _maca_backend_active()
    )
    if tree:
        name += "_tree"
    # Scalar FP64 loads and vector FP32 loads share the same boundary rules.
    if inverse:
        nyquist = f" | (j == {n // 2})" if n % 2 == 0 else ""
        loads = f"""
                source_j = tl.where(j < {half}, j, {n} - j)
                src = in_ptr + (pid_batch * {half} + source_j) * 2
                xr = tl.load(src, mask=j_mask, other=0.0)
                xi = tl.load(src + 1, mask=j_mask, other=0.0)
                xi = tl.where(j < {half}, xi, -xi)
                xi = tl.where((j == 0){nyquist}, 0.0, xi)
        """
        term_r = "xr * wr - xi * wi"
    else:
        loads = f"""
                xr = tl.load(in_ptr + pid_batch * {n} + j, mask=j_mask, other=0.0)
        """
        term_r = "xr * wr"
    init_i = "" if inverse else f"acc_i = tl.zeros(({block},), dtype={acc_dtype})"
    if tree:
        init = ""
        lines = []
        # Each output lane computes independent FP64 terms from the existing
        # FP64 input/table ABI. DFT[j,k] == DFT[k,j] exactly: the table builder
        # forms the integer product before the angle. Read contiguous k lanes.
        # Explicit additions avoid the serial Kahan chain and cross-lane sums.
        for j in range(n):
            lines.extend([f"j = {j}", f"j_mask = j < {n}"])
            lines.extend(dedent(loads).strip().splitlines())
            lines.extend([
                f"wr = tl.load(dft_r_ptr + j * {n} + k, mask=mask, other=0.0)",
                f"wi = tl.load(dft_i_ptr + j * {n} + k, mask=mask, other=0.0)",
                f"tree_r_0_{j} = {term_r}",
            ])
            if not inverse:
                lines.append(f"tree_i_0_{j} = xr * wi")
        for component in ("r",) if inverse else ("r", "i"):
            terms = [f"tree_{component}_0_{j}" for j in range(n)]
            level = 0
            while len(terms) > 1:
                level += 1
                parents = []
                for j in range(0, len(terms) - 1, 2):
                    parent = f"tree_{component}_{level}_{j // 2}"
                    lines.append(f"{parent} = {terms[j]} + {terms[j + 1]}")
                    parents.append(parent)
                if len(terms) % 2:
                    parents.append(terms[-1])
                terms = parents
            lines.append(f"acc_{component} = {terms[0]}")
        loop = "            " + "\n            ".join(lines)
    elif dtype == "complex128":
        init = f"comp_r = tl.zeros(({block},), dtype={acc_dtype})"
        if not inverse:
            init += f"\ncomp_i = tl.zeros(({block},), dtype={acc_dtype})"
        imag = "" if inverse else """
                corrected_i = xr * wi - comp_i
                next_i = acc_i + corrected_i
                comp_i = (next_i - acc_i) - corrected_i
                acc_i = next_i
        """
        loop = f"""
            for j in tl.range(0, {n}):
                j_mask = j < {n}
                {loads}
                wr = tl.load(dft_r_ptr + k * {n} + j, mask=mask, other=0.0)
                wi = tl.load(dft_i_ptr + k * {n} + j, mask=mask, other=0.0)
                corrected_r = {term_r} - comp_r
                next_r = acc_r + corrected_r
                comp_r = (next_r - acc_r) - corrected_r
                acc_r = next_r
                {imag}
        """
    else:
        init = ""
        imag = "" if inverse else "acc_i += tl.sum(xr * wi, axis=0)"
        loop = f"""
            for j_base in tl.static_range(0, {n}, 32):
                j = j_base + tl.arange(0, 32)[:, None]
                j_mask = j < {n}
                {loads}
                matrix_mask = j_mask & mask[None, :]
                matrix_offsets = j * {n} + k[None, :]
                wr = tl.load(dft_r_ptr + matrix_offsets, mask=matrix_mask, other=0.0)
                wi = tl.load(dft_i_ptr + matrix_offsets, mask=matrix_mask, other=0.0)
                acc_r += tl.sum({term_r}, axis=0)
                {imag}
        """
    if inverse:
        stores = f"tl.store(out_ptr + pid_batch * {n} + k, acc_r, mask=mask)"
    else:
        nyquist = f" | (k == {n // 2})" if n % 2 == 0 else ""
        stores = f"""
            dst = out_ptr + (pid_batch * {half} + k) * 2
            tl.store(dst, acc_r, mask=mask)
            acc_i = tl.where((k == 0){nyquist}, 0.0, acc_i)
            tl.store(dst + 1, acc_i, mask=mask)
        """
    statements = [
        f"k = tl.arange(0, {block})",
        f"mask = k < {outputs}",
        f"acc_r = tl.zeros(({block},), dtype={acc_dtype})",
    ]
    statements.extend(line for line in init_i.splitlines() if line)
    statements.extend(dedent(init).strip("\n").splitlines())
    statements.extend(dedent(loop).strip("\n").splitlines())
    statements.extend(dedent(stores).strip("\n").splitlines())
    batch_tile = direct_dft_batch_tile(n, dtype=dtype)
    if batch_tile > 1:
        name += f"_bt{batch_tile}"
        bt = batch_tile
        statements = [
            "pid = tl.program_id(0)",
            f"batch = pid * {bt} + tl.arange(0, {bt})",
            "bmask = batch < nbatch",
            f"k = tl.arange(0, {block})",
            f"mask = k < {outputs}",
            f"acc_r = tl.zeros(({bt}, {block}), dtype={acc_dtype})",
        ]
        if inverse:
            for j in range(n):
                source_j = j if j < half else n - j
                nyquist_j = j == 0 or (n % 2 == 0 and j == n // 2)
                statements.append(
                    f"xr = tl.load(in_ptr + (batch * {half} + {source_j}) * 2, "
                    "mask=bmask, other=0.0)"
                )
                if not nyquist_j:
                    statements.append(
                        f"xi = tl.load(in_ptr + (batch * {half} + {source_j}) * 2 + 1, "
                        "mask=bmask, other=0.0)"
                    )
                    if j >= half:
                        statements.append("xi = -xi")
                statements.append(
                    f"wr = tl.load(dft_r_ptr + {j} * {n} + k, mask=mask, other=0.0)"
                )
                if nyquist_j:
                    statements.append("acc_r += xr[:, None] * wr[None, :]")
                else:
                    statements.append(
                        f"wi = tl.load(dft_i_ptr + {j} * {n} + k, mask=mask, other=0.0)"
                    )
                    statements.append(
                        "acc_r += xr[:, None] * wr[None, :] - xi[:, None] * wi[None, :]"
                    )
        else:
            statements.append(f"acc_i = tl.zeros(({bt}, {block}), dtype={acc_dtype})")
            for j in range(n):
                statements.extend([
                    f"xr = tl.load(in_ptr + (batch * {n} + {j}), mask=bmask, other=0.0)",
                    f"wr = tl.load(dft_r_ptr + {j} * {n} + k, mask=mask, other=0.0)",
                    f"wi = tl.load(dft_i_ptr + {j} * {n} + k, mask=mask, other=0.0)",
                    "acc_r += xr[:, None] * wr[None, :]",
                    "acc_i += xr[:, None] * wi[None, :]",
                ])
        statements.append("store_mask = bmask[:, None] & mask[None, :]")
        if inverse:
            statements.append(
                f"tl.store(out_ptr + batch[:, None] * {n} + k[None, :], acc_r, "
                "mask=store_mask)"
            )
        else:
            statements.extend([
                f"dst = out_ptr + (batch[:, None] * {half} + k[None, :]) * 2",
                "acc_i = tl.where(k[None, :] == 0, 0.0, acc_i)",
                "tl.store(dst, acc_r, mask=store_mask)",
                "tl.store(dst + 1, acc_i, mask=store_mask)",
            ])
    else:
        statements = [
            "pid_batch = tl.program_id(0)",
            "if pid_batch >= nbatch:",
            "    return",
        ] + statements
    body = "\n".join(f"    {line}" if line else line for line in statements)
    source = (
        "@triton.jit\n"
        f"def {name}(\n    in_ptr,\n    out_ptr,\n    dft_r_ptr,\n    dft_i_ptr,\n    nbatch,\n):\n"
        + body
        + "\n"
    )
    return name, source, ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"]


def _packed_layout(n_cols: int, elements_per_program: int = 256) -> tuple[int, int]:
    """Choose a (columns, rows-per-block) tile for tiny row-wise kernels.

    Rows shorter than one program tile are packed to keep the program busy;
    longer rows keep one row per program with the column axis spread across
    grid.x (the historical behavior).
    """
    block_cols = min(elements_per_program, _next_power_of_two(n_cols))
    rows_per_block = max(1, elements_per_program // block_cols)
    return block_cols, rows_per_block


def _build_real_to_complex_kernel_source(
    n: int, dtype: str, elements_per_program: int = 256
) -> tuple[str, list[str], list[str]]:
    block_cols, rows_per_block = _packed_layout(n, elements_per_program)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_real_to_complex_kernel_n{n}_{suffix}"
    npu = current_profile().backend == "npu" and dtype == "complex64"
    if npu:
        R, C = rows_per_block, block_cols
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(
                in_ptr,
                out_ptr,
                input_distance,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                rows = pid_batch * {R} + tl.arange(0, {R})[:, None]
                col_lane = tl.arange(0, {C})[None, :]
                out_lane = tl.arange(0, {2 * C})[None, :]
                row_mask = rows < nbatch
                cols = pid_block * {C} + col_lane
                mask = row_mask & (cols < {n})
                safe_rows = tl.minimum(rows, nbatch - 1)
                xr = tl.load(
                    in_ptr + safe_rows * input_distance + pid_block * {C} + col_lane,
                    mask=mask,
                    other={zero},
                )
                packed = tl.zeros(({R}, {2 * C}), tl.float32)
                packed = tle.dsa.insert_slice(
                    packed, xr, [0, 0], [{R}, {C}], [1, 1])
                out_idx = (out_lane // 2) + (out_lane % 2) * {C} + 0 * safe_rows
                result = tl.gather(packed, out_idx, 1)
                out_mask = row_mask & (
                    out_lane < 2 * ({n} - pid_block * {C}))
                tl.store(
                    out_ptr + (safe_rows * {n} + pid_block * {C}) * 2 + out_lane,
                    result,
                    mask=out_mask,
                )
            """
        )
        return (kernel_name, source,
                ["in_ptr", "out_ptr", "input_distance", "nbatch"],
                rows_per_block)
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            xr = tl.load(
                in_ptr + safe_rows * input_distance + safe_cols, mask=mask, other={zero}
            )
            dst = out_ptr + (safe_rows * {n} + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, 0.0, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_r2c_half_pack_kernel_source(
    n: int, dtype: str, elements_per_program: int = 256
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    block_cols, rows_per_block = _packed_layout(half, elements_per_program)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_half_pack_kernel_n{n}_{suffix}"
    npu = current_profile().backend == "npu" and dtype == "complex64"
    if npu:
        R, C = rows_per_block, block_cols
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(
                in_ptr,
                out_ptr,
                output_distance,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                rows = pid_batch * {R} + tl.arange(0, {R})[:, None]
                lane = tl.arange(0, {2 * C})[None, :]
                col = pid_block * {C} + tl.arange(0, {C})[None, :]
                row_mask = rows < nbatch
                src_mask = row_mask & (lane < 2 * ({half} - pid_block * {C}))
                safe_rows = tl.minimum(rows, nbatch - 1)
                src = tl.load(
                    in_ptr + (safe_rows * {n} + pid_block * {C}) * 2 + lane,
                    mask=src_mask,
                    other={zero},
                )
                out_lane = lane
                result = src
                out_mask = row_mask & (
                    out_lane < 2 * ({half} - pid_block * {C}))
                tl.store(
                    out_ptr + (safe_rows * output_distance + pid_block * {C}) * 2
                    + out_lane,
                    result,
                    mask=out_mask,
                )
            """
        )
        return (kernel_name, source,
                ["in_ptr", "out_ptr", "output_distance", "nbatch"],
                rows_per_block)
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {half})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {half - 1})
            src = in_ptr + (safe_rows * {n} + safe_cols) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            dst = out_ptr + (safe_rows * output_distance + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


def _npu_packed_boundary_groups(n, paired):
    """Amortize adjacent windows without increasing a window's UB state."""
    groups = int(os.getenv("FLAGFFT_NPU_REAL_BOUNDARY_GROUP", "2" if n >= 65536 else "1"))
    if groups not in (1, 2):
        raise ValueError("NPU real boundary group must be 1 or 2")
    return groups if paired else 1


def _build_npu_packed_boundary(n, inverse, count=256, paired=False):
    """Use contiguous complex windows and local reversal for the real boundary."""
    packed=n//2;half=packed+1
    if count not in (256,512,1024,2048):
        raise ValueError("NPU packed boundary tile must be 256, 512, 1024 or 2048")
    name=f"_{'c2r_packed_preprocess' if inverse else 'r2c_packed_postprocess'}_kernel_n{n}_f32"
    if paired: name += "_paired"
    groups = _npu_packed_boundary_groups(n, paired)
    if groups > 1: name += f"_g{groups}"
    distance='input_distance' if inverse else 'output_distance'
    source=f'''@triton.jit
def {name}(in_ptr,twiddle_ptr,out_ptr,{distance},nbatch):
    batch=tl.program_id(1).to(tl.int64)
    start=tl.program_id(0)*{count}
    lane=tl.arange(0,{count})
    word=tl.arange(0,{2*count})
    k=start+lane
    valid=(batch<nbatch)&(k<{packed if inverse else half})
    reverse_start=tl.where({packed}-start-{count}+1>0,{packed}-start-{count}+1,0)
    stride={distance if inverse else packed}
    forward=tl.load(in_ptr+batch*stride*2+start*2+word,
        (batch<nbatch)&(start*2+word<{2*half if inverse else 2*packed}),0.)
    backward=tl.load(in_ptr+batch*stride*2+reverse_start*2+word,
        (batch<nbatch)&(reverse_start*2+word<{2*half if inverse else 2*packed}),0.)
    ar,ai=tl.split(tl.reshape(forward,({count},2)))
    br0,bi0=tl.split(tl.reshape(backward,({count},2)))
    reverse_index=tl.minimum(tl.where({packed}-k-reverse_start>0,{packed}-k-reverse_start,0),{count-1})
    br=tl.gather(br0,reverse_index,0)
    bi=-tl.gather(bi0,reverse_index,0)
'''
    if inverse:
        source+='    ai=tl.where(k==0,0.,ai);bi=tl.where(k==0,0.,bi)\n'
    if not inverse:
        source+=f'''    dc_r=tl.load(in_ptr+batch*{2*packed},batch<nbatch,0.)
    dc_i=tl.load(in_ptr+batch*{2*packed}+1,batch<nbatch,0.)
    ar=tl.where(k=={packed},dc_r,ar);ai=tl.where(k=={packed},dc_i,ai)
    br=tl.where((k==0)|(k=={packed}),dc_r,br)
    bi=tl.where((k==0)|(k=={packed}),-dc_i,bi)
'''
    source+=f'''    phase=tl.load(twiddle_ptr+start*2+word,start*2+word<{2*half},0.)
    wr,wi=tl.split(tl.reshape(phase,({count},2)))
    sr=ar+br;si=ai+bi;dr=ar-br;di=ai-bi
'''
    if inverse:
        source+='    real=sr-(di*wr-dr*wi)\n    imag=si+(dr*wr+di*wi)\n'
        destination=f'batch*{2*packed}+start*2+word'
    else:
        source+='    real=0.5*(sr+di*wr+dr*wi)\n    imag=0.5*(si-dr*wr+di*wi)\n'
        destination='batch*output_distance*2+start*2+word'
    if paired:
        # One pair of loads determines both partners. Store two disjoint
        # contiguous windows, with DC/Nyquist and the midpoint written once.
        center=packed//2
        if inverse:
            source+='    mr=sr+(di*wr-dr*wi)\n    mi=-si+(dr*wr+di*wi)\n'
            base=f'batch*{2*packed}'
        else:
            source+='    mr=0.5*(sr-di*wr-dr*wi)\n    mi=-0.5*(si+dr*wr-di*wi)\n'
            base='batch*output_distance*2'
        source+=f'''    value=tl.reshape(tl.join(real,imag),({2*count},))
    tl.store(out_ptr+{base}+start*2+word,value,(batch<nbatch)&(start*2+word<{2*(center+1)}))
    mirror_start=tl.maximum({packed}-start-{count}+1,0)
    mirror_k=mirror_start+lane
    source_k={packed}-mirror_k
    ix=tl.minimum(tl.maximum(source_k-start,0),{count-1})
    mr=tl.gather(mr,ix,0);mi=tl.gather(mi,ix,0)
    mirror_value=tl.reshape(tl.join(mr,mi),({2*count},))
    mask=(batch<nbatch)&(mirror_start*2+word>={2*(center+1)})&(mirror_start*2+word<{2*packed if inverse else 2*half})
    tl.store(out_ptr+{base}+mirror_start*2+word,mirror_value,mask)
'''
    else:
        source+=f'''    value=tl.reshape(tl.join(real,imag),({2*count},))
    tl.store(out_ptr+{destination},value,(batch<nbatch)&(start*2+word<{2*packed if inverse else 2*half}))
'''
    if groups > 1:
        lines = source.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("    start="))
        lines = (lines[:start] + [f"    for window in tl.static_range({groups}):"] +
                 ["    " + line.replace("start=tl.program_id(0)*",
                    f"start=(tl.program_id(0)*{groups}+window)*") for line in lines[start:]])
        source = "\n".join(lines) + "\n"
    return name,source,['in_ptr','twiddle_ptr','out_ptr',distance,'nbatch'],1


def _build_r2c_packed_postprocess_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    if n % 2 != 0:
        raise ValueError("packed real FFT requires an even length")
    if current_profile().backend=="npu" and dtype=="complex64" and n>=512:
        return _build_npu_packed_boundary(n,False,int(os.getenv("FLAGFFT_NPU_REAL_BOUNDARY_TILE","256")),os.getenv("FLAGFFT_NPU_PAIRED_REAL_BOUNDARY","0")=="1")
    packed = n // 2
    half = packed + 1
    block = 256
    block_cols, rows_per_block = _packed_layout(half, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_packed_postprocess_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {half})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            k = tl.minimum(col_offsets, {packed})
            a_k = k % {packed}
            b_k = ({packed} - k) % {packed}
            a_ptr = in_ptr + (safe_rows * {packed} + a_k) * 2
            b_ptr = in_ptr + (safe_rows * {packed} + b_k) * 2
            ar = tl.load(a_ptr, mask=mask, other={zero})
            ai = tl.load(a_ptr + 1, mask=mask, other={zero})
            br = tl.load(b_ptr, mask=mask, other={zero})
            bi = -tl.load(b_ptr + 1, mask=mask, other={zero})
            wr = tl.load(twiddle_ptr + k * 2, mask=mask, other={zero})
            wi = tl.load(twiddle_ptr + k * 2 + 1, mask=mask, other={zero})
            sum_r = ar + br
            sum_i = ai + bi
            diff_r = ar - br
            diff_i = ai - bi
            prod_r = diff_r * wr - diff_i * wi
            prod_i = diff_i * wr + diff_r * wi
            out_r = 0.5 * (sum_r + prod_i)
            out_i = 0.5 * (sum_i - prod_r)
            dst = out_ptr + (safe_rows * output_distance + k) * 2
            tl.store(dst, out_r, mask=mask)
            tl.store(dst + 1, out_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "twiddle_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


def _build_c2r_packed_preprocess_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    if n % 2 != 0:
        raise ValueError("packed real FFT requires an even length")
    if current_profile().backend=="npu" and dtype=="complex64" and n>=512:
        return _build_npu_packed_boundary(n,True,int(os.getenv("FLAGFFT_NPU_REAL_BOUNDARY_TILE","256")),os.getenv("FLAGFFT_NPU_PAIRED_REAL_BOUNDARY","0")=="1")
    packed = n // 2
    block = 256
    block_cols, rows_per_block = _packed_layout(packed, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_c2r_packed_preprocess_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {packed})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            k = tl.minimum(col_offsets, {packed - 1})
            q = {packed} - k
            x_ptr = in_ptr + (safe_rows * input_distance + k) * 2
            q_ptr = in_ptr + (safe_rows * input_distance + q) * 2
            xr = tl.load(x_ptr, mask=mask, other={zero})
            xi = tl.load(x_ptr + 1, mask=mask, other={zero})
            qr = tl.load(q_ptr, mask=mask, other={zero})
            qi = -tl.load(q_ptr + 1, mask=mask, other={zero})
            wr = tl.load(twiddle_ptr + k * 2, mask=mask, other={zero})
            wi = tl.load(twiddle_ptr + k * 2 + 1, mask=mask, other={zero})
            sum_r = xr + qr
            sum_i = xi + qi
            diff_r = xr - qr
            diff_i = xi - qi
            # Multiply the difference by conj(W), then by +i.  Omitting the
            # conventional 0.5 here supplies the factor of two needed to turn
            # an unnormalised N/2 inverse FFT into an unnormalised N-point C2R.
            prod_r = diff_r * wr + diff_i * wi
            prod_i = diff_i * wr - diff_r * wi
            packed_r = sum_r - prod_i
            packed_i = sum_i + prod_r
            dst = out_ptr + (safe_rows * {packed} + k) * 2
            tl.store(dst, packed_r, mask=mask)
            tl.store(dst + 1, packed_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "twiddle_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_compact_to_hermitian_full_kernel_source(
    n: int, dtype: str, elements_per_program: int = 256
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    block_cols, rows_per_block = _packed_layout(n, elements_per_program)
    nyquist_guard = f" | (safe_cols == {n // 2})" if n % 2 == 0 else ""
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_compact_to_hermitian_full_kernel_n{n}_{suffix}"
    npu = current_profile().backend == "npu" and dtype == "complex64"
    if npu:
        R, C = rows_per_block, block_cols
        nyquist_out_guard = f" | (freq == {n // 2})" if n % 2 == 0 else ""
        # A bounded contiguous input window contains every mirrored source
        # index of this output tile, including tiles crossing the midpoint.
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(
                in_ptr,
                out_ptr,
                input_distance,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                rows = pid_batch * {R} + tl.arange(0, {R})[:, None]
                lane = tl.arange(0, {2 * C})[None, :]
                row_mask = rows < nbatch
                safe_rows = tl.minimum(rows, nbatch - 1)
                start = pid_block * {C}
                end = tl.minimum(start + {C - 1}, {n - 1})
                src_start = tl.minimum(tl.minimum(start, {half - 1}), {n} - end)
                src = tl.load(
                    in_ptr + (safe_rows * input_distance + src_start) * 2 + lane,
                    mask=row_mask & (lane < 2 * ({half} - src_start)),
                    other={zero},
                )
                freq = tl.minimum(start + lane // 2, {n - 1})
                upper = freq >= {half}
                src_k = tl.where(upper, {n} - freq, freq)
                index = 2 * (src_k - src_start) + lane % 2 + 0 * safe_rows
                result = tl.gather(src, index, 1)
                imag = lane % 2 == 1
                result = tl.where(imag & upper, -result, result)
                result = tl.where(imag & ((freq == 0){nyquist_out_guard}), {zero}, result)
                tl.store(
                    out_ptr + (safe_rows * {n} + start) * 2 + lane,
                    result,
                    mask=row_mask & (lane < 2 * ({n} - start)),
                )
            """
        )
        return (kernel_name, source,
                ["in_ptr", "out_ptr", "input_distance", "nbatch"],
                rows_per_block)
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            src_k = tl.where(safe_cols < {half}, safe_cols, {n} - safe_cols)
            src = in_ptr + (safe_rows * input_distance + src_k) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            xi = tl.where(safe_cols < {half}, xi, -xi)
            xi = tl.where((safe_cols == 0){nyquist_guard}, 0.0, xi)
            dst = out_ptr + (safe_rows * {n} + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_complex_to_real_kernel_source(
    n: int, dtype: str, elements_per_program: int = 256
) -> tuple[str, list[str], list[str]]:
    block_cols, rows_per_block = _packed_layout(n, elements_per_program)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_complex_to_real_kernel_n{n}_{suffix}"
    npu = current_profile().backend == "npu" and dtype == "complex64"
    if npu:
        R, C = rows_per_block, block_cols
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(
                in_ptr,
                out_ptr,
                output_distance,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                rows = pid_batch * {R} + tl.arange(0, {R})[:, None]
                lane = tl.arange(0, {2 * C})[None, :]
                row_mask = rows < nbatch
                safe_rows = tl.minimum(rows, nbatch - 1)
                src_mask = row_mask & (lane < 2 * ({n} - pid_block * {C}))
                src = tl.load(
                    in_ptr + (safe_rows * {n} + pid_block * {C}) * 2 + lane,
                    mask=src_mask,
                    other={zero},
                )
                idx = 2 * tl.arange(0, {C})[None, :] + 0 * safe_rows
                xr = tl.gather(src, idx, 1)
                cols = pid_block * {C} + tl.arange(0, {C})[None, :]
                out_mask = row_mask & (cols < {n})
                tl.store(
                    out_ptr + safe_rows * output_distance + cols,
                    xr,
                    mask=out_mask,
                )
            """
        )
        return kernel_name, source, ["in_ptr", "out_ptr", "output_distance", "nbatch"], rows_per_block
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            src = in_ptr + (safe_rows * {n} + safe_cols) * 2
            xr = tl.load(src, mask=mask, other={zero})
            dst = out_ptr + safe_rows * output_distance + safe_cols
            tl.store(dst, xr, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


__all__ = [
    "_build_real_direct_dft_kernel_source",
    "_build_c2r_packed_preprocess_kernel_source",
    "_build_compact_to_hermitian_full_kernel_source",
    "_build_complex_to_real_kernel_source",
    "_build_r2c_half_pack_kernel_source",
    "_build_r2c_packed_postprocess_kernel_source",
    "_build_real_to_complex_kernel_source",
    "_packed_layout",
]
