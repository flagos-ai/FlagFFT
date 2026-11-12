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

from .kernels_common import (_dtype_suffix, _maca_backend_active, _next_power_of_two, _zero_other,
                             _ix_backend_active, _maca_knob)
from .maca_tail_policy import real_tree_default


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
        loop = "\n            ".join(lines)
    elif dtype == "complex128":
        init = f"comp_r = tl.zeros(({block},), dtype={acc_dtype})"
        if not inverse:
            init += f"\n            comp_i = tl.zeros(({block},), dtype={acc_dtype})"
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
    source = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, dft_r_ptr, dft_i_ptr, nbatch):
            pid_batch = tl.program_id(0)
            if pid_batch >= nbatch:
                return
            k = tl.arange(0, {block})
            mask = k < {outputs}
            acc_r = tl.zeros(({block},), dtype={acc_dtype})
            {init_i}
            {init}
            {loop}
            {stores}
    """)
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


def packed_real_boundary_block(n: int, dtype: str) -> int:
    if not (_ix_backend_active() and dtype=="complex64" and n==16384): return 256
    value=_maca_knob("REAL_BOUNDARY_BLOCK","256")
    if value not in {"256","512","1024","2048"}:
        raise ValueError("FLAGFFT_IX_REAL_BOUNDARY_BLOCK must be 256, 512, 1024 or 2048")
    return int(value)


def _ix_boundary_pair_io(source: str, pairs: tuple, real: str, imag: str) -> str:
    for ptr,r,i,negate in pairs:
        old=(f"    {r} = tl.load({ptr}, mask=mask, other=0.0)\n"
             f"    {i} = {'-' if negate else ''}tl.load({ptr} + 1, mask=mask, other=0.0)")
        new=(f"    pair = tl.load(tl.cast({ptr},tl.pointer_type(tl.uint64)),mask=mask,other=0)\n"
             f"    {r} = tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)\n"
             f"    {i} = {'-' if negate else ''}tl.cast(tl.cast(pair >> 32,tl.uint32),tl.float32,bitcast=True)")
        if old not in source: raise ValueError("missing packed-real pair load")
        source=source.replace(old,new,1)
    old=(f"    tl.store(dst, {real}, mask=mask)\n"
         f"    tl.store(dst + 1, {imag}, mask=mask)")
    new=(f"    pair = tl.cast(tl.cast({real},tl.uint32,bitcast=True),tl.uint64) | "
         f"(tl.cast(tl.cast({imag},tl.uint32,bitcast=True),tl.uint64) << 32)\n"
         "    tl.store(tl.cast(dst,tl.pointer_type(tl.uint64)),pair,mask=mask)")
    if old not in source: raise ValueError("missing packed-real pair store")
    return source.replace(old,new,1)


def _build_r2c_packed_postprocess_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    if n % 2 != 0:
        raise ValueError("packed real FFT requires an even length")
    packed = n // 2
    half = packed + 1
    block = packed_real_boundary_block(n,dtype)
    block_cols, rows_per_block = _packed_layout(half, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_packed_postprocess_kernel_n{n}_{suffix}"
    packed_io = (_ix_backend_active() and dtype=="complex64" and n==16384
                 and _maca_knob("VEC_IO","0")=="packed")
    if block!=256: kernel_name+=f"_b{block}"
    if packed_io: kernel_name+="_u64"
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
    if packed_io:
        source=_ix_boundary_pair_io(source,(("a_ptr","ar","ai",False),("b_ptr","br","bi",True),
            ("twiddle_ptr + k * 2","wr","wi",False)),"out_r","out_i")
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
    packed = n // 2
    block = packed_real_boundary_block(n,dtype)
    block_cols, rows_per_block = _packed_layout(packed, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_c2r_packed_preprocess_kernel_n{n}_{suffix}"
    packed_io = (_ix_backend_active() and dtype=="complex64" and n==16384
                 and _maca_knob("VEC_IO","0")=="packed")
    if block!=256: kernel_name+=f"_b{block}"
    if packed_io: kernel_name+="_u64"
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
    if packed_io:
        source=_ix_boundary_pair_io(source,(("x_ptr","xr","xi",False),("q_ptr","qr","qi",True),
            ("twiddle_ptr + k * 2","wr","wi",False)),"packed_r","packed_i")
    if _ix_backend_active() and dtype=="complex64":
        # The imaginary endpoints do not participate in a real inverse FFT.
        source=source.replace("    sum_r = xr + qr", "    xi = tl.where(k == 0,0.0,xi)\n    qi = tl.where(k == 0,0.0,qi)\n    sum_r = xr + qr",1)
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
