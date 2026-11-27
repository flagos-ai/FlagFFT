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

"""IX warp-local radix-4 x FFT64 prototype; source generator only."""
import math


def strided_input_tile_shape(pack, output_major=False):
    """Balance independent rows across the input and output fast axes."""
    if output_major:
        return pack, 1
    outer = 1
    while outer * outer < pack:
        outer *= 2
    return outer, pack // outer

def _reverse_lane_bits(index, bits):
    return "|".join(f"((({index} >> {bit}) & 1) << {bits - 1 - bit})" for bit in range(bits))


_PRIME_R2C_COLUMNS = 997
_PRIME_R2C_HALF = 33
PRIME_R2C_LAST_TILE = 8
_PRIME_R2C_COLUMN_TILES = (_PRIME_R2C_COLUMNS + PRIME_R2C_LAST_TILE - 1) // PRIME_R2C_LAST_TILE
IX_PRIME_R2C_LAST_GRID = (
    (_PRIME_R2C_HALF + PRIME_R2C_LAST_TILE - 1) // PRIME_R2C_LAST_TILE
) * _PRIME_R2C_COLUMN_TILES


def build_ix_prime_r2c_last_source(params):
    """Apply FFT16 on the strided axis and write the natural compact cube."""
    from .kernels_leaf import _emit_radix16_codelet_call

    name = "ix_prime_r2c_last"
    tile = PRIME_R2C_LAST_TILE
    columns = _PRIME_R2C_COLUMNS
    half = _PRIME_R2C_HALF
    stride = columns * half
    lines = [
        "@triton.jit",
        f"def {name}({', '.join(params)}):",
        "    tile_id = tl.program_id(0)",
        f"    batch_offset = tl.program_id(1) * {16 * stride}",
        f"    half_row = (tile_id // {_PRIME_R2C_COLUMN_TILES}) * {tile} + tl.arange(0, {tile})",
        f"    frequency = (tile_id % {_PRIME_R2C_COLUMN_TILES}) * {tile} + tl.arange(0, {tile})",
        f"    source = batch_offset + half_row[:, None] * {columns} + frequency[None, :]",
        f"    valid = (half_row[:, None] < {half}) & (frequency[None, :] < {columns})",
        "    pairs = tl.cast(in_ptr, tl.pointer_type(tl.uint64))",
    ]
    for axis in range(16):
        lines.extend([
            f"    pair = tl.load(pairs + source + {stride * axis}, valid, 0)",
            f"    r{axis} = tl.cast(tl.cast(pair, tl.uint32), tl.float32, bitcast=True)",
            f"    i{axis} = tl.cast(tl.cast(pair >> 32, tl.uint32), tl.float32, bitcast=True)",
        ])
    lines.extend(_emit_radix16_codelet_call("    ", "forward"))
    lines.extend([
        "    result = tl.cast(out_ptr, tl.pointer_type(tl.uint64))",
        f"    target = batch_offset + frequency[:, None] * {half} + half_row[None, :]",
        "    store_mask = tl.trans(valid)",
    ])
    for axis in range(16):
        lines.extend([
            f"    low = tl.cast(tl.cast(r{axis}, tl.uint32, bitcast=True), tl.uint64)",
            f"    high = tl.cast(tl.cast(i{axis}, tl.uint32, bitcast=True), tl.uint64)",
            f"    tl.store(result + target + {stride * axis}, tl.trans(low | (high << 32)), store_mask)",
        ])
    return name, "\n".join(lines) + "\n"


def _emit_wave_dif(pack, radix, sign, lane_width):
    lines = []
    for stage in reversed(range(lane_width.bit_length() - 1)):
        half = 1 << stage
        lines.extend([
            f"    peer = tl.broadcast_to(lane ^ {half}, ({pack}, {lane_width}))",
            f"    high = (lane & {half}) != 0",
        ])
        if half > 2:
            lines.extend([
                f"    angle = tl.where(high, (lane & {half - 1}) * {sign * math.pi / half:.17g}, 0.0)",
                "    wr = tl.cos(angle)",
                "    wi = tl.sin(angle)",
            ])
        elif half == 2:
            lines.append("    rotate = (lane & 3) == 3")
        for j in range(radix):
            lines.extend([
                f"    pr = tl.gather(r{j}, peer, axis=1)",
                f"    pi = tl.gather(i{j}, peer, axis=1)",
                f"    sr = pr + tl.where(high, -r{j}, r{j})",
                f"    si = pi + tl.where(high, -i{j}, i{j})",
            ])
            if half > 2:
                lines.extend([f"    r{j} = sr * wr - si * wi", f"    i{j} = sr * wi + si * wr"])
            elif half == 2:
                lines.extend([f"    r{j} = tl.where(rotate, {-sign}.0 * si, sr)",
                              f"    i{j} = tl.where(rotate, {sign}.0 * sr, si)"])
            else:
                lines.extend([f"    r{j} = sr", f"    i{j} = si"])
    return lines


def _join_ct_columns(lines, values, prefix):
    if len(values) == 1:
        return values[0]
    even = _join_ct_columns(lines, values[::2], prefix + "0")
    odd = _join_ct_columns(lines, values[1::2], prefix + "1")
    lines.append(f"    {prefix} = tl.join({even}, {odd})")
    return prefix


def build_ix_ct_packed_inverse_source(plan, params):
    """Fuse the Hermitian boundary with two local inverse FFT stages."""
    from .kernels_leaf import _emit_natural_order_codelet_call
    from .metadata import _module_source

    n = plan.length
    inner = 256
    groups = n // inner
    radix = inner // 64
    sign = 1
    name = f"ix_ct_two_wave_{n}_{inner}_inverse_packed_c2r"
    lines = [
        "from triton.language.extra import libdevice as ld",
        "@triton.jit",
        f"def {name}({', '.join(params)}):",
        "    row = tl.program_id(0)",
        f"    sample = tl.arange(0, {inner})",
    ]
    lines += [
        "    ip = tl.cast(in_ptr, tl.pointer_type(tl.uint64)) + row * input_distance",
        f"    op = tl.cast(out_ptr, tl.pointer_type(tl.uint64)) + row * {n}",
        "    roots = tl.cast(packed_twiddle_ptr, tl.pointer_type(tl.uint64))",
    ]
    for group in range(groups):
        lines.extend([
            f"    k = sample + {group * inner}",
            "    pair = tl.load(ip + k, row < nbatch, 0)",
            f"    r{group} = tl.cast(tl.cast(pair, tl.uint32), tl.float32, bitcast=True)",
            f"    i{group} = tl.cast(tl.cast(pair >> 32, tl.uint32), tl.float32, bitcast=True)",
        ])
        lines.extend([
            f"    peer = tl.load(ip + {n} - k, row < nbatch, 0)",
            "    pr = tl.cast(tl.cast(peer, tl.uint32), tl.float32, bitcast=True)",
            "    pi = tl.cast(tl.cast(peer >> 32, tl.uint32), tl.float32, bitcast=True)",
            "    root = tl.load(roots + k, row < nbatch, 0)",
            "    wr = tl.cast(tl.cast(root, tl.uint32), tl.float32, bitcast=True)",
            "    wi = tl.cast(tl.cast(root >> 32, tl.uint32), tl.float32, bitcast=True)",
            f"    xi = tl.where(k == 0, 0.0, i{group})",
            "    pi = tl.where(k == 0, 0.0, pi)",
            f"    dr = r{group} - pr",
            "    di = xi + pi",
            f"    r{group} = r{group} + pr - (di * wr - dr * wi)",
            f"    i{group} = xi - pi + (dr * wr + di * wi)",
        ])
    lines += _emit_natural_order_codelet_call("    ", groups, plan.direction)
    lines += [
        f"    angle = sample * {sign * 2 * math.pi / n:.17g}",
        "    base_r = ld.fast_cosf(angle)",
        "    base_i = ld.fast_sinf(angle)",
        "    wr = base_r",
        "    wi = base_i",
    ]
    for group in range(1, groups):
        if group > 1:
            lines += ["    next_r = wr * base_r - wi * base_i",
                      "    wi = wr * base_i + wi * base_r", "    wr = next_r"]
        lines += [f"    tr = r{group} * wr - i{group} * wi",
                  f"    i{group} = r{group} * wi + i{group} * wr", f"    r{group} = tr"]

    # The shape change gives each wave adjacent samples from a single tail FFT.
    for component in ("r", "i"):
        values = [f"{component}{group}" for group in range(groups)]
        joined = _join_ct_columns(lines, values, component + "_first")
        lines.append(f"    tile = tl.trans(tl.reshape({joined}, ({inner}, {groups})))")
        lines += [
            f"    tile = tl.permute(tl.reshape(tile, ({groups}, {radix}, 64)), (0, 2, 1))",
            f"    even, odd = tl.split(tl.reshape(tile, ({groups}, 64, 2, 2)))",
            f"    {component}0, {component}2 = tl.split(even)",
            f"    {component}1, {component}3 = tl.split(odd)",
        ]
    lines += _emit_natural_order_codelet_call("    ", radix, plan.direction)
    lines.append("    lane = tl.arange(0, 64)[None, :]")
    for digit in range(1, radix):
        lines += [
            f"    angle = lane * {digit * sign * 2 * math.pi / inner:.17g}",
            "    wr = ld.fast_cosf(angle)",
            "    wi = ld.fast_sinf(angle)",
            f"    tr = r{digit} * wr - i{digit} * wi",
            f"    i{digit} = r{digit} * wi + i{digit} * wr",
            f"    r{digit} = tr",
        ]
    lines += _emit_wave_dif(groups, radix, sign, 64)
    for digit in range(radix):
        lines += [
            f"    low = tl.cast(tl.cast(r{digit}, tl.uint32, bitcast=True), tl.uint64)",
            f"    high = tl.cast(tl.cast(i{digit}, tl.uint32, bitcast=True), tl.uint64)",
            f"    pair{digit} = low | (high << 32)",
        ]
    combined = _join_ct_columns(lines, [f"pair{digit}" for digit in range(radix)], "tail")
    lines += [
        f"    values = tl.trans(tl.reshape({combined}, ({groups}, {inner})))",
        f"    freq = tl.arange(0, {inner})",
        f"    tail_lane = freq // {radix}",
        f"    freq = ({_reverse_lane_bits('tail_lane', 6)}) * {radix} + freq % {radix}",
        f"    offset = freq[:, None] * {groups} + tl.arange(0, {groups})[None, :]",
    ]
    lines.append("    tl.store(op + offset, values, row < nbatch)")
    source = "\n".join(lines).replace("tl.cos(", "ld.fast_cosf(").replace("tl.sin(", "ld.fast_sinf(")
    codelets = tuple(sorted({groups, radix}))
    return name, _module_source(source, codelets)


def build_ix_warp256_source(pack: int = 8, mode: str = "outer", inverse: bool = False) -> tuple[str, str]:
    return build_ix_warp_source(256, pack, mode, inverse)


def _emit_band_input(n, bands, mode, sign, digit):
    """Fold the time groups for one residue class of the output frequency."""
    length = n * bands
    lines = []
    for group in range(bands // 2):
        for half in (0, 1):
            address = (group + half * (bands // 2)) * n
            lines += [f"    k=pos+{address}"]
            if mode == "contiguous_r2c":
                lines += [f"    r{half}t=tl.load(inp+k)"]
            else:
                if mode == "c2r":
                    lines += [f"    pair=tl.load(ip+tl.minimum(k,{length}-k))"]
                else:
                    lines += ["    pair=tl.load(ip+k)"]
                lines += [
                    f"    r{half}t=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)",
                    f"    i{half}t=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)",
                ]
                if mode == "c2r":
                    lines += [
                        f"    i{half}t=tl.where(k<={length//2},i{half}t,-i{half}t)",
                        f"    i{half}t=tl.where((k==0)|(k=={length//2}),0.0,i{half}t)",
                    ]
        lines += ["    vr=r0t+pair_sign*r1t"]
        if mode == "contiguous_r2c":
            lines += ["    vi=tl.full((1,64),0.0,tl.float32)"]
        else:
            lines += ["    vi=i0t+pair_sign*i1t"]
        if group == 0:
            lines += ["    sum_r=vr", "    sum_i=vi"]
        else:
            lines += [
                f"    sum_r=sum_r+vr*group_r{group}-vi*group_i{group}",
                f"    sum_i=sum_i+vr*group_i{group}+vi*group_r{group}",
            ]
    lines += [
        f"    angle=pos*band*{sign*2*math.pi/length:.17g}",
        "    wr=tl.cos(angle)",
        "    wi=tl.sin(angle)",
        f"    x{digit}r=sum_r*wr-sum_i*wi",
        f"    x{digit}i=sum_r*wi+sum_i*wr",
    ]
    return lines


def _build_ix_pfa210(mode, inverse, batch_pack=0):
    """Seven independent residue classes of the 5 x 6 x 7 prime-factor FFT."""
    name = f"ix_warp210_pfa_{mode}_{'inv' if inverse else 'fwd'}_frequency_bands7_leaf"
    if batch_pack:
        name += f"_batch{batch_pack}"
    sign = 1 if inverse else -1
    lines = ["import triton", "import triton.language as tl", "@triton.jit",
             f"def {name}(in_ptr,out_ptr,nbatch):"]
    if batch_pack:
        input_n = 106 if mode == "c2r" else 210
        output_n = 106 if mode == "contiguous_r2c" else 210
        lines += ["    pid=tl.program_id(0)", "    band=pid%7",
                  f"    batch=(pid//7)*{batch_pack}+tl.arange(0,{batch_pack})[:,None]",
                  "    lane=tl.arange(0,32)[None,:]",
                  "    active=batch<nbatch",
                  f"    input_base=batch*{input_n}",
                  f"    output_base=batch*{output_n}"]
    else:
        lines += ["    band=tl.program_id(0)", "    lane=tl.arange(0,32)",
                  "    active=True", "    input_base=0", "    output_base=0"]
    zero_shape = f"({batch_pack},32)" if batch_pack else "(32,)"
    lines += ["    p=tl.minimum(lane//6,4)",
             "    q=lane%6",
             "    base=42*p+35*q",
             "    ip=tl.cast(in_ptr,tl.pointer_type(tl.uint64))"]
    for j in range(7):
        lines += [f"    k=(base+{30*j})%210"]
        if mode == "contiguous_r2c":
            lines += [f"    x{j}r=tl.load(in_ptr+input_base+k,active,0.0)",
                      f"    x{j}i=tl.full({zero_shape},0.0,tl.float32)"]
        else:
            if mode == "c2r":
                lines += ["    src=tl.minimum(k,210-k)"]
            else:
                lines += ["    src=k"]
            lines += ["    pair=tl.load(ip+input_base+src,active,0)",
                      f"    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)",
                      f"    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)"]
            if mode == "c2r":
                lines += [f"    x{j}i=tl.where((k==0)|(k==105),0.0,tl.where(k>105,-x{j}i,x{j}i))"]
    def fold(radix, frequency, source, result):
        lines.extend([f"    {result}r={source}0r", f"    {result}i={source}0i"])
        for j in range(1, (radix + 1)//2):
            other = radix - j
            lines.extend([
                f"    phase=(({j}*{frequency})%{radix})*{sign*2*math.pi/radix:.17g}",
                "    c=tl.cos(phase)",
                "    s=tl.sin(phase)",
                f"    {result}r=tl.fma(c,{source}{j}r+{source}{other}r,{result}r)"
                f"-s*({source}{j}i-{source}{other}i)",
                f"    {result}i=tl.fma(c,{source}{j}i+{source}{other}i,{result}i)"
                f"+s*({source}{j}r-{source}{other}r)",
            ])
        if radix % 2 == 0:
            lines.extend([f"    edge=tl.where(({frequency}&1)==0,1.0,-1.0)",
                          f"    {result}r=tl.fma(edge,{source}{radix//2}r,{result}r)",
                          f"    {result}i=tl.fma(edge,{source}{radix//2}i,{result}i)"])
    axis = 1 if batch_pack else 0
    fold(7, "band", "x", "a")
    for j in range(5):
        index = f"{6*j}+q"
        if batch_pack:
            index = f"tl.broadcast_to({index},({batch_pack},32))"
        lines += [f"    v{j}r=tl.gather(ar,{index},{axis})",
                  f"    v{j}i=tl.gather(ai,{index},{axis})"]
    fold(5, "p", "v", "b")
    for j in range(6):
        index = f"6*p+{j}"
        if batch_pack:
            index = f"tl.broadcast_to({index},({batch_pack},32))"
        lines += [f"    w{j}r=tl.gather(br,{index},{axis})",
                  f"    w{j}i=tl.gather(bi,{index},{axis})"]
    fold(6, "q", "w", "y")
    lines += ["    k=(126*p+175*q+120*band)%210", "    valid=(lane<30)&active"]
    if mode == "c2r":
        lines += ["    tl.store(out_ptr+output_base+k,yr,valid)"]
    else:
        if mode == "contiguous_r2c":
            lines += ["    valid=valid&(k<=105)", "    yi=tl.where((k==0)|(k==105),0.0,yi)"]
        lines += [
            "    out=tl.cast(out_ptr,tl.pointer_type(tl.uint64))",
            "    yr_bits=tl.cast(tl.cast(yr,tl.uint32,bitcast=True),tl.uint64)",
            "    yi_bits=tl.cast(tl.cast(yi,tl.uint32,bitcast=True),tl.uint64)",
            "    pair=yr_bits|(yi_bits<<32)",
            "    tl.store(out+output_base+k,pair,valid)",
        ]
    return _finish_warp_source(name, lines, True, 0, False)


def build_ix_warp_bands(plan, bands, mode, permutation_batch=0):
    if plan.length == 210 and bands == 7:
        from .target import ix_warp_fft_pack
        return _build_ix_pfa210(mode, plan.direction == "inverse",
                                ix_warp_fft_pack() if permutation_batch else 0)
    name, source = build_ix_warp_source(plan.length, 1, mode, plan.direction == "inverse",
                                        fast_math=True, local_codelet=True, frequency_bands=bands)
    leaf_name = "ix_" + name + "_leaf"
    header = f"def {name}(inp,out,span,nbatch,input_distance,output_distance):"
    return leaf_name, source.replace(header,
        f"def {leaf_name}(in_ptr,out_ptr,nbatch):\n    inp=in_ptr\n    out=out_ptr", 1)


def build_ix_warp_source(n: int, pack: int = 8, mode: str = "outer", inverse: bool = False,
                         *, recurrence: bool = False, fast_math: bool = False,
                         local_codelet: bool = False, manual_store: bool = False,
                         output_radix_chunk: int = 0, lane_width: int = 64,
                         packed_gather: bool = False,
                         four_step_other: int = 0, wide: bool = False,
                         table_roots: bool = False, root_period: int = 0,
                         strided_input: bool = False,
                         strided_output_major: bool = False,
                         packed_output_transpose: bool = False,
                         packed_c2r_input: bool = False,
                         dif_store: bool = False,
                         ragged_r2c: bool = False,
                         frequency_bands: int = 0,
                         input_padding: int = 0,
                         output_padding: int = 0) -> tuple[str, str]:
    """Screen register FFTs followed by a lane-local FFT64; no native dispatch."""
    if n not in {16,32,64,128,256,512,1024,2048}:
        raise ValueError("unsupported IX warp FFT length")
    if n<64 and lane_width==64:lane_width=n
    if lane_width<16 or lane_width>n or lane_width&(lane_width-1) or n%lane_width:
        raise ValueError("invalid IX FFT lane width")
    if manual_store and lane_width!=64:
        raise ValueError("manual output experiments require 64-lane FFTs")
    radix=n//lane_width
    chunk=output_radix_chunk or radix
    if chunk<1 or chunk>radix or chunk&(chunk-1) or (manual_store and output_radix_chunk):
        raise ValueError("invalid IX warp output chunk")
    if pack not in {1,2,4,8,16,32,64}:
        raise ValueError("IX warp FFT pack must be 1, 2, 4, 8, 16, 32 or 64")
    if root_period not in {0, 4, 8} or (root_period and not (wide or table_roots)):
        raise ValueError("periodic wave roots require supplied tables and period 4 or 8")
    if strided_input and (mode not in {"inner", "outer"} or manual_store):
        raise ValueError("strided wave input requires a permuted store")
    if strided_output_major and not strided_input:
        raise ValueError("output-major row tiles require strided input")
    if packed_output_transpose and mode != "packed_r2c":
        raise ValueError("packed output transpose requires a forward real boundary")
    if packed_c2r_input and (mode != "c2r" or wide):
        raise ValueError("packed input requires an FP32 inverse real boundary")
    row_modes={'four_step_row','four_step_real_row','four_step_hermitian_row'}
    col_modes={'four_step_col','four_step_r2c_col','four_step_c2r_col'}
    if mode not in {"contiguous","outer","inner","r2c","contiguous_r2c","c2r","packed_r2c","packed_c2r"}|row_modes|col_modes:
        raise ValueError("unsupported IX warp FFT I/O mode")
    four_step=mode in row_modes|col_modes
    if wide and (mode not in {"contiguous", "contiguous_r2c", "c2r", "packed_r2c", "packed_c2r", "outer", "inner", "r2c"}|row_modes|col_modes
                 or recurrence or fast_math or manual_store or packed_gather):
        raise ValueError("wide wave FFT requires supplied FP64 roots and ordinary math")
    if four_step and (four_step_other<1 or lane_width!=64 or manual_store):
        raise ValueError("invalid IX warp FourStep geometry")
    if (mode in {"r2c","contiguous_r2c","packed_r2c","four_step_real_row","four_step_r2c_col"} and inverse) or (mode in {"c2r","packed_c2r","four_step_hermitian_row","four_step_c2r_col"} and not inverse):
        raise ValueError("real transform direction disagrees with the I/O mode")
    if dif_store:
        supported_layout = (
            n == 256 and mode in {"outer", "inner", "r2c"} and lane_width in {32, 64}
            or n == 2048 and mode == "outer" and lane_width == 64
        )
        if (not supported_layout or manual_store or strided_input
                or wide or table_roots or packed_gather):
            raise ValueError("DIF store requires a supported FP32 wave with a permuted output")
    if ragged_r2c and (mode != "r2c" or manual_store):
        raise ValueError("ragged tiles require a regular R2C permuted store")
    if frequency_bands and (frequency_bands not in {2,4,8,16}
            or n * frequency_bands not in {1024,2048} or not 64 <= n <= 512
            or pack != 1 or wide or lane_width != 64
            or mode not in {"contiguous", "contiguous_r2c", "c2r"}):
        raise ValueError("frequency bands require one FP32 1024- or 2048-point FFT")
    sign=1 if inverse else -1
    name=f'warp{n}_{mode}_p{pack}_{"inv" if inverse else "fwd"}'
    if four_step: name+=f'_other{four_step_other}'
    if recurrence: name+='_rec'
    if fast_math: name+='_fast'
    if local_codelet: name+='_codelet'
    if manual_store: name+='_manual'
    if output_radix_chunk: name+=f'_chunk{chunk}'
    if ragged_r2c: name+='_ragged'
    if input_padding: name+=f'_padin{input_padding}'
    if output_padding: name+=f'_padout{output_padding}'
    if frequency_bands: name+=f'_frequency_bands{frequency_bands}'
    if lane_width!=64: name+=f'_lanes{lane_width}'
    if packed_gather: name+='_pairgather'
    if wide: name+='_f64'
    if table_roots: name+='_roots'
    if root_period: name+=f'_period{root_period}'
    if strided_input: name+='_stridedinput'
    if strided_output_major: name+='_outtile'
    if packed_output_transpose: name+='_transpose'
    if packed_c2r_input: name+='_packedin'
    if dif_store: name+='_dif'
    a=['import triton','import triton.language as tl','@triton.jit',
       f'def {name}(inp,out,span,nbatch,input_distance,output_distance):',
       '    pid=tl.program_id(0)',f'    slot=tl.arange(0,{pack})',
       f'    lane=tl.arange(0,{lane_width})[None,:]']
    if strided_input and mode == "inner":
        outer_tile, span_tile = strided_input_tile_shape(pack, strided_output_major)
        a += [f'    span_groups=(span+{span_tile-1})//{span_tile}',
              f'    outer_row=(pid//span_groups)*{outer_tile}+slot//{span_tile}',
              f'    span_row=(pid%span_groups)*{span_tile}+slot%{span_tile}',
              '    row=outer_row*span+span_row',
              '    gbase=(row%span)*(nbatch//span)+row//span']
    elif mode=='inner':
        a+= [f'    row_base=(pid//span)*{pack}*span+pid%span',
             '    row=row_base+slot*span',
             f'    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+slot']
    elif mode=='outer' or ragged_r2c:
        output_length = n // 2 + 1 if ragged_r2c else n
        a += [f'    groups=(span+{pack-1})//{pack}',
              f'    row_base=(pid//groups)*span+(pid%groups)*{pack}',
              '    row=row_base+slot',
              f'    gbase=(pid//groups)*({output_length}*(span+{output_padding}))+(pid%groups)*{pack}+slot']
    else:
        a+= [f'    row_base=pid*{pack}','    row=row_base+slot']
        if mode in {'outer','r2c'}:
            outn=n//2+1 if mode=='r2c' else n
            a+=[f'    gbase=(row_base//span)*({outn}*span)+(row_base%span)+slot']
    if frequency_bands:
        a += ["    band=pid", "    pair_sign=tl.where((band&1)==0,1.0,-1.0)"]
        if mode != "contiguous_r2c":
            a += ["    ip=tl.cast(inp,tl.pointer_type(tl.uint64))"]
        for group in range(1, frequency_bands // 2):
            a += [
                f"    angle=band*{sign*2*math.pi*group/frequency_bands:.17g}",
                f"    group_r{group}=tl.cos(angle)",
                f"    group_i{group}=tl.sin(angle)",
            ]
    else:
        a += ['    valid=row[:,None]<nbatch']
    if packed_output_transpose:
        a += [f'    gbase=(row//span)*({n+1}*span)+row%span']
    if strided_input:
        if mode == "outer":
            a += [f'    outer_row=pid//groups+tl.full(({pack},),0,tl.int32)',
                  f'    span_row=(pid%groups)*{pack}+slot']
        a += ['    valid_row=(outer_row<(nbatch//span))&(span_row<span)',
              '    valid=valid_row[:,None]']
    if four_step:
        a += [f'    valid=row[:,None]<{four_step_other}',
              f'    matrix_base=tl.program_id(1)*{n*four_step_other}']
    if mode=='outer' or ragged_r2c:
        a += [f'    valid=valid&(((pid%groups)*{pack}+slot[:,None])<span)']
    for j in range(radix):
        a+=[f'    pos=lane+{j*lane_width}']
        if frequency_bands:
            a += _emit_band_input(n, frequency_bands, mode, sign, j)
        elif four_step:
            # Read adjacent columns together before routing each FFT to a warp.
            a += [f'    position=(tl.arange(0,64)[:,None]+{j*64})*{four_step_other}+row[None,:]']
            if mode=='four_step_real_row':
                a += ['    offset=tl.program_id(1)*input_distance+position',
                      f'    x{j}r=tl.trans(tl.load(inp+offset,row[None,:]<{four_step_other},0.0))',
                      f'    x{j}i=tl.full(({pack},64),0.0,tl.float64)' if wide else
                      f'    x{j}i=tl.full(({pack},64),0.0,tl.float32)']
                continue
            if mode=='four_step_hermitian_row':
                a += [f'    compact=tl.minimum(position,{n*four_step_other}-position)',
                      '    offset=tl.program_id(1)*input_distance+compact']
            else:a += ['    offset=matrix_base+position']
            if wide:
                a += [f'    x{j}r=tl.trans(tl.load(inp+offset*2,row[None,:]<{four_step_other},0.0))',
                      f'    x{j}i=tl.trans(tl.load(inp+offset*2+1,row[None,:]<{four_step_other},0.0))']
            else:a += [
                  '    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  f'    pair=tl.load(ip+offset,row[None,:]<{four_step_other},0)',
                  '    pair=tl.trans(pair)',
                  f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
            if mode=='four_step_hermitian_row':
                a += [f'    x{j}i=tl.where(tl.trans(position)<={n*four_step_other//2},x{j}i,-x{j}i)',
                      f'    x{j}i=tl.where((tl.trans(position)==0)|(tl.trans(position)=={n*four_step_other//2}),0.0,x{j}i)']
        elif mode=='packed_c2r':
            if wide:
                a += ['    offset=row[:,None]*input_distance+pos',
                      '    xr=tl.load(inp+offset*2,valid,0.0)',
                      '    xi=tl.load(inp+offset*2+1,valid,0.0)',
                      f'    offset=row[:,None]*input_distance+{n}-pos',
                      '    qr=tl.load(inp+offset*2,valid,0.0)',
                      '    qi=-tl.load(inp+offset*2+1,valid,0.0)',
                      '    xi=tl.where(pos==0,0.0,xi)', '    qi=tl.where(pos==0,0.0,qi)',
                      '    wr=tl.load(packed_twiddle_ptr+pos*2)',
                      '    wi=tl.load(packed_twiddle_ptr+pos*2+1)',
                      '    dr=xr-qr', '    di=xi-qi',
                      f'    x{j}r=xr+qr-(di*wr-dr*wi)',
                      f'    x{j}i=xi+qi+(dr*wr+di*wi)']
                continue
            a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  '    pair=tl.load(ip+row[:,None]*input_distance+pos,valid,0)',
                  f'    xr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'    xi=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  f'    pair=tl.load(ip+row[:,None]*input_distance+{n}-pos,valid,0)',
                  '    qr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  '    qi=-tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  '    xi=tl.where(pos==0,0.0,xi)','    qi=tl.where(pos==0,0.0,qi)']
            if table_roots:
                a += ['    wr=tl.load(packed_twiddle_ptr+pos*2)',
                      '    wi=tl.load(packed_twiddle_ptr+pos*2+1)']
            else:a += [f'    angle=pos*{-math.pi/n:.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += ['    dr=xr-qr','    di=xi-qi',
                  f'    x{j}r=xr+qr-(di*wr-dr*wi)',
                  f'    x{j}i=xi+qi+(dr*wr+di*wi)']
        elif mode=='packed_r2c':
            if wide:
                a += ['    offset=row[:,None]*input_distance+pos*2',
                      f'    x{j}r=tl.load(inp+offset,valid,0.0)',
                      f'    x{j}i=tl.load(inp+offset+1,valid,0.0)']
                continue
            # Real distances may be odd for a custom batched descriptor.
            a += ['    if input_distance%2==0:',
                  '        ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  '        pair=tl.load(ip+row[:,None]*(input_distance//2)+pos,valid,0)',
                  f'        x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'        x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  '    else:',
                  f'        x{j}r=tl.load(inp+row[:,None]*input_distance+pos*2,valid,0.0)',
                  f'        x{j}i=tl.load(inp+row[:,None]*input_distance+pos*2+1,valid,0.0)']
        elif mode=='c2r':
            a+= [f'    hpos=tl.where(pos<={n//2},pos,{n}-pos)',
                 '    offset=row[:,None]*input_distance+hpos']
            if packed_c2r_input:
                a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                      '    pair=tl.load(ip+offset,valid,0)',
                      f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                      f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
            else:
                a += [f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                      f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)']
            a += [f'    x{j}i=tl.where(pos<={n//2},x{j}i,-x{j}i)']
            if wide:a += [f'    x{j}i=tl.where((pos==0)|(pos=={n//2}),0.0,x{j}i)']
        elif mode in {'r2c','contiguous_r2c'}:
            a+=[f'    x{j}r=tl.load(inp+row[:,None]*input_distance+pos,valid,0.0)',
                f'    x{j}i=tl.full(({pack},{lane_width}),0.0,tl.float64)' if wide else
                f'    x{j}i=tl.full(({pack},{lane_width}),0.0,tl.float32)']
        elif strided_input:
            # Load neighboring independent rows together, then transpose
            # into the wave FFT layout. Odd spans and tiles crossing a
            # matrix boundary retain their exact, unhinted addresses.
            a += [f'    offset=outer_row[None,:]*({n}*span)+span_row[None,:]+tl.trans(pos)*span']
            if wide:
                a += [f'    x{j}r=tl.trans(tl.load(inp+offset*2,tl.trans(valid),0.0))',
                      f'    x{j}i=tl.trans(tl.load(inp+offset*2+1,tl.trans(valid),0.0))']
            else:
                a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                      '    pair=tl.trans(tl.load(ip+offset,tl.trans(valid),0))',
                      f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                      f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
        else:
            alignment = math.gcd(n + input_padding, lane_width)
            a+=['    offset=row[:,None]*input_distance+pos',
                 f'    offset=tl.max_contiguous(tl.multiple_of(offset,(1,{alignment})),(1,{lane_width}))']
            if wide:
                a += [f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                      f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)']
            else:a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                 '    pair=tl.load(ip+offset,valid,0)',
                 f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                 f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
    if radix==4:
        for c in 'ri': a += [f'    a{c}=x0{c}+x2{c}',f'    b{c}=x1{c}+x3{c}',f'    c{c}=x0{c}-x2{c}',f'    d{c}=x1{c}-x3{c}']
        a+= ['    r0=ar+br','    i0=ai+bi','    r2=ar-br','    i2=ai-bi',
             f'    r1=cr-({sign})*di',f'    i1=ci+({sign})*dr',
             f'    r3=cr+({sign})*di',f'    i3=ci-({sign})*dr']
    elif local_codelet and radix in {16,32}:
        from .kernels_leaf import _emit_radix16_codelet_call, _emit_natural_order_radix32_codelet_call
        for j in range(radix): a += [f'    r{j}=x{j}r',f'    i{j}=x{j}i']
        emit = _emit_radix16_codelet_call if radix==16 else _emit_natural_order_radix32_codelet_call
        a += emit('    ','inverse' if inverse else 'forward')
    else:
        bits=radix.bit_length()-1
        for j in range(radix):
            rev=int(f'{j:0{bits}b}'[::-1],2)
            a += [f'    r{j}=x{rev}r',f'    i{j}=x{rev}i']
        for stage in range(bits):
            half=1<<stage
            for group in range(0,radix,half*2):
                for k in range(half):
                    lo,hi=group+k,group+k+half
                    wr=math.cos(sign*math.pi*k/half)
                    wi=math.sin(sign*math.pi*k/half)
                    if k==0:
                        a += [f'    tr=r{hi}',f'    ti=i{hi}']
                    elif k*2==half:
                        a += [f'    tr={-sign}.0*i{hi}',f'    ti={sign}.0*r{hi}']
                    else:
                        a += [f'    tr=r{hi}*{wr:.17g}-i{hi}*{wi:.17g}',
                              f'    ti=r{hi}*{wi:.17g}+i{hi}*{wr:.17g}']
                    a += [f'    ar=r{lo}',f'    ai=i{lo}',f'    r{lo}=ar+tr',f'    i{lo}=ai+ti',
                          f'    r{hi}=ar-tr',f'    i{hi}=ai-ti']
    for j in range(1,radix):
        if root_period and j > 1 and j % root_period:
            a += ['    next_wr=wr*base_wr-wi*base_wi',
                  '    wi=wr*base_wi+wi*base_wr', '    wr=next_wr']
        elif wide or table_roots:
            a += [f'    root_index=(lane*{j})%{n}',
                  '    wr=tl.load(wave_tw_r_ptr+root_index)',
                  '    wi=tl.load(wave_tw_i_ptr+root_index)']
            if root_period and j == 1:
                a += ['    base_wr=wr', '    base_wi=wi']
        elif recurrence and j>1:
            a+=['    next_wr=wr*base_wr-wi*base_wi','    wi=wr*base_wi+wi*base_wr','    wr=next_wr']
        else:
            a+=[f'    angle=lane*{sign*2*math.pi*j/n:.17g}',
                '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            if recurrence: a+=['    base_wr=wr','    base_wi=wi']
        a+=[f'    tr=r{j}*wr-i{j}*wi',f'    ti=r{j}*wi+i{j}*wr',f'    r{j}=tr',f'    i{j}=ti']
    if dif_store:
        a += _emit_wave_dif(pack, radix, sign, lane_width)
    else:
        if lane_width==64:
            a+=['    rev=((lane&1)<<5)|((lane&2)<<3)|((lane&4)<<1)|((lane&8)>>1)|((lane&16)>>3)|((lane&32)>>5)']
        else:
            bits=lane_width.bit_length()-1
            a+=['    rev='+'|'.join(f'(((lane>>{i})&1)<<{bits-1-i})' for i in range(bits))]
        a += [f'    idx=tl.broadcast_to(rev,({pack},{lane_width}))']
        def pair_source(j):
            return f'tl.cast(tl.cast(r{j},tl.uint32,bitcast=True),tl.uint64)|(tl.cast(tl.cast(i{j},tl.uint32,bitcast=True),tl.uint64)<<32)'
        for j in range(radix):
            if packed_gather:
                a += [f'    pair={pair_source(j)}','    pair=tl.gather(pair,idx,axis=1)',
                      f'    r{j}=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                      f'    i{j}=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
            else: a += [f'    r{j}=tl.gather(r{j},idx,axis=1)',f'    i{j}=tl.gather(i{j},idx,axis=1)']
        for stage in range(lane_width.bit_length()-1):
            half=1<<stage
            a += [f'    idx=tl.broadcast_to(lane^{half},({pack},{lane_width}))',f'    high=(lane&{half})!=0']
            if half==1: a += ['    wr=1.0','    wi=0.0']
            elif half==2: a += ['    wr=tl.where((lane&1)==0,1.0,0.0)',f'    wi=tl.where((lane&1)==0,0.0,{float(sign)})']
            elif wide or table_roots:
                a += [f'    root_index=(lane&{half-1})*{n//(2*half)}',
                      '    wr=tl.load(wave_tw_r_ptr+root_index)',
                      '    wi=tl.load(wave_tw_i_ptr+root_index)']
            else: a += [f'    angle=(lane&{half-1})*{sign*math.pi/half:.17g}','    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            for j in range(radix):
                if packed_gather:
                    a += [f'    pair={pair_source(j)}','    pair=tl.gather(pair,idx,axis=1)',
                          '    pr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                          '    pi=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
                else: a += [f'    pr=tl.gather(r{j},idx,axis=1)',f'    pi=tl.gather(i{j},idx,axis=1)']
                a += [f'    br=tl.where(high,r{j},pr)',f'    bi=tl.where(high,i{j},pi)',
                      '    tr=br*wr-bi*wi','    ti=br*wi+bi*wr',
                      f'    ar=tl.where(high,pr,r{j})',f'    ai=tl.where(high,pi,i{j})',
                      f'    r{j}=tl.where(high,ar-tr,ar+tr)',f'    i{j}=tl.where(high,ai-ti,ai+ti)']
    if mode=='packed_r2c':
        a += ['    nyquist_r=r0-i0']
        for j in range(radix):
            partner=(-j)%radix
            a += [f'    idx=tl.broadcast_to((-lane-{int(j>0)})&{lane_width-1},({pack},{lane_width}))',
                  f'    br=tl.gather(r{partner},idx,1)',f'    bi=tl.gather(i{partner},idx,1)']
            if wide or table_roots:
                a += [f'    packed_k=lane*{radix}+{j}',
                      '    wr=tl.load(packed_twiddle_ptr+packed_k*2)',
                      '    wi=tl.load(packed_twiddle_ptr+packed_k*2+1)']
            else:a += [f'    angle=(lane*{radix}+{j})*{-math.pi/n:.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += [f'    dr=r{j}-br',f'    di=i{j}+bi',
                  f'    post_r{j}=0.5*(r{j}+br+wi*dr+wr*di)',
                  f'    post_i{j}=0.5*(i{j}-bi-wr*dr+wi*di)']
        for j in range(radix):a += [f'    r{j}=post_r{j}',f'    i{j}=post_i{j}']
    if mode in row_modes:
        for j in range(radix):
            if wide or table_roots:
                a += [f'    root_index=(row[:,None]*{n}+lane*{radix}+{j})*2',
                      '    wr=tl.load(twiddle_ptr+root_index,valid,0.0)',
                      '    wi=tl.load(twiddle_ptr+root_index+1,valid,0.0)']
            else:a += [f'    angle=row[:,None]*(lane*{radix}+{j})*{sign*2*math.pi/(n*four_step_other):.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += [f'    tr=r{j}*wr-i{j}*wi',f'    i{j}=r{j}*wi+i{j}*wr',f'    r{j}=tr']
    if mode in {'c2r','four_step_c2r_col'}:
        for j in range(radix): a += [f'    z{j}=r{j}']
    elif not wide:
        for j in range(radix): a += [f'    z{j}=tl.cast(tl.cast(r{j},tl.uint32,bitcast=True),tl.uint64)|(tl.cast(tl.cast(i{j},tl.uint32,bitcast=True),tl.uint64)<<32)']
    if manual_store and mode in {'outer','inner','r2c'}:
        shift=(radix*pack).bit_length()-1
        a += [f'    shared=tle.gpu.alloc([{n*pack}],dtype=tl.uint64,layout=None,scope=tle.gpu.smem,nv_mma_shared_layout=False)']
        for j in range(radix):
            a += [f'    index=(lane*{radix}+{j})*{pack}+slot[:,None]',
                  f'    index=index^((index>>{shift})&15)',
                  '    index=ld.brev(ld.brev(index.to(tl.int32)))',
                  f'    tl.store(tle.gpu.local_ptr(shared,(index,)),z{j})']
        a += ['    tl.debug_barrier()',f'    flat=tl.arange(0,{n*pack})',
              f'    index=flat^((flat>>{shift})&15)',
              '    index=ld.brev(ld.brev(index.to(tl.int32)))',
              '    value=tl.load(tle.gpu.local_ptr(shared,(index,)))',
              f'    output_slot=flat%{pack}',f'    freq=flat//{pack}']
        if mode=='inner':
            a += [f'    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+output_slot',
                  '    mask=(row_base+output_slot*span)<nbatch','    stride=nbatch']
        elif mode=='outer':
            a += [f'    gbase=(pid//groups)*({n}*span)+(pid%groups)*{pack}+output_slot',
                  f'    mask=((row_base+output_slot)<nbatch)&((pid%groups)*{pack}+output_slot<span)',
                  '    stride=span']
        else:
            a += [f'    gbase=(row_base//span)*({n//2+1}*span)+(row_base%span)+output_slot',
                  f'    mask=((row_base+output_slot)<nbatch)&(freq<{n//2+1})','    stride=span']
        a += ['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+freq*stride+gbase,value,mask)']
        return _finish_warp_source(name,a,fast_math,radix if local_codelet else 0,True)
    def join(indices, component='z'):
        if len(indices)==1: return f'{component}{indices[0]}'
        name=('j_' if component=='z' else f'j{component}_')+'_'.join(map(str,indices))
        lhs,rhs=join(indices[::2],component),join(indices[1::2],component)
        a.append(f'    {name}=tl.join({lhs},{rhs})')
        return name
    for start in range(0,radix,chunk):
        indices=list(range(start,start+chunk))
        wide_complex=wide and mode not in {'c2r','four_step_c2r_col'}
        if wide_complex:
            zr,zi=join(indices,'r'),join(indices,'i')
            a += [f'    zr=tl.reshape({zr},({pack},{lane_width*chunk}))',
                  f'    zi=tl.reshape({zi},({pack},{lane_width*chunk}))']
        else:
            values=join(indices)
            a += [f'    z=tl.reshape({values},({pack},{lane_width*chunk}))']
        a += [f'    freq=tl.arange(0,{lane_width*chunk})']
        if dif_store:
            a += [f"    out_lane = freq // {chunk}",
                  f"    out_rev = {_reverse_lane_bits('out_lane', lane_width.bit_length() - 1)}",
                  f"    freq = out_rev * {radix} + freq % {chunk} + {start}"]
        elif chunk!=radix: a += [f'    freq=(freq//{chunk})*{radix}+freq%{chunk}+{start}']
        if frequency_bands:
            length = n * frequency_bands
            a += [f"    target=freq[None,:]*{frequency_bands}+band"]
            if mode == "c2r":
                a += ["    tl.store(out+target,z)"]
            elif mode == "contiguous_r2c":
                a += [
                    f"    negative=target>{length//2}",
                    f"    independent=(band!=0)&(band!={frequency_bands//2})",
                    "    imag_sign=tl.full((),0x8000000000000000,tl.uint64)",
                    "    z=tl.where(negative,z^imag_sign,z)",
                    f"    target=tl.where(negative,{length}-target,target)",
                    f"    z=tl.where((target==0)|(target=={length//2}),z&0xffffffff,z)",
                    "    out_pairs=tl.cast(out,tl.pointer_type(tl.uint64))",
                    "    tl.store(out_pairs+target,z,~negative|independent)",
                ]
            else:
                a += ["    out_pairs=tl.cast(out,tl.pointer_type(tl.uint64))",
                      "    tl.store(out_pairs+target,z)"]
        elif four_step:
            a += ['    ptr=out' if wide or mode=='four_step_c2r_col' else '    ptr=tl.cast(out,tl.pointer_type(tl.uint64))']
            if mode in row_modes:
                a += [f'    address=matrix_base+row[:,None]*{n}+freq[None,:]']
                if wide_complex:
                    a += ['    z=tl.join(zr,zi)',
                          f'    tl.store(ptr+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,(row[:,None]<{four_step_other})[:,:,None])']
                else:a += [f'    tl.store(ptr+address,z,row[:,None]<{four_step_other})']
            else:
                a += [f'    position=freq[:,None]*{four_step_other}+row[None,:]',
                      '    base=tl.program_id(1)*output_distance' if mode!='four_step_col' else '    base=matrix_base',
                      '    address=base+position',f'    mask=row[None,:]<{four_step_other}']
                if mode=='four_step_r2c_col':a += [f'    mask=mask&(position<{n*four_step_other//2+1})']
                if wide_complex:
                    a += ['    z=tl.join(tl.trans(zr),tl.trans(zi))',
                          '    tl.store(ptr+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
                else:a += ['    tl.store(ptr+address,tl.trans(z),mask)']
        elif mode in {'contiguous','c2r','contiguous_r2c','packed_r2c','packed_c2r'} and not packed_output_transpose:
            a+=['    address=row[:,None]*output_distance+freq[None,:]',
                 '    mask=row[:,None]<nbatch']
            if mode=='contiguous_r2c': a += [f'    mask=mask&(freq[None,:]<{n//2+1})']
            if mode=='c2r': a+=['    tl.store(out+address,z,mask)']
            elif wide_complex:
                a += ['    z=tl.join(zr,zi)',
                      '    tl.store(out+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
            else: a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,z,mask)']
            if mode=='packed_r2c' and start==0:
                a += [f'    pos=lane*{radix}',f'    address=row[:,None]*output_distance+{n}',
                      f'    address=tl.broadcast_to(address,({pack},{lane_width}))']
                if wide:
                    a += ['    tl.store(out+address*2,nyquist_r,valid&(pos==0))',
                          '    tl.store(out+address*2+1,0.0,valid&(pos==0))']
                else:a += [
                      '    nyquist=tl.cast(tl.cast(nyquist_r,tl.uint32,bitcast=True),tl.uint64)',
                      '    tl.store(ptr+address,nyquist,valid&(pos==0))']
        else:
            if packed_output_transpose:
                # A CTA may cross an odd-sized matrix boundary. Its row IDs
                # still map separately; only aligned full runs get a hint.
                a += [f'    if span%{pack}==0:',
                      f'        gbase=tl.max_contiguous(tl.multiple_of(gbase,{pack}),{pack})']
            elif not strided_input and not ragged_r2c:
                a+=['    gbase=tl.max_contiguous(tl.multiple_of(gbase,'+str(pack)+'),'+str(pack)+')']
            elif strided_output_major:
                a += [f'    if (nbatch//span)%{pack}==0:',
                      f'        gbase=tl.max_contiguous(tl.multiple_of(gbase,{pack}),{pack})']
            a+=['    stride=nbatch' if mode=='inner' else f'    stride=span+{output_padding}',
                 '    address=freq[:,None]*stride+gbase[None,:]',
                 '    mask=valid_row[None,:]' if strided_input else '    mask=(row[None,:]<nbatch)']
            if mode=='outer' or ragged_r2c:
                a += [f'    mask=mask&(((pid%groups)*{pack}+slot[None,:])<span)']
            if mode=='r2c': a+=[f'    mask=mask&(freq[:,None]<{n//2+1})']
            if wide_complex:
                a += ['    z=tl.join(tl.trans(zr),tl.trans(zi))',
                      '    tl.store(out+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
            else:a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,tl.trans(z),mask)']
            if packed_output_transpose and start == 0:
                a += [f'    pos=lane*{radix}',
                      f'    nyquist_address=tl.broadcast_to(gbase[:,None]+{n}*span,({pack},{lane_width}))']
                if wide:
                    a += ['    tl.store(out+nyquist_address*2,nyquist_r,valid&(pos==0))',
                          '    tl.store(out+nyquist_address*2+1,0.0,valid&(pos==0))']
                else:
                    a += ['    nyquist=tl.cast(tl.cast(nyquist_r,tl.uint32,bitcast=True),tl.uint64)',
                          '    tl.store(ptr+nyquist_address,nyquist,valid&(pos==0))']
    return _finish_warp_source(name,a,fast_math,radix if local_codelet else 0,False)


def _finish_warp_source(name, lines, fast_math, codelet_radix, manual_store):
    source='\n'.join(lines)+'\n'
    if fast_math:
        source=source.replace('import triton.language as tl',
                              'import triton.language as tl\nfrom triton.language.extra import libdevice as ld')
        source=source.replace('tl.cos(', 'ld.fast_cosf(').replace('tl.sin(', 'ld.fast_sinf(')
    if codelet_radix in {16,32}:
        from .metadata import _module_source
        source=_module_source(source,(16,))
    if manual_store:
        source=('import triton.experimental.tle.language as tle\n'
                'from triton.language.extra import libdevice as ld\n')+source
    return name,source


def build_ix_warp256_leaf_source(plan, io_mode: str, perm_form: str, params: list[str],
                                 pack: int, permutation_span: int = 0) -> tuple[str,str]:
    """Retain the raw-leaf argument list, including unused twiddle pointers."""
    input_padding = 16 if perm_form.endswith("_padinput16") else 0
    output_padding = 16 if perm_form.endswith("_padoutput16") else 0
    perm_form = perm_form.removesuffix("_padinput16").removesuffix("_padoutput16")
    packed_c2r_input = perm_form == "outer_c2r_chunk16"
    square_chunk = packed_c2r_input or perm_form == "outer_strided_chunk16"
    single_output = perm_form.endswith("_dif_chunk1")
    four_outputs = perm_form.endswith("_dif_chunk4")
    perm_form = perm_form.removesuffix("_chunk1").removesuffix("_chunk4")
    dif_store = perm_form.endswith("_dif")
    strided_columns = perm_form in {"outer_strided", "outer_strided_chunk16"}
    mode = {
        "contiguous":"contiguous", "contiguous_c2r":"c2r",
        "contiguous_r2c":"contiguous_r2c",
        "packed_r2c":"packed_r2c", "packed_c2r":"packed_c2r",
        "permuted_r2c":"r2c", "permuted_store_r2c":"r2c",
    }.get(io_mode, "outer" if strided_columns else perm_form.removesuffix("_dif"))
    n=plan.length
    chunk=(2 if n==256 else 8 if n>=1024 else 0) if mode in {'outer','inner','r2c'} else 0
    if single_output:
        chunk = 1
    elif four_outputs:
        chunk = 4
    if square_chunk:
        chunk = 16
    if mode in {'packed_r2c','packed_c2r'}:chunk=1
    ragged_r2c = mode == "r2c" and permutation_span % pack != 0
    packed_output_transpose = mode == "packed_r2c" and perm_form == "outer_packed"
    cube_subwave = n == 256 and dif_store and pack == 32
    name,source=build_ix_warp_source(n,pack,mode,plan.direction=='inverse',fast_math=True,
                                    output_radix_chunk=chunk, dif_store=dif_store, ragged_r2c=ragged_r2c,
                                    strided_input=strided_columns,
                                    packed_c2r_input=packed_c2r_input,
                                    packed_output_transpose=packed_output_transpose,
                                    lane_width=32 if cube_subwave else 64, recurrence=cube_subwave,
                                    input_padding=input_padding, output_padding=output_padding)
    leaf_name='ix_'+name+'_leaf'
    aliases=['    inp=in_ptr','    out=out_ptr']
    if permutation_span:
        aliases.append(f'    span=tl.full((),{permutation_span},tl.int64)')
    else:
        aliases.append('    span=perm_span' if 'perm_span' in params else '    span=tl.full((),1,tl.int64)')
    if 'input_distance' not in params:
        aliases.append(f'    input_distance=tl.full((),{n+input_padding},tl.int64)')
    if 'output_distance' not in params:
        aliases.append(f'    output_distance=tl.full((),{n//2+1 if mode in {"r2c","contiguous_r2c"} else n+1 if mode=="packed_r2c" else n},tl.int64)')
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    assert header in source
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+'\n'.join(aliases),1)
    return leaf_name,source


def build_ix_warp_four_step_source(plan, io_mode, n1, n2, params, pack):
    row=io_mode in {'four_step_row','four_step_real_row','four_step_hermitian_row'}
    other=n2 if row else n1
    name,source=build_ix_warp_source(plan.length,pack,io_mode,
        plan.direction=='inverse',fast_math=True,four_step_other=other)
    leaf_name='ix_'+name+'_leaf'
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    aliases='    inp=in_ptr\n    out=out_ptr'
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+aliases,1)
    return leaf_name,source
