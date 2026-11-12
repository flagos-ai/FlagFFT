"""Explicit-layout IX FFT screening kernels; no production dispatch."""
import math


def build_ix_gluon_source(n: int, warps: int = 8, inverse: bool = False,
                          mode: str = "contiguous") -> tuple[str, str]:
    if n not in {1024, 2048, 8192, 16384} or warps not in {4, 8, 16}:
        raise ValueError("unsupported IX Gluon FFT geometry")
    if mode not in {"contiguous", "packed_c2r"} or (mode == "packed_c2r" and not inverse):
        raise ValueError("unsupported IX Gluon FFT boundary")
    rows, cols = n // 64, 64
    sign = 1 if inverse else -1
    name = f"ix_gluon_{n}_w{warps}_{mode}_{'inv' if inverse else 'fwd'}"
    a = ["import triton", "from triton.experimental import gluon as g",
         "from triton.experimental.gluon import language as gl",
         "from triton.experimental.gluon.language import BlockedLayout, SliceLayout",
         "from triton.experimental.gluon.language.extra import libdevice as ld",
         "@g.jit", f"def {name}(inp, out, input_distance, output_distance, nbatch):",
         "    batch = gl.program_id(0)", "    if batch >= nbatch:", "        return",
         f"    lc: gl.constexpr = BlockedLayout([1,1], [1,64], [{warps},1], [1,0])",
         f"    lr: gl.constexpr = BlockedLayout([1,1], [64,1], [1,{warps}], [0,1])",
         f"    rr = gl.arange(0, {rows}, layout=SliceLayout(1,lc))",
         f"    cc = gl.arange(0, {cols}, layout=SliceLayout(0,lc))"]
    def reverse(var, size):
        bits = size.bit_length() - 1
        return " | ".join(f"((({var} >> {i}) & 1) << {bits-1-i})" for i in range(bits))
    a += [f"    ri = {reverse('rr', rows)}", f"    ci = {reverse('cc', cols)}",
          f"    k = ri[:,None] * {cols} + ci[None,:]",
          "    ip = gl.cast(inp, gl.pointer_type(gl.uint64))",
          "    pair = gl.load(ip + batch * input_distance + k)",
          "    xr = gl.cast(gl.cast(pair,gl.uint32),gl.float32,bitcast=True)",
          "    xi = gl.cast(gl.cast(pair >> 32,gl.uint32),gl.float32,bitcast=True)"]
    if mode == 'packed_c2r':
        a += [f"    pair = gl.load(ip + batch * input_distance + {n} - k)",
              "    qr = gl.cast(gl.cast(pair,gl.uint32),gl.float32,bitcast=True)",
              "    qi = -gl.cast(gl.cast(pair >> 32,gl.uint32),gl.float32,bitcast=True)",
              "    xi = gl.where(k == 0, 0.0, xi)", "    qi = gl.where(k == 0, 0.0, qi)",
              f"    angle = k * {-math.pi/n:.17g}",
              "    wr = ld.fast_cosf(angle)", "    wi = ld.fast_sinf(angle)",
              "    dr = xr - qr", "    di = xi - qi",
              "    zi = xi + qi + dr * wr + di * wi",
              "    xr = xr + qr - di * wr + dr * wi", "    xi = zi"]
    a += ["    xr = gl.convert_layout(xr,lr)", "    xi = gl.convert_layout(xi,lr)",
          f"    rr = gl.arange(0,{rows},layout=SliceLayout(1,lr))",
          f"    cc = gl.arange(0,{cols},layout=SliceLayout(0,lr))"]
    def fft_axis(axis, size, var):
        for stage in range(size.bit_length() - 1):
            half = 1 << stage
            idx = f"({var} ^ {half})" + ("[:,None]" if axis == 0 else "[None,:]")
            a.extend([f"    idx = gl.broadcast_to({idx},({rows},{cols}))",
                      f"    qr = gl.gather(xr,idx,{axis})", f"    qi = gl.gather(xi,idx,{axis})",
                      f"    low = (({var} & {half}) == 0)" + ("[:,None]" if axis == 0 else "[None,:]"),
                      "    ar = gl.where(low,xr,qr)", "    ai = gl.where(low,xi,qi)",
                      "    br = gl.where(low,qr,xr)", "    bi = gl.where(low,qi,xi)"])
            if half == 1:
                a.extend(["    tr = br", "    ti = bi"])
            else:
                a.extend([f"    angle = ({var} % {half}) * {sign*math.pi/half:.17g}",
                          "    wr = ld.fast_cosf(angle)" + ("[:,None]" if axis == 0 else "[None,:]"),
                          "    wi = ld.fast_sinf(angle)" + ("[:,None]" if axis == 0 else "[None,:]"),
                          "    tr = br * wr - bi * wi", "    ti = br * wi + bi * wr"])
            a.extend(["    xr = ar + gl.where(low,tr,-tr)", "    xi = ai + gl.where(low,ti,-ti)"])
    fft_axis(0, rows, "rr")
    a += ["    xr = gl.convert_layout(xr,lc)", "    xi = gl.convert_layout(xi,lc)",
          f"    rr = gl.arange(0,{rows},layout=SliceLayout(1,lc))",
          f"    cc = gl.arange(0,{cols},layout=SliceLayout(0,lc))",
          f"    ci = {reverse('cc', cols)}",
          f"    angle = rr[:,None] * ci[None,:] * {sign*2*math.pi/n:.17g}",
          "    wr = ld.fast_cosf(angle)", "    wi = ld.fast_sinf(angle)",
          "    zr = xr * wr - xi * wi", "    xi = xr * wi + xi * wr", "    xr = zr"]
    fft_axis(1, cols, "cc")
    a += ["    xr = gl.convert_layout(xr,lr)", "    xi = gl.convert_layout(xi,lr)",
          f"    rr = gl.arange(0,{rows},layout=SliceLayout(1,lr))",
          f"    cc = gl.arange(0,{cols},layout=SliceLayout(0,lr))",
          f"    k = rr[:,None] + {rows} * cc[None,:]"]
    if mode == 'packed_c2r':
        a += ["    gl.store(out + batch * output_distance + 2 * k, xr)",
              "    gl.store(out + batch * output_distance + 2 * k + 1, xi)"]
    else:
        a += ["    op = gl.cast(out, gl.pointer_type(gl.uint64))",
              "    pair = gl.cast(gl.cast(xr,gl.uint32,bitcast=True),gl.uint64) | (gl.cast(gl.cast(xi,gl.uint32,bitcast=True),gl.uint64) << 32)",
              "    gl.store(op + batch * output_distance + k, pair)"]
    return name, "\n".join(a) + "\n"
