@triton.jit
def _fwd_rad4_b1(r0, r1, r2, r3, i0, i1, i2, i3):
    e0r = r0 + r2
    e0i = i0 + i2
    e1r = r0 - r2
    e1i = i0 - i2
    o0r = r1 + r3
    o0i = i1 + i3
    d0r = r1 - r3
    d0i = i1 - i3

    t1r = d0i
    t1i = -d0r

    y0r = e0r + o0r
    y0i = e0i + o0i
    y2r = e0r - o0r
    y2i = e0i - o0i
    y1r = e1r + t1r
    y1i = e1i + t1i
    y3r = e1r - t1r
    y3i = e1i - t1i
    return y0r, y1r, y2r, y3r, y0i, y1i, y2i, y3i
