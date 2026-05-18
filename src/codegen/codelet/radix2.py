@triton.jit
def _fwd_rad2_b1(r0, r1, i0, i1):
    y0r = r0 + r1
    y0i = i0 + i1
    y1r = r0 - r1
    y1i = i0 - i1
    return y0r, y1r, y0i, y1i