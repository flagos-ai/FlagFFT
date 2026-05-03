@triton.jit
def _fwd_rad3_b1(r0, r1, r2, i0, i1, i2):
    c3qa = 0.5
    c3qb = 0.8660254037844386

    y0r = r0 + r1 + r2
    y1r = (r0 - c3qa * (r1 + r2)) + c3qb * (i1 - i2)
    y2r = (r0 - c3qa * (r1 + r2)) - c3qb * (i1 - i2)

    y0i = i0 + i1 + i2
    y1i = (i0 - c3qa * (i1 + i2)) - c3qb * (r1 - r2)
    y2i = (i0 - c3qa * (i1 + i2)) + c3qb * (r1 - r2)
    return y0r, y1r, y2r, y0i, y1i, y2i
