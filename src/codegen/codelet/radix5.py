@triton.jit
def _fwd_rad5_b1(r0, r1, r2, r3, r4, i0, i1, i2, i3, i4):
    c5qa = 0.30901699437494745
    c5qb = 0.9510565162951535
    c5qc = 0.5
    c5qd = 0.5877852522924732

    y0r = r0 + r1 + r2 + r3 + r4
    y1r = (r0 - c5qc * (r2 + r3)) + c5qb * (i1 - i4) + c5qd * (i2 - i3) + c5qa * ((r1 - r2) + (r4 - r3))
    y4r = (r0 - c5qc * (r2 + r3)) - c5qb * (i1 - i4) - c5qd * (i2 - i3) + c5qa * ((r1 - r2) + (r4 - r3))
    y2r = (r0 - c5qc * (r1 + r4)) - c5qb * (i2 - i3) + c5qd * (i1 - i4) + c5qa * ((r2 - r1) + (r3 - r4))
    y3r = (r0 - c5qc * (r1 + r4)) + c5qb * (i2 - i3) - c5qd * (i1 - i4) + c5qa * ((r2 - r1) + (r3 - r4))

    y0i = i0 + i1 + i2 + i3 + i4
    y1i = (i0 - c5qc * (i2 + i3)) - c5qb * (r1 - r4) - c5qd * (r2 - r3) + c5qa * ((i1 - i2) + (i4 - i3))
    y4i = (i0 - c5qc * (i2 + i3)) + c5qb * (r1 - r4) + c5qd * (r2 - r3) + c5qa * ((i1 - i2) + (i4 - i3))
    y2i = (i0 - c5qc * (i1 + i4)) + c5qb * (r2 - r3) - c5qd * (r1 - r4) + c5qa * ((i2 - i1) + (i3 - i4))
    y3i = (i0 - c5qc * (i1 + i4)) - c5qb * (r2 - r3) + c5qd * (r1 - r4) + c5qa * ((i2 - i1) + (i3 - i4))

    return y0r, y1r, y2r, y3r, y4r, y0i, y1i, y2i, y3i, y4i
