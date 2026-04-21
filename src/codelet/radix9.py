@triton.jit
def _fwd_rad9_b1(r0, r1, r2, r3, r4, r5, r6, r7, r8, i0, i1, i2, i3, i4, i5, i6, i7, i8):
    c9qa = 0.7660444431189771
    c9qb = 0.6427876096865395
    c9qc = 0.17364817766693127
    c9qd = 0.9848077530122085
    c9qe = 0.5
    c9qf = 0.8660254037844373
    c9qg = 0.9396926207859084
    c9qh = 0.3420201433256684

    v0r = r1 + r8
    v0i = i1 + i8
    v1r = r2 + r7
    v1i = i2 + i7
    v2r = r3 + r6
    v2i = i3 + i6

    p0r = r1 - r8
    p0i = i1 - i8
    p1r = r2 - r7
    p1i = i2 - i7
    p2r = (r3 - r6) * c9qf
    p2i = (i3 - i6) * c9qf
    d45r = r4 - r5
    d45i = i4 - i5
    s45r = r4 + r5
    s45i = i4 + i5

    t8r = (c9qb * p0r) + (c9qd * p1r) + p2r + (c9qh * d45r)
    t8i = (c9qb * p0i) + (c9qd * p1i) + p2i + (c9qh * d45i)
    b1r = r0 + (c9qa * v0r) + (c9qc * v1r) - (c9qe * v2r) - (c9qg * s45r)
    b1i = i0 + (c9qa * v0i) + (c9qc * v1i) - (c9qe * v2i) - (c9qg * s45i)
    y1r = b1r + t8i
    y1i = b1i - t8r
    y8r = b1r - t8i
    y8i = b1i + t8r

    t7r = -(c9qb * d45r) + (c9qd * p0r) - p2r + (c9qh * p1r)
    t7i = -(c9qb * d45i) + (c9qd * p0i) - p2i + (c9qh * p1i)
    b2r = r0 + (c9qa * s45r) + (c9qc * v0r) - (c9qe * v2r) - (c9qg * v1r)
    b2i = i0 + (c9qa * s45i) + (c9qc * v0i) - (c9qe * v2i) - (c9qg * v1i)
    y2r = b2r + t7i
    y2i = b2i - t7r
    y7r = b2r - t7i
    y7i = b2i + t7r

    t6r = c9qf * (p0r + d45r - p1r)
    t6i = c9qf * (p0i + d45i - p1i)
    b3r = r0 + v2r - (c9qe * (v0r + v1r + s45r))
    b3i = i0 + v2i - (c9qe * (v0i + v1i + s45i))
    y3r = b3r + t6i
    y3i = b3i - t6r
    y6r = b3r - t6i
    y6i = b3i + t6r

    t0r = -(c9qb * p1r) - (c9qd * d45r) + p2r + (c9qh * p0r)
    t0i = -(c9qb * p1i) - (c9qd * d45i) + p2i + (c9qh * p0i)
    y0r = r0 + v0r + v1r + v2r + s45r
    y0i = i0 + v0i + v1i + v2i + s45i
    b4r = r0 + (c9qa * v1r) + (c9qc * s45r) - (c9qe * v2r) - (c9qg * v0r)
    b4i = i0 + (c9qa * v1i) + (c9qc * s45i) - (c9qe * v2i) - (c9qg * v0i)
    y4r = b4r + t0i
    y4i = b4i - t0r
    y5r = b4r - t0i
    y5i = b4i + t0r

    return (
        y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r, y8r,
        y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i, y8i,
    )
