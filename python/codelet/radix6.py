@triton.jit
def _fwd_rad6_b1(r0, r1, r2, r3, r4, r5, i0, i1, i2, i3, i4, i5):
    c3qa = 0.5
    c3qb = 0.8660254037844386

    tr0 = r0 + r2 + r4
    tr2 = (r0 - c3qa * (r2 + r4)) + c3qb * (i2 - i4)
    tr4 = (r0 - c3qa * (r2 + r4)) - c3qb * (i2 - i4)

    ti0 = i0 + i2 + i4
    ti2 = (i0 - c3qa * (i2 + i4)) - c3qb * (r2 - r4)
    ti4 = (i0 - c3qa * (i2 + i4)) + c3qb * (r2 - r4)

    tr1 = r1 + r3 + r5
    tr3 = (r1 - c3qa * (r3 + r5)) + c3qb * (i3 - i5)
    tr5 = (r1 - c3qa * (r3 + r5)) - c3qb * (i3 - i5)

    ti1 = i1 + i3 + i5
    ti3 = (i1 - c3qa * (i3 + i5)) - c3qb * (r3 - r5)
    ti5 = (i1 - c3qa * (i3 + i5)) + c3qb * (r3 - r5)

    y0r = tr0 + tr1
    y1r = tr2 + (c3qa * tr3 + c3qb * ti3)
    y2r = tr4 + (-c3qa * tr5 + c3qb * ti5)
    y3r = tr0 - tr1
    y4r = tr2 - (c3qa * tr3 + c3qb * ti3)
    y5r = tr4 - (-c3qa * tr5 + c3qb * ti5)

    y0i = ti0 + ti1
    y1i = ti2 + (-c3qb * tr3 + c3qa * ti3)
    y2i = ti4 + (-c3qb * tr5 - c3qa * ti5)
    y3i = ti0 - ti1
    y4i = ti2 - (-c3qb * tr3 + c3qa * ti3)
    y5i = ti4 - (-c3qb * tr5 - c3qa * ti5)

    return y0r, y1r, y2r, y3r, y4r, y5r, y0i, y1i, y2i, y3i, y4i, y5i
