@triton.jit
def _fwd_rad8_b1(r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7):

    e02r = r0 + r4
    e02i = i0 + i4
    d02r = r0 - r4
    d02i = i0 - i4
    e13r = r2 + r6
    e13i = i2 + i6
    d13r = r2 - r6
    d13i = i2 - i6

    E0r = e02r + e13r
    E0i = e02i + e13i
    E2r = e02r - e13r
    E2i = e02i - e13i
    E1r = d02r + d13i
    E1i = d02i - d13r
    E3r = d02r - d13i
    E3i = d02i + d13r

    o02r = r1 + r5
    o02i = i1 + i5
    od02r = r1 - r5
    od02i = i1 - i5
    o13r = r3 + r7
    o13i = i3 + i7
    od13r = r3 - r7
    od13i = i3 - i7

    O0r = o02r + o13r
    O0i = o02i + o13i
    O2r = o02r - o13r
    O2i = o02i - o13i
    O1r = od02r + od13i
    O1i = od02i - od13r
    O3r = od02r - od13i
    O3i = od02i + od13r

    T0r = O0r
    T0i = O0i
    T1r, T1i = _cmul(O1r, O1i, 0.7071067811865475, -0.7071067811865475)
    T2r, T2i = _cmul(O2r, O2i, 0.0, -1.0)
    T3r, T3i = _cmul(O3r, O3i, -0.7071067811865475, -0.7071067811865475)

    y0r = E0r + T0r
    y0i = E0i + T0i
    y4r = E0r - T0r
    y4i = E0i - T0i
    y1r = E1r + T1r
    y1i = E1i + T1i
    y5r = E1r - T1r
    y5i = E1i - T1i
    y2r = E2r + T2r
    y2i = E2i + T2i
    y6r = E2r - T2r
    y6i = E2i - T2i
    y3r = E3r + T3r
    y3i = E3i + T3i
    y7r = E3r - T3r
    y7i = E3i - T3i

    return (
        y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r,
        y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i,
    )