@triton.jit
def _fwd_rad16_b1(
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
):
  c8q = 0.7071067811865475
  c16a = 0.9238795325112867
  c16b = 0.38268343236508984

  i1 = i0 - i1
  r1 = r0 - r1
  i0 = 2.0 * i0 - i1
  r0 = 2.0 * r0 - r1
  i3 = i2 - i3
  r3 = r2 - r3
  i2 = 2.0 * i2 - i3
  r2 = 2.0 * r2 - r3
  i5 = i4 - i5
  r5 = r4 - r5
  i4 = 2.0 * i4 - i5
  r4 = 2.0 * r4 - r5
  i7 = i6 - i7
  r7 = r6 - r7
  i6 = 2.0 * i6 - i7
  r6 = 2.0 * r6 - r7
  i9 = i8 - i9
  r9 = r8 - r9
  i8 = 2.0 * i8 - i9
  r8 = 2.0 * r8 - r9
  i11 = i10 - i11
  r11 = r10 - r11
  i10 = 2.0 * i10 - i11
  r10 = 2.0 * r10 - r11
  i13 = i12 - i13
  r13 = r12 - r13
  i12 = 2.0 * i12 - i13
  r12 = 2.0 * r12 - r13
  i15 = i14 - i15
  r15 = r14 - r15
  i14 = 2.0 * i14 - i15
  r14 = 2.0 * r14 - r15

  i2 = i0 - i2
  r2 = r0 - r2
  i0 = 2.0 * i0 - i2
  r0 = 2.0 * r0 - r2
  t3r = r1 + (-i3)
  t3i = i1 + r3
  r1 = 2.0 * r1 - t3r
  i1 = 2.0 * i1 - t3i
  r3 = t3r
  i3 = t3i
  i6 = i4 - i6
  r6 = r4 - r6
  i4 = 2.0 * i4 - i6
  r4 = 2.0 * r4 - r6
  t7r = r5 + (-i7)
  t7i = i5 + r7
  r5 = 2.0 * r5 - t7r
  i5 = 2.0 * i5 - t7i
  r7 = t7r
  i7 = t7i
  i10 = i8 - i10
  r10 = r8 - r10
  i8 = 2.0 * i8 - i10
  r8 = 2.0 * r8 - r10
  t11r = r9 + (-i11)
  t11i = i9 + r11
  r9 = 2.0 * r9 - t11r
  i9 = 2.0 * i9 - t11i
  r11 = t11r
  i11 = t11i
  i14 = i12 - i14
  r14 = r12 - r14
  i12 = 2.0 * i12 - i14
  r12 = 2.0 * r12 - r14
  t15r = r13 + (-i15)
  t15i = i13 + r15
  r13 = 2.0 * r13 - t15r
  i13 = 2.0 * i13 - t15i
  r15 = t15r
  i15 = t15i

  i4 = i0 - i4
  r4 = r0 - r4
  i0 = 2.0 * i0 - i4
  r0 = 2.0 * r0 - r4
  t5r = (r1 - c8q * r5) - c8q * i5
  t5i = (i1 - c8q * i5) + c8q * r5
  r1 = 2.0 * r1 - t5r
  i1 = 2.0 * i1 - t5i
  r5 = t5r
  i5 = t5i
  t6r = r2 + (-i6)
  t6i = i2 + r6
  r2 = 2.0 * r2 - t6r
  i2 = 2.0 * i2 - t6i
  r6 = t6r
  i6 = t6i
  t7r = (r3 + c8q * r7) - c8q * i7
  t7i = (i3 + c8q * i7) + c8q * r7
  r3 = 2.0 * r3 - t7r
  i3 = 2.0 * i3 - t7i
  r7 = t7r
  i7 = t7i
  i12 = i8 - i12
  r12 = r8 - r12
  i8 = 2.0 * i8 - i12
  r8 = 2.0 * r8 - r12
  t13r = (r9 - c8q * r13) - c8q * i13
  t13i = (i9 - c8q * i13) + c8q * r13
  r9 = 2.0 * r9 - t13r
  i9 = 2.0 * i9 - t13i
  r13 = t13r
  i13 = t13i
  t14r = r10 + (-i14)
  t14i = i10 + r14
  r10 = 2.0 * r10 - t14r
  i10 = 2.0 * i10 - t14i
  r14 = t14r
  i14 = t14i
  t15r = (r11 + c8q * r15) - c8q * i15
  t15i = (i11 + c8q * i15) + c8q * r15
  r11 = 2.0 * r11 - t15r
  i11 = 2.0 * i11 - t15i
  r15 = t15r
  i15 = t15i

  i8 = i0 - i8
  r8 = r0 - r8
  i0 = 2.0 * i0 - i8
  r0 = 2.0 * r0 - r8
  t9r = (r1 - c16a * r9) - c16b * i9
  t9i = (i1 - c16a * i9) + c16b * r9
  r1 = 2.0 * r1 - t9r
  i1 = 2.0 * i1 - t9i
  r9 = t9r
  i9 = t9i
  t10r = (r2 - c8q * r10) - c8q * i10
  t10i = (i2 - c8q * i10) + c8q * r10
  r2 = 2.0 * r2 - t10r
  i2 = 2.0 * i2 - t10i
  r10 = t10r
  i10 = t10i
  t11r = (r3 - c16b * r11) - c16a * i11
  t11i = (i3 - c16b * i11) + c16a * r11
  r3 = 2.0 * r3 - t11r
  i3 = 2.0 * i3 - t11i
  r11 = t11r
  i11 = t11i
  t12r = r4 + (-i12)
  t12i = i4 + r12
  r4 = 2.0 * r4 - t12r
  i4 = 2.0 * i4 - t12i
  r12 = t12r
  i12 = t12i
  t13r = (r5 + c16b * r13) - c16a * i13
  t13i = (i5 + c16b * i13) + c16a * r13
  r5 = 2.0 * r5 - t13r
  i5 = 2.0 * i5 - t13i
  r13 = t13r
  i13 = t13i
  t14r = (r6 + c8q * r14) - c8q * i14
  t14i = (i6 + c8q * i14) + c8q * r14
  r6 = 2.0 * r6 - t14r
  i6 = 2.0 * i6 - t14i
  r14 = t14r
  i14 = t14i
  t15r = (r7 + c16a * r15) - c16b * i15
  t15i = (i7 + c16a * i15) + c16b * r15
  r7 = 2.0 * r7 - t15r
  i7 = 2.0 * i7 - t15i
  r15 = t15r
  i15 = t15i

  return (
      r0,
      r1,
      r2,
      r3,
      r4,
      r5,
      r6,
      r7,
      r8,
      r9,
      r10,
      r11,
      r12,
      r13,
      r14,
      r15,
      i0,
      i1,
      i2,
      i3,
      i4,
      i5,
      i6,
      i7,
      i8,
      i9,
      i10,
      i11,
      i12,
      i13,
      i14,
      i15,
  )