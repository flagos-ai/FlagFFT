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

@triton.jit
def _fwd_rad11_b1(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10):
    p1r = r1 + r10
    m1r = r1 - r10
    p1i = i1 + i10
    m1i = i1 - i10
    p2r = r2 + r9
    m2r = r2 - r9
    p2i = i2 + i9
    m2i = i2 - i9
    p3r = r3 + r8
    m3r = r3 - r8
    p3i = i3 + i8
    m3i = i3 - i8
    p4r = r4 + r7
    m4r = r4 - r7
    p4i = i4 + i7
    m4i = i4 - i7
    p5r = r5 + r6
    m5r = r5 - r6
    p5i = i5 + i6
    m5i = i5 - i6
    y0r = r0 + p1r + p2r + p3r + p4r + p5r
    y0i = i0 + p1i + p2i + p3i + p4i + p5i
    c1r = r0
    c1r = c1r + 0.8412535328311812 * p1r
    s1r = 0.5406408174555976 * m1r
    c1r = c1r + 0.41541501300188644 * p2r
    s1r = s1r + 0.9096319953545183 * m2r
    c1r = c1r + -0.142314838273285 * p3r
    s1r = s1r + 0.9898214418809328 * m3r
    c1r = c1r + -0.654860733945285 * p4r
    s1r = s1r + 0.7557495743542583 * m4r
    c1r = c1r + -0.9594929736144974 * p5r
    s1r = s1r + 0.28173255684142967 * m5r
    c1i = i0
    c1i = c1i + 0.8412535328311812 * p1i
    s1i = 0.5406408174555976 * m1i
    c1i = c1i + 0.41541501300188644 * p2i
    s1i = s1i + 0.9096319953545183 * m2i
    c1i = c1i + -0.142314838273285 * p3i
    s1i = s1i + 0.9898214418809328 * m3i
    c1i = c1i + -0.654860733945285 * p4i
    s1i = s1i + 0.7557495743542583 * m4i
    c1i = c1i + -0.9594929736144974 * p5i
    s1i = s1i + 0.28173255684142967 * m5i
    y1r = c1r + s1i
    y1i = c1i - s1r
    y10r = c1r - s1i
    y10i = c1i + s1r
    c2r = r0
    c2r = c2r + 0.41541501300188644 * p1r
    s2r = 0.9096319953545183 * m1r
    c2r = c2r + -0.654860733945285 * p2r
    s2r = s2r + 0.7557495743542583 * m2r
    c2r = c2r + -0.9594929736144975 * p3r
    s2r = s2r + -0.2817325568414294 * m3r
    c2r = c2r + -0.14231483827328523 * p4r
    s2r = s2r + -0.9898214418809327 * m4r
    c2r = c2r + 0.8412535328311812 * p5r
    s2r = s2r + -0.5406408174555974 * m5r
    c2i = i0
    c2i = c2i + 0.41541501300188644 * p1i
    s2i = 0.9096319953545183 * m1i
    c2i = c2i + -0.654860733945285 * p2i
    s2i = s2i + 0.7557495743542583 * m2i
    c2i = c2i + -0.9594929736144975 * p3i
    s2i = s2i + -0.2817325568414294 * m3i
    c2i = c2i + -0.14231483827328523 * p4i
    s2i = s2i + -0.9898214418809327 * m4i
    c2i = c2i + 0.8412535328311812 * p5i
    s2i = s2i + -0.5406408174555974 * m5i
    y2r = c2r + s2i
    y2i = c2i - s2r
    y9r = c2r - s2i
    y9i = c2i + s2r
    c3r = r0
    c3r = c3r + -0.142314838273285 * p1r
    s3r = 0.9898214418809328 * m1r
    c3r = c3r + -0.9594929736144975 * p2r
    s3r = s3r + -0.2817325568414294 * m2r
    c3r = c3r + 0.41541501300188605 * p3r
    s3r = s3r + -0.9096319953545186 * m3r
    c3r = c3r + 0.8412535328311812 * p4r
    s3r = s3r + 0.5406408174555976 * m4r
    c3r = c3r + -0.654860733945285 * p5r
    s3r = s3r + 0.7557495743542583 * m5r
    c3i = i0
    c3i = c3i + -0.142314838273285 * p1i
    s3i = 0.9898214418809328 * m1i
    c3i = c3i + -0.9594929736144975 * p2i
    s3i = s3i + -0.2817325568414294 * m2i
    c3i = c3i + 0.41541501300188605 * p3i
    s3i = s3i + -0.9096319953545186 * m3i
    c3i = c3i + 0.8412535328311812 * p4i
    s3i = s3i + 0.5406408174555976 * m4i
    c3i = c3i + -0.654860733945285 * p5i
    s3i = s3i + 0.7557495743542583 * m5i
    y3r = c3r + s3i
    y3i = c3i - s3r
    y8r = c3r - s3i
    y8i = c3i + s3r
    c4r = r0
    c4r = c4r + -0.654860733945285 * p1r
    s4r = 0.7557495743542583 * m1r
    c4r = c4r + -0.14231483827328523 * p2r
    s4r = s4r + -0.9898214418809327 * m2r
    c4r = c4r + 0.8412535328311812 * p3r
    s4r = s4r + 0.5406408174555976 * m3r
    c4r = c4r + -0.9594929736144974 * p4r
    s4r = s4r + 0.28173255684142967 * m4r
    c4r = c4r + 0.41541501300188605 * p5r
    s4r = s4r + -0.9096319953545186 * m5r
    c4i = i0
    c4i = c4i + -0.654860733945285 * p1i
    s4i = 0.7557495743542583 * m1i
    c4i = c4i + -0.14231483827328523 * p2i
    s4i = s4i + -0.9898214418809327 * m2i
    c4i = c4i + 0.8412535328311812 * p3i
    s4i = s4i + 0.5406408174555976 * m3i
    c4i = c4i + -0.9594929736144974 * p4i
    s4i = s4i + 0.28173255684142967 * m4i
    c4i = c4i + 0.41541501300188605 * p5i
    s4i = s4i + -0.9096319953545186 * m5i
    y4r = c4r + s4i
    y4i = c4i - s4r
    y7r = c4r - s4i
    y7i = c4i + s4r
    c5r = r0
    c5r = c5r + -0.9594929736144974 * p1r
    s5r = 0.28173255684142967 * m1r
    c5r = c5r + 0.8412535328311812 * p2r
    s5r = s5r + -0.5406408174555974 * m2r
    c5r = c5r + -0.654860733945285 * p3r
    s5r = s5r + 0.7557495743542583 * m3r
    c5r = c5r + 0.41541501300188605 * p4r
    s5r = s5r + -0.9096319953545186 * m4r
    c5r = c5r + -0.142314838273285 * p5r
    s5r = s5r + 0.9898214418809328 * m5r
    c5i = i0
    c5i = c5i + -0.9594929736144974 * p1i
    s5i = 0.28173255684142967 * m1i
    c5i = c5i + 0.8412535328311812 * p2i
    s5i = s5i + -0.5406408174555974 * m2i
    c5i = c5i + -0.654860733945285 * p3i
    s5i = s5i + 0.7557495743542583 * m3i
    c5i = c5i + 0.41541501300188605 * p4i
    s5i = s5i + -0.9096319953545186 * m4i
    c5i = c5i + -0.142314838273285 * p5i
    s5i = s5i + 0.9898214418809328 * m5i
    y5r = c5r + s5i
    y5i = c5i - s5r
    y6r = c5r - s5i
    y6i = c5i + s5r
    return y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r, y8r, y9r, y10r, y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i, y8i, y9i, y10i
