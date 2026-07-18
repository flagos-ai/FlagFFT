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
def _fwd_rad6_b1(r0, r1, r2, r3, r4, r5, i0, i1, i2, i3, i4, i5):
    tr0 = r0 + r2 + r4
    tr2 = (r0 - 0.5 * (r2 + r4)) + 0.8660254037844386 * (i2 - i4)
    tr4 = (r0 - 0.5 * (r2 + r4)) - 0.8660254037844386 * (i2 - i4)

    ti0 = i0 + i2 + i4
    ti2 = (i0 - 0.5 * (i2 + i4)) - 0.8660254037844386 * (r2 - r4)
    ti4 = (i0 - 0.5 * (i2 + i4)) + 0.8660254037844386 * (r2 - r4)

    tr1 = r1 + r3 + r5
    tr3 = (r1 - 0.5 * (r3 + r5)) + 0.8660254037844386 * (i3 - i5)
    tr5 = (r1 - 0.5 * (r3 + r5)) - 0.8660254037844386 * (i3 - i5)

    ti1 = i1 + i3 + i5
    ti3 = (i1 - 0.5 * (i3 + i5)) - 0.8660254037844386 * (r3 - r5)
    ti5 = (i1 - 0.5 * (i3 + i5)) + 0.8660254037844386 * (r3 - r5)

    y0r = tr0 + tr1
    y1r = tr2 + (0.5 * tr3 + 0.8660254037844386 * ti3)
    y2r = tr4 + (-0.5 * tr5 + 0.8660254037844386 * ti5)
    y3r = tr0 - tr1
    y4r = tr2 - (0.5 * tr3 + 0.8660254037844386 * ti3)
    y5r = tr4 - (-0.5 * tr5 + 0.8660254037844386 * ti5)

    y0i = ti0 + ti1
    y1i = ti2 + (-0.8660254037844386 * tr3 + 0.5 * ti3)
    y2i = ti4 + (-0.8660254037844386 * tr5 - 0.5 * ti5)
    y3i = ti0 - ti1
    y4i = ti2 - (-0.8660254037844386 * tr3 + 0.5 * ti3)
    y5i = ti4 - (-0.8660254037844386 * tr5 - 0.5 * ti5)

    return y0r, y1r, y2r, y3r, y4r, y5r, y0i, y1i, y2i, y3i, y4i, y5i
