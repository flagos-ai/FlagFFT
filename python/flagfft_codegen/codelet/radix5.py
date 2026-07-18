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
def _fwd_rad5_b1(r0, r1, r2, r3, r4, i0, i1, i2, i3, i4):
    y0r = r0 + r1 + r2 + r3 + r4
    y1r = (
        (r0 - 0.5 * (r2 + r3))
        + 0.9510565162951535 * (i1 - i4)
        + 0.5877852522924732 * (i2 - i3)
        + 0.30901699437494745 * ((r1 - r2) + (r4 - r3))
    )
    y4r = (
        (r0 - 0.5 * (r2 + r3))
        - 0.9510565162951535 * (i1 - i4)
        - 0.5877852522924732 * (i2 - i3)
        + 0.30901699437494745 * ((r1 - r2) + (r4 - r3))
    )
    y2r = (
        (r0 - 0.5 * (r1 + r4))
        - 0.9510565162951535 * (i2 - i3)
        + 0.5877852522924732 * (i1 - i4)
        + 0.30901699437494745 * ((r2 - r1) + (r3 - r4))
    )
    y3r = (
        (r0 - 0.5 * (r1 + r4))
        + 0.9510565162951535 * (i2 - i3)
        - 0.5877852522924732 * (i1 - i4)
        + 0.30901699437494745 * ((r2 - r1) + (r3 - r4))
    )

    y0i = i0 + i1 + i2 + i3 + i4
    y1i = (
        (i0 - 0.5 * (i2 + i3))
        - 0.9510565162951535 * (r1 - r4)
        - 0.5877852522924732 * (r2 - r3)
        + 0.30901699437494745 * ((i1 - i2) + (i4 - i3))
    )
    y4i = (
        (i0 - 0.5 * (i2 + i3))
        + 0.9510565162951535 * (r1 - r4)
        + 0.5877852522924732 * (r2 - r3)
        + 0.30901699437494745 * ((i1 - i2) + (i4 - i3))
    )
    y2i = (
        (i0 - 0.5 * (i1 + i4))
        + 0.9510565162951535 * (r2 - r3)
        - 0.5877852522924732 * (r1 - r4)
        + 0.30901699437494745 * ((i2 - i1) + (i3 - i4))
    )
    y3i = (
        (i0 - 0.5 * (i1 + i4))
        - 0.9510565162951535 * (r2 - r3)
        + 0.5877852522924732 * (r1 - r4)
        + 0.30901699437494745 * ((i2 - i1) + (i3 - i4))
    )

    return y0r, y1r, y2r, y3r, y4r, y0i, y1i, y2i, y3i, y4i
