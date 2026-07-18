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
def _fwd_rad2_b1(r0, r1, i0, i1):
    y0r = r0 + r1
    y0i = i0 + i1
    y1r = r0 - r1
    y1i = i0 - i1
    return y0r, y1r, y0i, y1i
