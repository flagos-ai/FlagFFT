// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ascendc_fft_radix4_pair_impl.hpp"

#define FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(name, length, group)     \
  extern "C" __global__ __aicore__ void name(GM_ADDR input,        \
                                             GM_ADDR output,       \
                                             GM_ADDR indices,      \
                                             GM_ADDR twiddles,     \
                                             uint32_t block_count, \
                                             uint32_t mode) {      \
    FftRadix4PairAiv<length, group> op;                            \
    op.Init(input, output, indices, twiddles, block_count, mode);  \
    op.Process();                                                  \
  }

FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft64_radix4_pair_g4, 64, 4)
FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft64_radix4_pair_g8, 64, 8)
FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft64_radix4_pair_g1, 64, 1)
FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft128_radix4_pair_g4, 128, 4)
FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft128_radix4_pair_g8, 128, 8)
FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL(flagfft_npu_fft2048_radix4_pair_g1, 2048, 1)

#undef FLAGFFT_DEFINE_RADIX4_PAIR_KERNEL
