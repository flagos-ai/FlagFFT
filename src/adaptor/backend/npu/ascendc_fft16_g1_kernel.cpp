// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#include "ascendc_fft_small_impl.hpp"

extern "C" __global__ __aicore__ void flagfft_npu_fft16_g1(GM_ADDR input,
                                                           GM_ADDR output,
                                                           GM_ADDR indices,
                                                           GM_ADDR twiddles,
                                                           uint32_t block_count,
                                                           uint32_t stride,
                                                           uint32_t mode) {
  FftSmallAiv<16, 4, 1> op;
  op.Init(input, output, indices, twiddles, block_count, stride, mode);
  op.Process();
}
