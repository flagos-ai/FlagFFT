// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <cstdint>
namespace flagfft::adaptor::npu {
inline constexpr uint32_t kNpuStockhamTile = 128;
inline constexpr uint32_t kNpuStockhamMaxRadix = 19;
inline constexpr uint32_t kNpuStockhamRootStride = 24;
inline constexpr uint32_t kNpuStockhamIndexVectors = 5;
inline constexpr uint32_t kNpuStockhamComplexAlignment = 4;
inline constexpr uint64_t npu_stockham_batch_group(uint64_t extent) {
  return extent > 0 && extent < kNpuStockhamTile && extent % kNpuStockhamComplexAlignment == 0
      ? kNpuStockhamTile / extent : 1;
}
}
