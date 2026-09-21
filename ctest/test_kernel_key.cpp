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

#include "flagfft/core.hpp"
#include "flagfft_test.h"

#include <string>

// The in-process kernel cache is keyed by KernelKey::repr().  A key entering
// the cache must describe every dimension that reaches the generator;
// otherwise two different kernels share one entry.  The regression that
// motivated these checks was a tiled-transpose pair whose forward and inverse
// dimensions collided because repr() omitted the stored n0/n1 fields.

TEST(KernelKeyRepr, TiledTransposeDistinguishesDimensions) {
  const auto forward = flagfft::KernelKey::tiled_transpose("corex:71:64", "complex64", 46189, 48);
  const auto inverse = flagfft::KernelKey::tiled_transpose("corex:71:64", "complex64", 48, 46189);
  EXPECT_NE(forward.repr(), inverse.repr());
}

TEST(KernelKeyRepr, SameKindDifferentFieldsDiffer) {
  EXPECT_NE(flagfft::KernelKey::leaf("t", "forward", "complex64", 64, {4, 4, 4}, 16, 1, {}, 64).repr(),
            flagfft::KernelKey::leaf("t", "inverse", "complex64", 64, {4, 4, 4}, 16, 1, {}, 64).repr());
  EXPECT_NE(
      flagfft::KernelKey::four_step_col("t", "forward", "complex64", 209, 221, 46189, {19, 11}, 1, 2, {}, 256)
          .repr(),
      flagfft::KernelKey::four_step_col("t", "forward", "complex64", 209, 221, 46189, {17, 13}, 1, 2, {}, 256)
          .repr());
  EXPECT_NE(flagfft::KernelKey::transpose3d("t", "complex64", 128, 2048, 64, "021").repr(),
            flagfft::KernelKey::transpose3d("t", "complex64", 128, 2048, 64, "210").repr());
  EXPECT_NE(flagfft::KernelKey::reshape_pack("t", "complex64", 64, 128).repr(),
            flagfft::KernelKey::reshape_pack("t", "complex64", 128, 64).repr());
}

TEST(KernelKeyRepr, DistinctKindsWithSameValuesDiffer) {
  const auto reshape = flagfft::KernelKey::reshape_pack("t", "complex64", 64, 128).repr();
  const auto transpose = flagfft::KernelKey::tiled_transpose("t", "complex64", 64, 128).repr();
  EXPECT_NE(reshape, transpose);
}

namespace {

// Plans feed one StockhamStage key per radix through compile_kernel(); every
// stage of a plan shares target/dtype/length; repeated radices still have
// different spans, so the cache key has to separate both properties.
flagfft::KernelKey stockham_stage(const std::string &target,
                                  const std::string &direction,
                                  int64_t length,
                                  int64_t radix,
                                  int64_t span = 1,
                                  int64_t block = 128) {
  auto key = flagfft::KernelKey::direct_dft(target, direction, "complex64", length);
  key.kind = flagfft::KernelKind::StockhamStage;
  key.factors = {radix, span, block};
  return key;
}

}  // namespace

TEST(KernelKeyRepr, StockhamStageDistinguishesStages) {
  const std::string target = "npu:Ascend910B4-1:1";
  EXPECT_NE(stockham_stage(target, "forward", 16384, 8).repr(),
            stockham_stage(target, "forward", 16384, 4).repr());
  EXPECT_NE(stockham_stage(target, "forward", 16384, 8).repr(),
            stockham_stage(target, "forward", 32768, 8).repr());
  EXPECT_NE(stockham_stage(target, "forward", 16384, 8).repr(),
            stockham_stage(target, "inverse", 16384, 8).repr());
  EXPECT_NE(stockham_stage(target, "forward", 16384, 8, 1).repr(),
            stockham_stage(target, "forward", 16384, 8, 8).repr());
  EXPECT_NE(stockham_stage(target, "forward", 16384, 8, 8).repr(),
            stockham_stage(target, "forward", 16384, 8, 64).repr());
  EXPECT_NE(stockham_stage(target, "forward", 1024, 8, 8, 16).repr(),
            stockham_stage(target, "forward", 1024, 8, 8, 128).repr());
}
