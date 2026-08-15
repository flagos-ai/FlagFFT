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

namespace flagfft {

KernelKey KernelKey::leaf(std::string target,
                          std::string direction,
                          std::string dtype,
                          int64_t length,
                          std::vector<int64_t> factors,
                          int64_t lanes,
                          int64_t num_warps,
                          std::vector<int64_t> generic_radices,
                          int64_t smem_size) {
  KernelKey key;
  key.kind = KernelKind::Leaf;
  key.target = std::move(target);
  key.direction = std::move(direction);
  key.dtype = std::move(dtype);
  key.length = length;
  key.factors = std::move(factors);
  key.lanes = lanes;
  key.num_warps = num_warps;
  key.generic_radices = std::move(generic_radices);
  key.smem_size = smem_size;
  return key;
}

KernelKey KernelKey::leaf_strided(std::string target,
                                  std::string direction,
                                  std::string dtype,
                                  int64_t length,
                                  std::vector<int64_t> factors,
                                  int64_t lanes,
                                  int64_t num_warps,
                                  std::vector<int64_t> generic_radices,
                                  int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::LeafStrided;
  return key;
}

KernelKey KernelKey::leaf_r2c(std::string target,
                              std::string direction,
                              std::string dtype,
                              int64_t length,
                              std::vector<int64_t> factors,
                              int64_t lanes,
                              int64_t num_warps,
                              std::vector<int64_t> generic_radices,
                              int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::LeafR2C;
  return key;
}

KernelKey KernelKey::leaf_c2r(std::string target,
                              std::string direction,
                              std::string dtype,
                              int64_t length,
                              std::vector<int64_t> factors,
                              int64_t lanes,
                              int64_t num_warps,
                              std::vector<int64_t> generic_radices,
                              int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::LeafC2R;
  return key;
}

KernelKey KernelKey::leaf_bluestein(std::string target,
                                    std::string direction,
                                    std::string dtype,
                                    int64_t prime_n,
                                    int64_t length,
                                    std::vector<int64_t> factors,
                                    int64_t lanes,
                                    int64_t num_warps,
                                    std::vector<int64_t> generic_radices,
                                    int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::LeafBluestein;
  key.bluestein_n = prime_n;
  key.bluestein_m = length;
  return key;
}

KernelKey KernelKey::leaf_bluestein_prepare(std::string target,
                                            std::string direction,
                                            std::string dtype,
                                            int64_t prime_n,
                                            int64_t length,
                                            std::vector<int64_t> factors,
                                            int64_t lanes,
                                            int64_t num_warps,
                                            std::vector<int64_t> generic_radices,
                                            int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::LeafBluesteinPrepare;
  key.bluestein_n = prime_n;
  key.bluestein_m = length;
  return key;
}

KernelKey KernelKey::leaf_bluestein_finish(std::string target,
                                           std::string direction,
                                           std::string dtype,
                                           int64_t prime_n,
                                           int64_t length,
                                           std::vector<int64_t> factors,
                                           int64_t lanes,
                                           int64_t num_warps,
                                           std::vector<int64_t> generic_radices,
                                           int64_t smem_size) {
  KernelKey key = KernelKey::leaf_bluestein_prepare(std::move(target),
                                                    std::move(direction),
                                                    std::move(dtype),
                                                    prime_n,
                                                    length,
                                                    std::move(factors),
                                                    lanes,
                                                    num_warps,
                                                    std::move(generic_radices),
                                                    smem_size);
  key.kind = KernelKind::LeafBluesteinFinish;
  return key;
}

KernelKey KernelKey::direct_dft(std::string target,
                                std::string direction,
                                std::string dtype,
                                int64_t length) {
  KernelKey key;
  key.kind = KernelKind::DirectDft;
  key.target = std::move(target);
  key.direction = std::move(direction);
  key.dtype = std::move(dtype);
  key.length = length;
  return key;
}

KernelKey KernelKey::direct_dft_strided(std::string target,
                                        std::string direction,
                                        std::string dtype,
                                        int64_t length) {
  KernelKey key = KernelKey::direct_dft(std::move(target), std::move(direction), std::move(dtype), length);
  key.kind = KernelKind::DirectDftStrided;
  return key;
}

KernelKey KernelKey::four_step_row(std::string target,
                                   std::string direction,
                                   std::string dtype,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::FourStepRow;
  key.four_step_n1 = n1;
  key.four_step_n2 = n2;
  return key;
}

KernelKey KernelKey::four_step_row_strided(std::string target,
                                           std::string direction,
                                           std::string dtype,
                                           int64_t n1,
                                           int64_t n2,
                                           int64_t length,
                                           std::vector<int64_t> factors,
                                           int64_t lanes,
                                           int64_t num_warps,
                                           std::vector<int64_t> generic_radices,
                                           int64_t smem_size) {
  KernelKey key = KernelKey::four_step_row(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepRowStrided;
  return key;
}

KernelKey KernelKey::four_step_real_row(std::string target,
                                        std::string direction,
                                        std::string dtype,
                                        int64_t n1,
                                        int64_t n2,
                                        int64_t length,
                                        std::vector<int64_t> factors,
                                        int64_t lanes,
                                        int64_t num_warps,
                                        std::vector<int64_t> generic_radices,
                                        int64_t smem_size) {
  KernelKey key = KernelKey::four_step_row(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepRealRow;
  return key;
}

KernelKey KernelKey::four_step_hermitian_row(std::string target,
                                             std::string direction,
                                             std::string dtype,
                                             int64_t n1,
                                             int64_t n2,
                                             int64_t length,
                                             std::vector<int64_t> factors,
                                             int64_t lanes,
                                             int64_t num_warps,
                                             std::vector<int64_t> generic_radices,
                                             int64_t smem_size) {
  KernelKey key = KernelKey::four_step_row(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepHermitianRow;
  return key;
}

KernelKey KernelKey::four_step_col(std::string target,
                                   std::string direction,
                                   std::string dtype,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
  KernelKey key = KernelKey::leaf(std::move(target),
                                  std::move(direction),
                                  std::move(dtype),
                                  length,
                                  std::move(factors),
                                  lanes,
                                  num_warps,
                                  std::move(generic_radices),
                                  smem_size);
  key.kind = KernelKind::FourStepCol;
  key.four_step_n1 = n1;
  key.four_step_n2 = n2;
  return key;
}

KernelKey KernelKey::four_step_col_strided(std::string target,
                                           std::string direction,
                                           std::string dtype,
                                           int64_t n1,
                                           int64_t n2,
                                           int64_t length,
                                           std::vector<int64_t> factors,
                                           int64_t lanes,
                                           int64_t num_warps,
                                           std::vector<int64_t> generic_radices,
                                           int64_t smem_size) {
  KernelKey key = KernelKey::four_step_col(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepColStrided;
  return key;
}

KernelKey KernelKey::four_step_r2c_col(std::string target,
                                       std::string direction,
                                       std::string dtype,
                                       int64_t n1,
                                       int64_t n2,
                                       int64_t length,
                                       std::vector<int64_t> factors,
                                       int64_t lanes,
                                       int64_t num_warps,
                                       std::vector<int64_t> generic_radices,
                                       int64_t smem_size) {
  KernelKey key = KernelKey::four_step_col(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepR2CCol;
  return key;
}

KernelKey KernelKey::four_step_c2r_col(std::string target,
                                       std::string direction,
                                       std::string dtype,
                                       int64_t n1,
                                       int64_t n2,
                                       int64_t length,
                                       std::vector<int64_t> factors,
                                       int64_t lanes,
                                       int64_t num_warps,
                                       std::vector<int64_t> generic_radices,
                                       int64_t smem_size) {
  KernelKey key = KernelKey::four_step_col(std::move(target),
                                           std::move(direction),
                                           std::move(dtype),
                                           n1,
                                           n2,
                                           length,
                                           std::move(factors),
                                           lanes,
                                           num_warps,
                                           std::move(generic_radices),
                                           smem_size);
  key.kind = KernelKind::FourStepC2RCol;
  return key;
}

KernelKey KernelKey::bluestein_prepare(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::BluesteinPrepare;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.bluestein_n = n;
  key.bluestein_m = m;
  return key;
}

KernelKey KernelKey::bluestein_pointwise(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::BluesteinPointwise;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.bluestein_n = n;
  key.bluestein_m = m;
  return key;
}

KernelKey KernelKey::bluestein_finalize(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::BluesteinFinalize;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.bluestein_n = n;
  key.bluestein_m = m;
  return key;
}

KernelKey KernelKey::rader_prepare(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::RaderPrepare;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.rader_n = n;
  key.rader_m = m;
  return key;
}

KernelKey KernelKey::rader_pointwise(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::RaderPointwise;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.rader_n = n;
  key.rader_m = m;
  return key;
}

KernelKey KernelKey::rader_finalize(std::string target, std::string dtype, int64_t n, int64_t m) {
  KernelKey key;
  key.kind = KernelKind::RaderFinalize;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.rader_n = n;
  key.rader_m = m;
  return key;
}

KernelKey KernelKey::reshape_pack(std::string target, std::string dtype, int64_t n1, int64_t n2) {
  KernelKey key;
  key.kind = KernelKind::ReshapePack;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.reshape_n1 = n1;
  key.reshape_n2 = n2;
  return key;
}

KernelKey KernelKey::twiddle_reshape_pack(std::string target, std::string dtype, int64_t n1, int64_t n2) {
  KernelKey key;
  key.kind = KernelKind::TwiddleReshapePack;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.reshape_n1 = n1;
  key.reshape_n2 = n2;
  return key;
}

KernelKey KernelKey::tiled_transpose(std::string target, std::string dtype, int64_t n0, int64_t n1) {
  KernelKey key;
  key.kind = KernelKind::TiledTranspose;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.reshape_n1 = n0;
  key.reshape_n2 = n1;
  return key;
}

KernelKey KernelKey::transpose3d(
    std::string target, std::string dtype, int64_t n0, int64_t n1, int64_t n2, std::string order) {
  KernelKey key;
  key.kind = KernelKind::Transpose3D;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.transpose3d_n0 = n0;
  key.transpose3d_n1 = n1;
  key.transpose3d_n2 = n2;
  key.transpose3d_order = std::move(order);
  return key;
}

KernelKey KernelKey::real_to_complex(std::string target, std::string dtype, int64_t length) {
  KernelKey key;
  key.kind = KernelKind::RealToComplex;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.length = length;
  return key;
}

KernelKey KernelKey::r2c_half_pack(std::string target, std::string dtype, int64_t length) {
  KernelKey key;
  key.kind = KernelKind::R2CHalfPack;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.length = length;
  return key;
}

KernelKey KernelKey::compact_to_hermitian_full(std::string target, std::string dtype, int64_t length) {
  KernelKey key;
  key.kind = KernelKind::CompactToHermitianFull;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.length = length;
  return key;
}

KernelKey KernelKey::complex_to_real(std::string target, std::string dtype, int64_t length) {
  KernelKey key;
  key.kind = KernelKind::ComplexToReal;
  key.target = std::move(target);
  key.dtype = std::move(dtype);
  key.length = length;
  return key;
}

bool KernelKey::operator==(const KernelKey &other) const {
  return kind == other.kind && target == other.target && direction == other.direction &&
         dtype == other.dtype && length == other.length && factors == other.factors && lanes == other.lanes &&
         num_warps == other.num_warps && generic_radices == other.generic_radices &&
         smem_size == other.smem_size && four_step_n1 == other.four_step_n1 &&
         four_step_n2 == other.four_step_n2 && bluestein_n == other.bluestein_n &&
         bluestein_m == other.bluestein_m && rader_n == other.rader_n && rader_m == other.rader_m &&
         reshape_n1 == other.reshape_n1 && reshape_n2 == other.reshape_n2 &&
         transpose3d_n0 == other.transpose3d_n0 && transpose3d_n1 == other.transpose3d_n1 &&
         transpose3d_n2 == other.transpose3d_n2 && transpose3d_order == other.transpose3d_order;
}

std::string KernelKey::repr() const {
  std::ostringstream out;
  out << "kind=" << kernel_kind_name(kind) << ";target=" << target << ";dtype=" << dtype;
  if (kind == KernelKind::DirectDft || kind == KernelKind::DirectDftStrided) {
    out << ";direction=" << direction << ";length=" << length;
  }
  if (kind == KernelKind::Leaf || kind == KernelKind::LeafStrided || kind == KernelKind::LeafR2C ||
      kind == KernelKind::LeafC2R || kind == KernelKind::LeafBluestein ||
      kind == KernelKind::LeafBluesteinPrepare || kind == KernelKind::LeafBluesteinFinish ||
      kind == KernelKind::BluesteinFourStepPrepareRow || kind == KernelKind::BluesteinFourStepPointwiseRow ||
      kind == KernelKind::BluesteinFourStepFinishCol || kind == KernelKind::FourStepRow ||
      kind == KernelKind::FourStepRowStrided || kind == KernelKind::FourStepRealRow ||
      kind == KernelKind::FourStepHermitianRow || kind == KernelKind::FourStepCol ||
      kind == KernelKind::FourStepColStrided || kind == KernelKind::FourStepR2CCol ||
      kind == KernelKind::FourStepC2RCol) {
    out << ";direction=" << direction << ";length=" << length << ";factors=[" << join_ints(factors) << "]"
        << ";lanes=" << lanes << ";num_warps=" << num_warps << ";generic_radices=["
        << join_ints(generic_radices) << "];smem_size=" << smem_size;
    if (kind == KernelKind::FourStepRow || kind == KernelKind::FourStepRowStrided ||
        kind == KernelKind::FourStepRealRow || kind == KernelKind::FourStepHermitianRow ||
        kind == KernelKind::FourStepCol || kind == KernelKind::FourStepColStrided ||
        kind == KernelKind::FourStepR2CCol || kind == KernelKind::FourStepC2RCol ||
        kind == KernelKind::BluesteinFourStepPrepareRow ||
        kind == KernelKind::BluesteinFourStepPointwiseRow || kind == KernelKind::BluesteinFourStepFinishCol) {
      out << ";four_step_n1=" << four_step_n1 << ";four_step_n2=" << four_step_n2;
    }
  }
  if (kind == KernelKind::LeafBluestein || kind == KernelKind::LeafBluesteinPrepare ||
      kind == KernelKind::LeafBluesteinFinish || kind == KernelKind::BluesteinFourStepPrepareRow ||
      kind == KernelKind::BluesteinFourStepPointwiseRow || kind == KernelKind::BluesteinFourStepFinishCol ||
      kind == KernelKind::BluesteinPrepare || kind == KernelKind::BluesteinPointwise ||
      kind == KernelKind::BluesteinFinalize) {
    out << ";bluestein_n=" << bluestein_n << ";bluestein_m=" << bluestein_m;
  }
  if (kind == KernelKind::RaderPrepare || kind == KernelKind::RaderPointwise ||
      kind == KernelKind::RaderFinalize) {
    out << ";rader_n=" << rader_n << ";rader_m=" << rader_m;
  }
  if (kind == KernelKind::ReshapePack || kind == KernelKind::TwiddleReshapePack) {
    out << ";reshape_n1=" << reshape_n1 << ";reshape_n2=" << reshape_n2;
  }
  if (kind == KernelKind::Transpose3D) {
    out << ";transpose3d_n0=" << transpose3d_n0 << ";transpose3d_n1=" << transpose3d_n1
        << ";transpose3d_n2=" << transpose3d_n2 << ";order=" << transpose3d_order;
  }
  if (kind == KernelKind::RealToComplex || kind == KernelKind::R2CHalfPack ||
      kind == KernelKind::CompactToHermitianFull || kind == KernelKind::ComplexToReal) {
    out << ";length=" << length;
  }
  return out.str();
}

std::size_t KernelKeyHash::operator()(const KernelKey &key) const {
  std::size_t seed = 0;
  hash_value(seed, static_cast<int64_t>(key.kind));
  hash_value(seed, key.target);
  hash_value(seed, key.direction);
  hash_value(seed, key.dtype);
  hash_value(seed, key.length);
  hash_vector(seed, key.factors);
  hash_value(seed, key.lanes);
  hash_value(seed, key.num_warps);
  hash_vector(seed, key.generic_radices);
  hash_value(seed, key.smem_size);
  hash_value(seed, key.four_step_n1);
  hash_value(seed, key.four_step_n2);
  hash_value(seed, key.bluestein_n);
  hash_value(seed, key.bluestein_m);
  hash_value(seed, key.rader_n);
  hash_value(seed, key.rader_m);
  hash_value(seed, key.reshape_n1);
  hash_value(seed, key.reshape_n2);
  hash_value(seed, key.transpose3d_n0);
  hash_value(seed, key.transpose3d_n1);
  hash_value(seed, key.transpose3d_n2);
  hash_value(seed, key.transpose3d_order);
  return seed;
}

}  // namespace flagfft
