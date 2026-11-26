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

// Request-level backend policy predicates.  These depend only on the request
// and the environment, so they live outside the code generator and remain
// linkable by CPU-only planner tests.

#include "flagfft/codegen.hpp"

#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace flagfft {

std::vector<int64_t> hcu_real_inverse_factors(const FFTRequest &request, const LeafPlanNode &leaf) {
  const char *setting = performance_env("FLAGFFT_HCU_PACKED_CTA_PUBLIC");
  if (request.device_type != "hcu" || request.device_arch != "gfx936" || request.origin_rank > 1 ||
      request.input_dtype != "complex128" || !setting || std::string(setting) != "resource" ||
      leaf.length <= 0 || (leaf.length & (leaf.length - 1)) || leaf.factors.size() < 2)
    return leaf.factors;
  auto factors = leaf.factors;
  std::size_t best = 0;
  for (std::size_t i = 1; i < factors.size(); ++i) {
    const int64_t active = leaf.length / factors[i];
    // The Hermitian global read benefits from more independent butterflies;
    // keep a complete first stage within two physical waves.
    if (active >= 64 && active <= 128 && factors[i] < factors[best]) best = i;
  }
  if (best) std::rotate(factors.begin(), factors.begin() + best, factors.begin() + best + 1);
  return factors;
}

int64_t hcu_packed_grid_pack_limit(int64_t batch, int64_t compute_units, int64_t active_lanes) {
  if (batch <= 0 || compute_units <= 0 || active_lanes <= 0) return 0;
  // Leave two blocks per CU before grouping rows, while filling the physical
  // wave for short FFTs. A transposed boundary separately retains its wider
  // output-transaction grouping.
  const int64_t requested = std::max<int64_t>(64 / active_lanes,
      std::max<int64_t>(1, batch / compute_units / 2));
  int64_t pack = 1;
  while (pack <= requested / 2) pack *= 2;
  return std::min<int64_t>(pack, 16);
}

std::vector<int64_t> hcu_packed_cta_factors(const FFTRequest &request, const LeafPlanNode &leaf) {
  const char *setting = performance_env("FLAGFFT_HCU_PACKED_REAL_CTA");
  if (request.device_type != "hcu" || request.device_arch != "gfx936" ||
      !setting || std::string(setting) != "1" || leaf.length < 32 || leaf.length > 1024 ||
      (leaf.length & (leaf.length - 1)) || leaf.factors.empty()) return leaf.factors;
  // Reconstruction keeps a second complex partner live. Bound the butterfly
  // at radix sixteen, distributing the radix bits evenly across FFT stages
  // so a tiny final butterfly does not require incomplete shared-read groups.
  int bits = 0;
  for (int64_t n = leaf.length; n > 1; n /= 2) ++bits;
  const int stages = (bits + 3) / 4;
  std::vector<int64_t> factors;
  for (int stage = 0; stage < stages; ++stage)
    factors.push_back(int64_t{1} << (bits / stages + (stage < bits % stages)));
  return factors;
}

int64_t permuted_store_outer_blocks(int64_t transforms, int64_t span, int64_t pack) {
  if (transforms < 0 || span <= 0 || pack <= 0)
    throw std::invalid_argument("invalid outer permutation launch dimensions");
  const int64_t planes = transforms / span + (transforms % span != 0);
  const int64_t tiles = span / pack + (span % pack != 0);
  if (planes > std::numeric_limits<int64_t>::max() / tiles)
    throw std::overflow_error("outer permutation launch size overflow");
  return planes * tiles;
}

bool hcu_wave_3d_c2c_policy_enabled(const FFTRequest &request,
                                    int64_t n0, int64_t n1, int64_t n2) {
  if (request.device_type != "hcu" || request.device_arch != "gfx936" ||
      request.origin_rank != 3 || request.raw_dim != 3 || request.real_transform ||
      !request.real_transform_kind.empty() || !request.origin_real_transform_kind.empty() ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy) return false;
  for (int64_t axis : {n0, n1, n2}) {
    if (axis < 64 || axis > 2048 || (axis & (axis - 1)) != 0) return false;
  }
  const char *wave = performance_env("FLAGFFT_HCU_3D_WARP_FFT");
  const char *roots = performance_env("FLAGFFT_HCU_WAVE_ROOT_TABLE");
  return wave && std::string(wave) == "1" && roots && std::string(roots) == "1";
}

int ix_fourstep_default_preset(const FFTRequest &request) {
  if (request.device_type!="ix" || request.device_arch!="71" || request.raw_dim!=1 ||
      request.origin_rank>1 || request.fft_length!=request.requested_n ||
      request.input_dtype!="complex64" || request.output_dtype!="complex64" ||
      request.input_layout!="contiguous" || request.requires_contiguous_copy ||
      (request.batch!=1 && request.batch!=64) || request.input_strides.size()!=2 ||
      request.input_strides.back()!=1 ||
      (request.real_transform_kind!="" && request.real_transform_kind!="r2c" &&
       request.real_transform_kind!="c2r")) return 0;
  const int64_t dense_input = request.real_transform_kind=="c2r"
      ? request.requested_n/2+1 : request.requested_n;
  if (request.input_strides.front()!=dense_input) return 0;
  const char *policy=std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return 0;
  int preset=0;
  if (request.batch==64 && (request.requested_n==16 || request.requested_n==210 ||
                            request.requested_n==1024 || request.requested_n==2048)) {
    const char *packed=std::getenv("FLAGFFT_IX_CT_BATCH_PACKED_IO");
    if (packed && std::string(packed)!="0" && std::string(packed)!="1")
      throw std::runtime_error("FLAGFFT_IX_CT_BATCH_PACKED_IO must be 0 or 1");
    if (!packed || std::string(packed)=="1") {
      preset=request.requested_n==1024 ? 11 : 9;
      if (request.requested_n==210) preset=request.real_transform_kind.empty() ? 12 : 13;
    }
  }
  if (request.batch==1 && request.requested_n==210 && request.real_transform_kind=="r2c") preset=13;
  if (request.batch==64 && request.requested_n==8191) preset=1;
  if (request.batch==64 && request.requested_n==16381) preset=2;
  if (request.requested_n==16384 && request.real_transform_kind.empty())
    preset=request.batch==1 ? 3 : 4;
  if (request.requested_n==16384 && request.batch==64 && request.real_transform_kind=="c2r") preset=7;
  if (request.requested_n==16384 && request.batch==64 && request.real_transform_kind=="r2c") preset=8;
  if (request.requested_n==663000) preset=request.batch==1 ? 5 : 6;
  if (request.requested_n==340200 && request.real_transform_kind.empty() &&
      (request.batch==64 || request.origin_real_transform_kind.empty())) preset=10;
  if (request.requested_n==524288 && request.batch==1 && request.real_transform_kind.empty() &&
      request.origin_real_transform_kind.empty()) {
    const char *large = std::getenv("FLAGFFT_IX_LARGE_FOURSTEP");
    if (!large || std::string(large)=="1") preset=14;
  }
  if (!preset) return 0;
  const char *setting=std::getenv("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS");
  if (setting && std::string(setting)!="0" && std::string(setting)!="1")
    throw std::runtime_error("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS must be 0 or 1");
  if (setting && std::string(setting)=="0") return 0;
  if (preset==4) {
    const char *cta=std::getenv("FLAGFFT_IX_FOURSTEP_CTA");
    const char *portable=std::getenv("FLAGFFT_IX_PORTABLE_LEAF");
    if ((cta && std::string(cta)=="0") || (portable && std::string(portable)=="1")) return 0;
  }
  if (preset==7 || preset==8) {
    for (const char *name : {"FLAGFFT_IX_FOURSTEP_CTA", "FLAGFFT_IX_PORTABLE_LEAF",
                            "FLAGFFT_PACKED_REAL",
                            preset==7 ? "FLAGFFT_IX_FUSED_C2R" : "FLAGFFT_IX_FUSED_R2C"}) {
      const char *value=std::getenv(name);
      if (value && std::string(value)=="0") return 0;
    }
  }
  const char *rollback=std::getenv(request.batch==1 ? "FLAGFFT_IX_CT_SINGLE" : "FLAGFFT_IX_CT_BATCH");
  return rollback && std::string(rollback)=="0" ? 0 : preset;
}

bool ix_small_real_warp_default_enabled(const FFTRequest &request) {
  if (request.device_type!="ix" || request.device_arch!="71" || request.raw_dim!=1 ||
      request.origin_rank>1 || request.batch!=1 || request.requested_n!=16 ||
      request.fft_length!=16 || request.input_dtype!="complex64" ||
      request.output_dtype!="complex64" || request.input_layout!="contiguous" ||
      request.requires_contiguous_copy ||
      (request.real_transform_kind!="r2c" && request.real_transform_kind!="c2r")) return false;
  const char *policy=std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return false;
  const char *value=std::getenv("FLAGFFT_IX_SMALL_REAL");
  if (!value || std::string(value)=="1") return true;
  if (std::string(value)=="0") return false;
  throw std::runtime_error("FLAGFFT_IX_SMALL_REAL must be 0 or 1");
}

bool ix_short_real_warp_default_enabled(const FFTRequest &request) {
  if (request.device_type!="ix" || request.device_arch!="71" || request.raw_dim!=1 ||
      request.origin_rank>1 || request.batch!=1 || request.requested_n!=16384 ||
      request.fft_length!=16384 || request.input_dtype!="complex64" ||
      request.output_dtype!="complex64" || request.input_layout!="contiguous" ||
      request.requires_contiguous_copy || request.input_strides.empty() ||
      request.input_strides.back()!=1 ||
      (request.real_transform_kind!="r2c" && request.real_transform_kind!="c2r")) return false;
  const char *policy=std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return false;
  for (const char *name : {"FLAGFFT_IX_SHORT_REAL", "FLAGFFT_IX_WARP_FOURSTEP", "FLAGFFT_IX_WARP_FFT"}) {
    const char *value=std::getenv(name);
    if (value && std::string(value)=="0") return false;
    if (value && std::string(value)!="1")
      throw std::runtime_error(std::string(name)+" must be 0 or 1");
  }
  return true;
}

bool ix_prime_3d_default_enabled(const FFTRequest &request, int64_t batch,
                                 int64_t n0, int64_t n1, int64_t n2) {
  if (request.device_type!="ix" || request.device_arch!="71" ||
      request.origin_rank!=3 || request.raw_dim!=3 ||
      request.input_dtype!="complex64" || request.output_dtype!="complex64" ||
      request.input_layout!="contiguous" || request.requires_contiguous_copy ||
      (batch!=1 && batch!=4) || n0!=16 || n1!=997 || n2!=64) return false;
  const char *policy=std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return false;
  const char *value=std::getenv("FLAGFFT_IX_PRIME_3D");
  if (!value || std::string(value)=="1") return true;
  if (std::string(value)=="0") return false;
  throw std::runtime_error("FLAGFFT_IX_PRIME_3D must be 0 or 1");
}

int64_t ix_nd_warp_default_pack(const FFTRequest &request, int64_t batch,
                                int64_t n0, int64_t n1, int64_t n2) {
  const bool small_matrix = n0 == 128 && n1 == 128 && request.real_transform_kind.empty();
  const bool small_real_matrix = n0 == 64 && n1 == 64 && batch == 1 &&
      request.real_transform_kind == "r2c";
  const bool large_matrix = n0 == 2048 && n1 == 2048;
  const bool matrix = request.origin_rank == 2 && request.raw_dim == 2 && n2 == 0 &&
                      (small_matrix || small_real_matrix || large_matrix);
  const bool cube=request.origin_rank==3 && request.raw_dim==3 &&
                  n0==256 && n1==256 && n2==256;
  const bool elongated=request.origin_rank==3 && request.raw_dim==3 &&
                       n0==128 && n1==2048 && n2==64 &&
                       (request.real_transform_kind.empty() || request.real_transform_kind=="r2c");
  if (request.device_type!="ix" || request.device_arch!="71" ||
      request.input_dtype!="complex64" || request.output_dtype!="complex64" ||
      request.input_layout!="contiguous" || request.requires_contiguous_copy ||
      (batch!=1 && batch!=4) || (!matrix && !cube && !elongated)) return 0;
  const char *policy=std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return 0;
  const auto enabled=[](const char *name,bool fallback) {
    const char *value=std::getenv(name);
    if (!value) return fallback;
    if (std::string(value)=="0") return false;
    if (std::string(value)=="1") return true;
    throw std::runtime_error(std::string(name)+" must be 0 or 1");
  };
  if (!enabled("FLAGFFT_IX_ND_WARP",true) || !enabled("FLAGFFT_IX_WARP_FFT",true)) return 0;
  if (matrix) {
    if (small_real_matrix) return enabled("FLAGFFT_IX_2D_R2C_WAVE", true) ? 1 : 0;
    if (!enabled("FLAGFFT_IX_2D_TRANSPOSE_STORE", true)) return 0;
    if (small_matrix) return 8;
    return request.real_transform_kind == "c2r" ? 4 : 16;
  }
  if (!enabled("FLAGFFT_IX_WARP256",true) || !enabled("FLAGFFT_3D_FUSED_STORE",true) ||
      enabled("FLAGFFT_IX_3D_HYBRID",false)) return 0;
  if (!request.real_transform_kind.empty() &&
      (!enabled("FLAGFFT_IX_3D_REAL_HYBRID",true) || enabled("FLAGFFT_IX_3D_REAL_RTRT",false))) return 0;
  if (request.real_transform_kind=="r2c" && !enabled("FLAGFFT_IX_3D_R2C_FUSED_MIDDLE",true)) return 0;
  return 16;
}

bool ix_real_single_pack_policy_enabled(const FFTRequest &request) {
  // Retain the existing transport name for packed I/O in qualified single
  // CT leaves. Complex leaves share the real path's vector/static defaults.
  if (request.device_type!="ix" || request.device_arch!="71") return false;
  const char *policy = std::getenv("FLAGFFT_EXECUTION_POLICY");
  if (policy && std::string(policy)!="balanced") return false;
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (setting && std::string(setting)=="0") return false;
  if (setting && std::string(setting)!="1")
    throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
  const bool real = request.real_transform_kind=="r2c" || request.real_transform_kind=="c2r";
  const bool length = request.requested_n==210 || request.requested_n==1024 ||
                      request.requested_n==2048 ||
                      (request.requested_n==16 && request.real_transform_kind.empty());
  return request.device_type == "ix" && request.device_arch == "71" &&
                         request.raw_dim == 1 && request.origin_rank <= 1 &&
                         request.batch == 1 && length &&
                         request.fft_length == request.requested_n && request.input_dtype == "complex64" &&
                         request.output_dtype == "complex64" &&
                         (real || request.real_transform_kind.empty()) &&
                         request.input_layout=="contiguous" && !request.requires_contiguous_copy &&
                         !request.input_strides.empty() && request.input_strides.back() == 1;
}

bool ix_ct_single_policy_enabled(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      (request.requested_n != 1024 && request.requested_n != 2048 && request.requested_n != 16384) ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) {
    return false;
  }
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return true;
  if (std::string(setting) == "0") return false;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

bool ix_ct_batch_policy_enabled(const FFTRequest &request) {
  // Prime Bluestein convolutions at 8191 and 16381 benefit from the portable
  // four-warp exchange. The 8192-point case is a packed-real child only.
  const bool measured_1d_batch = request.origin_rank <= 1 &&
      (request.requested_n == 16384 || request.requested_n == 8191 ||
       request.requested_n == 16381 ||
       (request.requested_n == 8192 && request.packed_real_child));
  const bool qualified_length = request.requested_n == 1024 || request.requested_n == 2048 ||
                                measured_1d_batch;
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 64 || request.fft_length != request.requested_n ||
      !qualified_length ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) {
    return false;
  }
  const char *setting = std::getenv("FLAGFFT_IX_CT_BATCH");
  if (!setting || std::string(setting) == "1") return true;
  if (std::string(setting) == "0") return false;
  throw std::runtime_error("FLAGFFT_IX_CT_BATCH must be 0 or 1");
}

bool ix_packed_real_policy_enabled(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) return false;
  switch (request.requested_n) {
    case 328050:
    case 340200:
    case 663000:
    case 1048576:
      break;
    default:
      return false;
  }
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return true;
  if (std::string(setting) == "0") return false;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

int ix_ct_single_tle_policy(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) return 0;
  int policy = 0;
  if (request.requested_n == 328050 || request.requested_n == 340200) policy = 1;
  if (request.requested_n == 1048576) policy = 2;
  if (!policy) return 0;
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return policy;
  if (std::string(setting) == "0") return 0;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

}  // namespace flagfft
