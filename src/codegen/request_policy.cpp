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
#include <stdexcept>
#include <string>

namespace flagfft {

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
    if (packed && std::string(packed)=="1") preset=9;
  }
  if (request.batch==64 && request.requested_n==8191) preset=1;
  if (request.batch==64 && request.requested_n==16381) preset=2;
  if (request.requested_n==16384 && request.real_transform_kind.empty())
    preset=request.batch==1 ? 3 : 4;
  if (request.requested_n==16384 && request.batch==64 && request.real_transform_kind=="c2r") preset=7;
  if (request.requested_n==16384 && request.batch==64 && request.real_transform_kind=="r2c") preset=8;
  if (request.requested_n==663000) preset=request.batch==1 ? 5 : 6;
  if (request.requested_n==340200 && request.batch==64 && request.real_transform_kind.empty()) preset=10;
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
  const bool matrix=request.origin_rank==2 && request.raw_dim==2 &&
                    n0==2048 && n1==2048 && n2==0;
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
  if (matrix) return enabled("FLAGFFT_IX_2D_TRANSPOSE_STORE",true) ? 16 : 0;
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
