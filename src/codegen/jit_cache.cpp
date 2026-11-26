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
#include "flagfft/maca_tail_policy.hpp"

#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <sys/file.h>
#include <unistd.h>
#include <utility>

#if defined(BACKEND_MACA)
#include "triton_jit/jit_utils.h"
#endif

namespace flagfft {
namespace {

  constexpr std::pair<const char *, const char *> kIxBluesteinCtaDefaults[] = {
      {"FLAGFFT_IX_BLUESTEIN_SINGLE_SMEM", "1"},
      {"FLAGFFT_IX_WARPS", "16"},
      {"FLAGFFT_IX_SMEM_SWIZZLE", "1"},
      {"FLAGFFT_IX_SMEM_SWIZZLE_SHIFT", "5"},
      {"FLAGFFT_IX_TLE_DISJOINT_CTA", "1"},
  };

  constexpr std::pair<const char *, const char *> kIxPackedRealTleDefaults[] = {
      {"FLAGFFT_IX_PACKED_R2C_TLE", "1"},
      {"FLAGFFT_IX_PACKED_R2C_INPUT", "1"},
      {"FLAGFFT_IX_PACKED_C2R_TLE", "1"},
      {"FLAGFFT_IX_PORTABLE_LEAF", "0"},
      {"FLAGFFT_IX_BATCH_PACK", "1"},
      {"FLAGFFT_IX_SMEM_SWIZZLE", "1"},
      {"FLAGFFT_IX_TLE_DISJOINT_STAGE", "1"},
      {"FLAGFFT_IX_TLE_DISJOINT_CTA", "1"},
  };

  bool npu_auto_blockify_for(const KernelKey &key) {
    if (key.perm_form=="npu_paired_real_boundary_auto") return true;
    if (key.perm_form=="npu_frequency_dft") return true;
    const bool matrix=key.perm_form.rfind("npu_matrix_fourstep_",0)==0 ||
        key.perm_form=="npu_matrix_pack" || key.perm_form=="npu_matrix_cube";
    const bool long_power2 = key.length >= 512 &&
        (key.perm_form == "structured_power2" || key.perm_form == "structured_recursive_power2");
    const bool structured_fourstep = key.perm_form.rfind("npu_structured_recursive_axis_planar",0)==0;
    const bool vector_dif = key.perm_form.rfind("npu_vector_dif_",0)==0;
    // The planner records scheduling per pass, so batch-only choices cannot
    // silently change another plan that shares the same transform length.
    if (vector_dif && key.perm_form.size()>=5 &&
        key.perm_form.compare(key.perm_form.size()-5,5,"_auto")==0) return true;
    const char *policy = matrix ? "FLAGFFT_NPU_MATRIX_AUTO_BLOCK" :
                         long_power2 ? "FLAGFFT_NPU_POWER2_AUTO_BLOCK" :
                         structured_fourstep ? "FLAGFFT_NPU_FOURSTEP_AUTO_BLOCK" :
                         vector_dif ? "FLAGFFT_NPU_DIF_AUTO_BLOCK" : nullptr;
    if (policy != nullptr) {
      if (const char *value=performance_env(policy)) {
        const std::string setting(value);
        if (setting!="0" && setting!="1")
          throw std::runtime_error(std::string(policy) + " must be 0 or 1");
        if (setting == "1" && structured_fourstep) {
          const auto pack_pos = key.perm_form.find("_pack_");
          const bool row = key.kind == KernelKind::FourStepRow;
          const int64_t axis = row ? key.four_step_n1 : key.four_step_n2;
          const int64_t width = row ? key.four_step_n2 : key.four_step_n1;
          const int64_t budget = axis >= 512 ? 4096 : 2048;
          const int64_t pack = pack_pos != std::string::npos
              ? std::stoll(key.perm_form.substr(pack_pos + 6))
              : std::min({int64_t{8}, width, std::max<int64_t>(1, budget / axis)});
          // Narrow structured slices currently fail under CANN loop scheduling.
          // Keep their full logical grid; wider groups have native replay coverage.
          const bool dense_col=!row && key.four_step_n1>=512 && key.four_step_n2>=512 &&
              env_flag_enabled(performance_env("FLAGFFT_NPU_FOURSTEP_DENSE_SECOND"));
          if (pack < 8 && !(dense_col && pack>=4 && axis*pack==budget)) return false;
        }
        return setting=="1";
      }
    }
    return env_flag_enabled(std::getenv("TRITON_ALL_BLOCKS_PARALLEL"));
  }

  // std::hash is not guaranteed to be stable across standard library versions.
  // The two independent FNV streams give request directories stable names.
  std::string request_id(const std::string &request) {
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t first = 14695981039346656037ULL;
    uint64_t second = 7809847782465536322ULL;
    for (unsigned char byte : request) {
      first = (first ^ byte) * prime;
      second = (second ^ byte) * prime;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << first
        << std::setw(16) << second;
    return out.str();
  }

  class KernelFileLock {
   public:
    explicit KernelFileLock(const std::filesystem::path &path) {
      std::filesystem::create_directories(path.parent_path());
      fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
      if (fd_ < 0) {
        throw std::runtime_error("cannot open JIT lock " + path.string() + ": " +
                                 std::strerror(errno));
      }
      while (flock(fd_, LOCK_EX) != 0) {
        if (errno == EINTR) continue;
        const int error = errno;
        close(fd_);
        throw std::runtime_error("cannot lock JIT request " + path.string() + ": " +
                                 std::strerror(error));
      }
    }

    KernelFileLock(const KernelFileLock &) = delete;
    KernelFileLock &operator=(const KernelFileLock &) = delete;

    ~KernelFileLock() {
      flock(fd_, LOCK_UN);
      close(fd_);
    }

   private:
    int fd_ = -1;
  };

  void reject_legacy_backend_env() {
    for (const char *name : {"FLAGFFT_KERNEL_BACKEND", "FFT_BACKEND"}) {
      const char *value = performance_env(name);
      if (value == nullptr || std::strlen(value) == 0) {
        continue;
      }
      if (std::string(value) == "JIT") {
        continue;
      }
      throw std::runtime_error(std::string(name) + "=" + value + " is unsupported; FlagFFT is JIT-only");
    }
  }

  std::string run_command_capture_stdout(const std::string &command) {
    std::array<char, 4096> buffer {};
    std::string output;
    int status = 0;
    FILE *pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
      throw std::runtime_error("failed to start command: " + command);
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
      output += buffer.data();
    }
    status = pclose(pipe);
    if (status != 0) {
      std::ostringstream message;
      message << "command failed with status " << status << ": " << command << "\n" << output;
      throw std::runtime_error(message.str());
    }
    return output;
  }

  std::string json_string_field(const std::string &json, const std::string &field) {
    const std::string key = "\"" + field + "\"";
    std::size_t pos = json.find(key);
    if (pos == std::string::npos) {
      throw std::runtime_error("missing JSON field: " + field);
    }
    pos = json.find(':', pos);
    pos = json.find('"', pos);
    if (pos == std::string::npos) {
      throw std::runtime_error("invalid JSON string field: " + field);
    }
    std::size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated JSON string field: " + field);
    }
    return json.substr(pos + 1, end - pos - 1);
  }

  int64_t json_int_field(const std::string &json, const std::string &field) {
    const std::string key = "\"" + field + "\"";
    std::size_t pos = json.find(key);
    if (pos == std::string::npos) {
      throw std::runtime_error("missing JSON field: " + field);
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
      throw std::runtime_error("invalid JSON integer field: " + field);
    }
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
      ++pos;
    }
    std::size_t end = pos;
    while (end < json.size() && (json[end] == '-' || std::isdigit(static_cast<unsigned char>(json[end])))) {
      ++end;
    }
    return std::stoll(json.substr(pos, end - pos));
  }

  bool json_bool_field(const std::string &json, const std::string &field) {
    const std::string key = "\"" + field + "\"";
    std::size_t pos = json.find(key);
    if (pos == std::string::npos) {
      throw std::runtime_error("missing JSON field: " + field);
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
      throw std::runtime_error("invalid JSON boolean field: " + field);
    }
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
      ++pos;
    }
    if (json.compare(pos, 4, "true") == 0) {
      return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
      return false;
    }
    throw std::runtime_error("invalid JSON boolean field: " + field);
  }

  struct KernelCacheState {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<JitKernel>> cache;
    int64_t hits = 0;
    int64_t misses = 0;
  };

  KernelCacheState &kernel_cache_state() {
    static KernelCacheState *state = new KernelCacheState();
    return *state;
  }

}  // namespace

std::shared_ptr<JitKernel> TritonCompiler::compile_kernel(const KernelKey &key) const {
  const BackendPerformanceScope performance_policy(key.target, key.dtype);
  reject_legacy_backend_env();
  const auto device_profile = adaptor::device_capabilities_json();
  const char *policy_env = std::getenv("FLAGFFT_EXECUTION_POLICY");
  const std::string default_policy = adaptor::backend_name() == "ix"
                                         ? "balanced"
                                         : (adaptor::backend_name() == "hcu" ? "native" : "legacy");
  const std::string policy = policy_env ? policy_env : default_policy;
  const auto tail_mode = maca_tail_kernel_mode(maca_tail_policy_, key.kind, key.dtype,
                                               key.length, key.four_step_n1, key.four_step_n2);
  std::string cache_key = key.repr() + device_profile + policy + ";profile-v1;maca-1d-single=" +
                                (maca_1d_single_policy_ ? "1" : "0") + ";maca-1d-batch=" +
                                (maca_1d_batch_policy_ ? "1" : "0") + ";maca-2d-single=" +
                                (maca_2d_single_policy_ ? "1" : "0") + ";maca-3d=" +
                                (maca_3d_policy_ ? "1" : "0") + ";maca-3d-c2c32-single-cube=" +
                                (maca_3d_c2c32_single_cube_policy_ ? "1" : "0") + ";ix-ct-single=" +
                                (ix_ct_single_policy_ ? "1" : "0") + ";ix-ct-single-tle=" +
                                std::to_string(ix_ct_single_tle_policy_) + ";ix-ct-batch=" +
                                (ix_ct_batch_policy_ ? "1" : "0") + ";ix-real-single-pack=" +
                                (ix_real_single_pack_ ? "1" : "0") +
                                ";ix-prime-3d=" + (ix_prime_3d_policy_ ? "1" : "0") +
                                ";ix-fourstep-preset=" + std::to_string(ix_fourstep_preset_) +
                                (adaptor::backend_name() == "maca"
                                     ? maca_tail_codegen_identity(tail_mode) : "");
  const bool ix_bluestein_cta = key.perm_form == "ix_bluestein_cta";
  const bool ix_packed_real_tle = key.perm_form == "ix_r2c_tle" || key.perm_form == "ix_c2r_tle";
  const char *ix_packed_smem_shift = key.perm_form == "ix_c2r_tle" && key.length == 1024 ? "4" : "5";
  if (ix_bluestein_cta) {
    for (const auto &[name, default_value] : kIxBluesteinCtaDefaults) {
      const char *value = performance_env(name);
      cache_key += ";" + std::string(name) + "=" + (value ? value : default_value);
    }
  }
  if (ix_packed_real_tle) {
    for (const auto &[name, default_value] : kIxPackedRealTleDefaults) {
      const char *value = performance_env(name);
      cache_key += ";" + std::string(name) + "=" + (value ? value : default_value);
    }
    const char *shift = performance_env("FLAGFFT_IX_SMEM_SWIZZLE_SHIFT");
    cache_key += ";FLAGFFT_IX_SMEM_SWIZZLE_SHIFT=" + std::string(shift ? shift : ix_packed_smem_shift);
  }
  if (adaptor::backend_name() == "hcu") {
    cache_key += ";" + hcu_performance_identity();
    const char *vector_io = performance_env("FLAGFFT_HCU_VECTOR_IO");
    cache_key += ";hcu-vector-io-v2=" + std::string(vector_io ? vector_io : "0");
    const char *single_buffer = performance_env("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER");
    cache_key += ";hcu-fourstep-single-buffer-v1=" + std::string(single_buffer ? single_buffer : "0");
    const char *packing = performance_env("FLAGFFT_HCU_FOURSTEP_PACKING");
    cache_key += ";hcu-fourstep-packing-v1=" + std::string(packing ? packing : "resource");
    const char *row_twiddle = performance_env("FLAGFFT_HCU_FP32_ROW_TWIDDLE");
    cache_key += ";hcu-fp32-row-twiddle-v1=" + std::string(row_twiddle ? row_twiddle : "0");
    const char *leaf_buffer = performance_env("FLAGFFT_HCU_LEAF_SINGLE_BUFFER");
    cache_key += ";hcu-leaf-single-buffer-v1=" + std::string(leaf_buffer ? leaf_buffer : "0");
    const char *leaf_packing = performance_env("FLAGFFT_HCU_LEAF_PACKING");
    cache_key += ";hcu-leaf-packing-v1=" + std::string(leaf_packing ? leaf_packing : "resource");
    const char *cyclic_skew = performance_env("FLAGFFT_HCU_CYCLIC_SKEW");
    cache_key += ";hcu-cyclic-skew-v1=" + std::string(cyclic_skew ? cyclic_skew : "unit");
    const char *small_wave = performance_env("FLAGFFT_HCU_SMALL_WAVE");
    cache_key += ";hcu-small-wave-v1=" + std::string(small_wave ? small_wave : "0");
    const char *dense_wave = performance_env("FLAGFFT_HCU_DENSE_WAVE");
    const char *dense_wave_pack = performance_env("FLAGFFT_HCU_DENSE_WAVE_PACK");
    const char *dense_wave_fp64 = performance_env("FLAGFFT_HCU_DENSE_WAVE_FP64");
    cache_key += ";hcu-dense-wave-v2=" + std::string(dense_wave ? dense_wave : "0") +
                 ";fp64=" + std::string(dense_wave_fp64 ? dense_wave_fp64 : "0") +
                 ";pack=" + std::string(dense_wave_pack ? dense_wave_pack : "auto");
    const char *wave_roots = performance_env("FLAGFFT_HCU_WAVE_ROOT_TABLE");
    cache_key += ";hcu-wave-root-table-v1=" + std::string(wave_roots ? wave_roots : "0");
    const char *root_period = performance_env("FLAGFFT_HCU_WAVE_ROOT_PERIOD");
    cache_key += ";hcu-wave-root-period-v1=" + std::string(root_period ? root_period : "0");
    const char *wave_codelet = performance_env("FLAGFFT_HCU_WAVE_CODELET");
    const char *packed_store = performance_env("FLAGFFT_HCU_PACKED_REAL_STORE");
    cache_key += ";hcu-packed-real-store-v1=" + std::string(packed_store ? packed_store : "0");
    const char *fill_wave = performance_env("FLAGFFT_HCU_PERMUTED_FILL_WAVE");
    cache_key += ";hcu-permuted-fill-wave-v1=" + std::string(fill_wave ? fill_wave : "0");
    const char *packed_layout = performance_env("FLAGFFT_HCU_PACKED_WAVE_LAYOUT");
    cache_key += ";hcu-packed-wave-layout-v1=" + std::string(packed_layout ? packed_layout : "wave");
    const char *conv_buffer = performance_env("FLAGFFT_HCU_CONV_SINGLE_BUFFER");
    cache_key += ";hcu-conv-single-buffer-v1=" + std::string(conv_buffer ? conv_buffer : "0");
    const char *packed_cta = performance_env("FLAGFFT_HCU_PACKED_REAL_CTA");
    cache_key += ";hcu-packed-cta-v1=" + std::string(packed_cta ? packed_cta : "0");
    const char *transpose_2d = performance_env("FLAGFFT_HCU_2D_TRANSPOSE");
    cache_key += ";hcu-2d-transpose-v1=" + std::string(transpose_2d ? transpose_2d : "auto");
    if (key.kind == KernelKind::Transpose3D) {
      for (const char *name : {"FLAGFFT_HCU_3D_TRANSPOSE_MODE", "FLAGFFT_HCU_3D_TRANSPOSE_TILE",
                               "FLAGFFT_HCU_3D_TRANSPOSE_PAIR", "FLAGFFT_HCU_3D_TRANSPOSE_SLICE_GROUP"}) {
        const char *value = performance_env(name);
        cache_key += ";" + std::string(name) + "=" + (value ? value : "auto");
      }
    }
    cache_key += ";hcu-wave-codelet-v1=" + std::string(wave_codelet ? wave_codelet : "0");
    const char *wave_budget = performance_env("FLAGFFT_HCU_WAVE_SCALAR_BUDGET");
    cache_key += ";hcu-wave-batched-v1;budget=" + std::string(wave_budget ? wave_budget : "64");
    const char *wide_lanes = performance_env("FLAGFFT_HCU_WIDE_WAVE_LANES");
    cache_key += ";hcu-wide-wave-lanes-v1=" + std::string(wide_lanes ? wide_lanes : "64");
    const char *permuted_wave = performance_env("FLAGFFT_HCU_3D_WARP_FFT");
    const char *permuted_pack = performance_env("FLAGFFT_HCU_3D_WARP_PACK");
    const char *permuted_wide = performance_env("FLAGFFT_HCU_3D_WARP_FP64");
    const char *strided_wave = performance_env("FLAGFFT_HCU_WAVE_STRIDED_INPUT");
    cache_key += ";hcu-wave-strided-input-v1=" + std::string(strided_wave ? strided_wave : "0");
    const char *strided_tile = performance_env("FLAGFFT_HCU_WAVE_STRIDED_TILE");
    cache_key += ";strided-tile=" + std::string(strided_tile ? strided_tile : "balanced");
    cache_key += ";hcu-permuted-wave-v2=" + std::string(permuted_wave ? permuted_wave : "0") +
                 ";pack=" + std::string(permuted_pack ? permuted_pack : "auto") +
                 ";fp64=" + std::string(permuted_wide ? permuted_wide : "0");
  }
  if (musa_1d_batch_policy_) {
    const char *batch_pack = std::getenv("FLAGFFT_MUSA_1D_BATCH_PACK");
    const char *batch_warps = std::getenv("FLAGFFT_MUSA_1D_BATCH_WARPS");
    const char *fourstep_pack = std::getenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK");
    const char *fourstep_row_pack = std::getenv("FLAGFFT_MUSA_1D_FOURSTEP_ROW_PACK");
    const char *fourstep_col_pack = std::getenv("FLAGFFT_MUSA_1D_FOURSTEP_COL_PACK");
    cache_key += ";musa-1d-batch=1;pack=" +
        std::string(batch_pack != nullptr ? batch_pack : "auto") + ";warps=" +
        std::string(batch_warps != nullptr ? batch_warps : "auto") + ";fourstep-pack=" +
        std::string(fourstep_pack != nullptr ? fourstep_pack : "auto") + ";row-pack=" +
        std::string(fourstep_row_pack != nullptr ? fourstep_row_pack : "auto") + ";col-pack=" +
        std::string(fourstep_col_pack != nullptr ? fourstep_col_pack : "auto");
  }
  if (adaptor::backend_name() == "npu") {
    cache_key += ";" + npu_performance_identity();
    if (key.perm_form == "npu_matrix_cube" || key.perm_form.rfind("npu_matrix_fourstep_", 0) == 0)
      cache_key += ";npu-ieee-dot-partials-v2";
    if (key.kind == KernelKind::R2CPackedPostprocess || key.kind == KernelKind::C2RPackedPreprocess) {
      const char *group=performance_env("FLAGFFT_NPU_REAL_BOUNDARY_GROUP");
      cache_key += ";npu-real-boundary-group-v5=" + std::string(group!=nullptr?group:"auto");
      const char *midpoint=performance_env("FLAGFFT_NPU_REAL_MIDPOINT_DIRECT");
      cache_key += ";npu-real-midpoint=" + std::string(midpoint!=nullptr?midpoint:"1");
    }
    if (key.perm_form.rfind("npu_vector_dif_aligned_cube_",0)==0) {
      cache_key += ";npu-dif-cube-prefix-v6";
      if (key.perm_form.rfind("npu_vector_dif_aligned_cube_0_",0)==0 &&
          key.perm_form.find("_dense_batch_f64_c64")!=std::string::npos) {
        const char *pipeline=performance_env("FLAGFFT_NPU_DIF_BATCH_DENSE_PIPELINE");
        cache_key += ";npu-dif-batch-dense-pipeline-v1=" +
                     std::string(pipeline!=nullptr?pipeline:"1");
      }
      const char *fast9=performance_env("FLAGFFT_NPU_DIF_FAST9");
      cache_key += ";npu-dif-radix9-v1=" + std::string(fast9!=nullptr?fast9:"1");
      const char *pack13=performance_env("FLAGFFT_NPU_DIF_BATCH_PACK13");
      cache_key += ";npu-dif-batch13-v1=" + std::string(pack13!=nullptr?pack13:"1");
      const char *coprime=performance_env("FLAGFFT_NPU_DIF_COPRIME");
      cache_key += ";npu-dif-coprime-v2=" + std::string(coprime!=nullptr?coprime:"1");
      if (key.perm_form.find("_vector17")!=std::string::npos)
        cache_key += ";npu-dif-vector17-codelet-v1";
      if (key.perm_form.rfind("npu_vector_dif_aligned_cube_0_",0)==0 &&
          key.perm_form.find("_vectorbasis")!=std::string::npos &&
          key.perm_form.find("_batchpack")!=std::string::npos &&
          key.perm_form.find("_half17")!=std::string::npos &&
          key.direction=="forward" && !key.factors.empty() &&
          key.factors.front()==17 && (key.length/17+511)/512>=64) {
        const char *pipeline=performance_env("FLAGFFT_NPU_DIF_PREFIX_PIPELINE");
        cache_key += ";npu-dif-prefix-pipeline-v1=" +
                     std::string(pipeline!=nullptr?pipeline:"1");
      }
      const char *component17=performance_env("FLAGFFT_NPU_DIF_COMPONENT17");
      const char *wide7=performance_env("FLAGFFT_NPU_DIF_FIRST7_WIDE");
      if (key.perm_form.rfind("npu_vector_dif_aligned_cube_0_",0)==0 &&
          key.perm_form.find("_batchpack")!=std::string::npos &&
          key.perm_form.size()>=5 &&
          key.perm_form.compare(key.perm_form.size()-5,5,"_auto")==0 &&
          !key.factors.empty() && key.factors.front()==7 &&
          (wide7==nullptr || std::string(wide7)!="0"))
        cache_key += ";npu-dif-first7-wide-v1";
      if (key.perm_form.rfind("npu_vector_dif_aligned_cube_0_",0)==0 &&
          key.perm_form.find("_vector17")!=std::string::npos &&
          key.perm_form.find("_batchpack")!=std::string::npos &&
          (component17==nullptr || std::string(component17)!="0"))
        cache_key += ";npu-dif-component17-v1";
      const char *group_branches=performance_env("FLAGFFT_NPU_DIF_GROUP_BRANCHES");
      cache_key += ";npu-dif-group-branches-v2=" +
                   std::string(group_branches!=nullptr?group_branches:"1");
    }
    if (key.perm_form.rfind("npu_structured_recursive_axis_planar",0)==0)
      cache_key += ";npu-structured-contiguous-slices-v1";
    if (key.perm_form.rfind("npu_vector_dif_flat_power2_",0)==0)
      cache_key += ";npu-flat-power-dif-v1";
    if (key.perm_form.rfind("npu_vector_dif_aligned_",0)==0) {
      const char *dense_transpose=performance_env("FLAGFFT_NPU_DIF_DENSE_TRANSPOSE");
      cache_key += ";npu-dif-dense-transpose-v2=" +
                   std::string(dense_transpose!=nullptr?dense_transpose:"1");
      const char *transpose_points=performance_env("FLAGFFT_NPU_DIF_TRANSPOSE_POINTS");
      cache_key += ";npu-dif-transpose-points=" +
                   std::string(transpose_points!=nullptr?transpose_points:"auto");
      const char *single_gather=performance_env("FLAGFFT_NPU_DIF_SINGLE_GATHER_FINISH");
      cache_key += ";npu-dif-single-gather-finish-v1=" +
                   std::string(single_gather!=nullptr?single_gather:"1");
    }
    const char *symmetric_codelets=performance_env("FLAGFFT_NPU_SYMMETRIC_CODELETS");
    cache_key += ";npu-symmetric-codelets-v1=" +
                 std::string(symmetric_codelets!=nullptr?symmetric_codelets:"1");
    const char *join = performance_env("FLAGFFT_NPU_JOIN_MODE");
    cache_key += ";npu-join-mode=" + std::string(join != nullptr ? join : "auto");
    cache_key += ";npu-auto-blockify=" + std::string(npu_auto_blockify_for(key) ? "1" : "0");
    if (key.perm_form.rfind("npu_vector_dif_aligned_",0)==0 &&
        env_flag_enabled(performance_env("FLAGFFT_NPU_DIF_FUSED_FINISH")))
      cache_key += ";npu-dif-fused-finish=3";
    const char *vector_remainder = performance_env("FLAGFFT_NPU_VECTOR_REMAINDER");
    cache_key += ";npu-source-remainder=" + std::string(vector_remainder != nullptr ? vector_remainder : "0");
    for (const char *name : {"FLAGFFT_NPU_FOURSTEP_DENSE_SECOND", "FLAGFFT_NPU_DIRECT_DFT_PAIR", "FLAGFFT_NPU_DIRECT_DFT_PREFETCH", "FLAGFFT_NPU_DIF_TRANSPOSE_TILE", "FLAGFFT_NPU_DIRECT_DFT_BT", "FLAGFFT_NPU_PAIRED_REAL_BOUNDARY", "FLAGFFT_NPU_REAL_BOUNDARY_TILE", "FLAGFFT_NPU_COMPACT_STAGES", "FLAGFFT_NPU_FOURSTEP_PACK", "FLAGFFT_NPU_AFFINE_MATRIX_IO", "FLAGFFT_NPU_TRIG_TWIDDLES"}) {
      const char *value = performance_env(name);
      cache_key += ";" + std::string(name) + "=" + (value != nullptr ? value : "default");
    }
  }
  if (key.kind == KernelKind::LeafPermutedStore && adaptor::backend_name() == "ix") {
    const char *direct_store = std::getenv("FLAGFFT_IX_3D_DIRECT_STORE");
    const char *pair_store = std::getenv("FLAGFFT_IX_3D_PAIR_STORE");
    const char *pack = std::getenv("FLAGFFT_IX_3D_PACK");
    cache_key += ";ix-3d-direct-store=" + std::string(direct_store != nullptr &&
                                                       std::string(direct_store) == "1" ? "1" : "0");
    cache_key += ";ix-3d-pair-store=" + std::string(pair_store != nullptr &&
                                                     std::string(pair_store) == "1" ? "1" : "0");
    cache_key += ";ix-3d-pack=" + std::string(pack != nullptr ? pack : "auto");
  }
  if (key.kind == KernelKind::FourStepCol && adaptor::backend_name() == "ix" &&
      key.four_step_n1 == 1024 && key.four_step_n2 == 512) {
    if (const char *setting = performance_env("FLAGFFT_IX_FOURSTEP_COL_TWIDDLE_RECURRENCE")) {
      cache_key += ";ix-fourstep-col-recurrence=" + std::string(setting);
    }
  }
  if (key.kind == KernelKind::LeafPermutedStore && adaptor::backend_name() == "npu") {
    const char *transpose_store = performance_env("FLAGFFT_NPU_2D_TRANSPOSE_STORE");
    const char *portable_leaf = performance_env("FLAGFFT_NPU_FOURSTEP_LEAF");
    const char *transpose_pack = performance_env("FLAGFFT_NPU_2D_TRANSPOSE_PACK");
    cache_key += ";npu-2d-transpose-store=" +
                 std::string(transpose_store != nullptr && std::string(transpose_store) == "1" ? "1" : "0");
    cache_key += ";npu-portable-leaf=" +
                 std::string(portable_leaf != nullptr && std::string(portable_leaf) == "1" ? "1" : "0");
    cache_key += ";npu-2d-transpose-pack=" +
                 std::string(transpose_pack != nullptr ? transpose_pack : "auto");
  }
  if (adaptor::backend_name() == "npu" &&
      (key.kind == KernelKind::Leaf || key.kind == KernelKind::LeafStrided ||
       key.kind == KernelKind::LeafPermutedStore || key.kind == KernelKind::LeafR2C ||
       key.kind == KernelKind::LeafC2R || key.kind == KernelKind::FourStepRow ||
       key.kind == KernelKind::FourStepCol || key.kind == KernelKind::FourStepRealRow ||
       key.kind == KernelKind::FourStepHermitianRow || key.kind == KernelKind::FourStepR2CCol ||
       key.kind == KernelKind::FourStepC2RCol || key.kind == KernelKind::FourStepRowStrided ||
       key.kind == KernelKind::FourStepColStrided)) {
    const char *vector_remainder = performance_env("FLAGFFT_NPU_VECTOR_REMAINDER");
    cache_key += ";npu-vector-remainder=" + std::string(vector_remainder != nullptr ? vector_remainder : "0");
    const char *vector_io = performance_env("FLAGFFT_NPU_PORTABLE_VECTOR_IO");
    cache_key += ";npu-vector-io=" + std::string(vector_io != nullptr ? vector_io : "0");
    const char *portable_leaf = performance_env("FLAGFFT_NPU_FOURSTEP_LEAF");
    const bool use_portable_leaf = key.npu_portable_leaf ||
                                   (portable_leaf != nullptr && std::string(portable_leaf) == "1");
    cache_key += ";npu-portable-leaf=" + std::string(use_portable_leaf ? "1" : "0");
    if (key.npu_portable_leaf) {
      const char *exchange = performance_env("FLAGFFT_NPU_2D_EXCHANGE");
      const char *lane_min = performance_env("FLAGFFT_NPU_2D_LANE_MIN");
      const char *batch_pack = performance_env("FLAGFFT_NPU_2D_BATCH_PACK");
      const char *split_order = performance_env("FLAGFFT_NPU_2D_SPLIT_ORDER");
      if (exchange == nullptr) exchange = std::getenv("FLAGFFT_MACA_EXCHANGE");
      if (lane_min == nullptr) lane_min = std::getenv("FLAGFFT_MACA_LANE_MIN");
      if (batch_pack == nullptr) batch_pack = std::getenv("FLAGFFT_MACA_BATCH_PACK");
      if (split_order == nullptr) split_order = std::getenv("FLAGFFT_MACA_SPLIT_ORDER");
      cache_key += ";npu-2d-exchange=" + std::string(exchange != nullptr ? exchange : "auto");
      cache_key += ";npu-2d-lane-min=" + std::string(lane_min != nullptr ? lane_min : "auto");
      cache_key += ";npu-2d-batch-pack=" + std::string(batch_pack != nullptr ? batch_pack : "auto");
      cache_key += ";npu-2d-split-order=" +
                   std::string(split_order != nullptr ? split_order : "auto");
    }
  }
  if (key.kind == KernelKind::LeafPackedR2C && adaptor::backend_name() == "ix") {
    const char *packed_3d = std::getenv("FLAGFFT_IX_3D_PACKED_R2C");
    cache_key += ";ix-3d-packed-r2c=" + std::string(packed_3d != nullptr &&
                                                     std::string(packed_3d) == "1" ? "1" : "0");
  }
  if (key.kind == KernelKind::Transpose3D && adaptor::backend_name() == "ix") {
    const char *tile = std::getenv("FLAGFFT_IX_3D_TRANSPOSE_TILE");
    const char *packed = std::getenv("FLAGFFT_IX_3D_PACKED_TRANSPOSE");
    cache_key += ";ix-3d-transpose-tile=" + std::string(tile != nullptr ? tile : "16");
    cache_key += ";ix-3d-packed-transpose=" + std::string(packed != nullptr &&
                                                         std::string(packed) == "1" ? "1" : "0");
  }
  if (adaptor::backend_name() == "cuda") {
    const char *fused_packed = std::getenv("FLAGFFT_CUDA_FUSED_PACKED_REAL");
    cache_key += ";cuda-fused-packed-real=" + std::string(fused_packed ? fused_packed : "0");
    const char *wide_stage = std::getenv("FLAGFFT_CUDA_WIDE_STAGE");
    cache_key += ";cuda-wide-stage=" + std::string(wide_stage ? wide_stage : "0");
    const char *power_register = std::getenv("FLAGFFT_CUDA_POWER2_REGISTER");
    cache_key += ";cuda-power2-register=" + std::string(power_register ? power_register : "0");
    const char *blue_register = std::getenv("FLAGFFT_CUDA_BLUESTEIN_REGISTER");
    cache_key += ";cuda-bluestein-register=" + std::string(blue_register ? blue_register : "0");
    const char *fourstep_pack = std::getenv("FLAGFFT_CUDA_FOURSTEP_PACK4");
    cache_key += ";cuda-fourstep-pack4=" + std::string(fourstep_pack ? fourstep_pack : "0");
    const char *single_buffer = std::getenv("FLAGFFT_CUDA_LEAF_SINGLE_BUFFER");
    cache_key += ";cuda-single-buffer=" + std::string(single_buffer ? single_buffer : "0");
    const char *swizzle = std::getenv("FLAGFFT_CUDA_SMEM_SWIZZLE");
    cache_key += ";cuda-smem-swizzle=" + std::string(swizzle ? swizzle : "0");
    const char *short_cooperative = std::getenv("FLAGFFT_CUDA_SHORT_COOPERATIVE");
    cache_key += ";cuda-short-cooperative=" + std::string(short_cooperative ? short_cooperative : "0");
    const char *tree = std::getenv("FLAGFFT_CUDA_DIRECT_DFT_TREE");
    const char *dft_warps = std::getenv("FLAGFFT_CUDA_DIRECT_DFT_WARPS");
    cache_key += ";cuda-dft-tree=" + std::string(tree ? tree : "0");
    cache_key += ";cuda-dft-warps=" + std::string(dft_warps ? dft_warps : "auto");
    const char *portable = std::getenv("FLAGFFT_CUDA_PORTABLE_LEAF");
    cache_key += ";cuda-portable-leaf=" + std::string(portable ? portable : "0");
  }
  if (key.kind == KernelKind::Transpose3D && adaptor::backend_name() == "cuda") {
    const char *fp64_transpose = std::getenv("FLAGFFT_CUDA_3D_FP64_TRANSPOSE");
    cache_key += ";cuda-3d-fp64-transpose=" + std::string(fp64_transpose ? fp64_transpose : "v1");
  }
  if (key.kind == KernelKind::TiledTranspose && adaptor::backend_name() == "cuda") {
    const char *transpose = std::getenv("FLAGFFT_CUDA_2D_TRANSPOSE");
    cache_key += ";cuda-2d-transpose=" + std::string(transpose ? transpose : "auto");
  }
  if (adaptor::backend_name() == "cuda" &&
      (key.kind == KernelKind::LeafPermutedStore || key.kind == KernelKind::LeafStridedPermutedStore ||
       key.kind == KernelKind::LeafR2CPermutedStore ||
       (key.kind == KernelKind::LeafR2C && key.perm_form == "permuted"))) {
    const char *pack = std::getenv("FLAGFFT_CUDA_PERMSTORE_PACK");
    const char *padding = std::getenv("FLAGFFT_CUDA_PERMSTORE_PAD");
    cache_key += ";cuda-permstore-pack=" + std::string(pack ? pack : "auto");
    cache_key += ";cuda-permstore-pad=" + std::string(padding ? padding : "1");
    const char *vector_store = std::getenv("FLAGFFT_CUDA_PERMSTORE_VEC_STORE");
    cache_key += ";cuda-permstore-vec=" + std::string(vector_store ? vector_store : "0");
  }
  KernelCacheState &state = kernel_cache_state();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.cache.find(cache_key);
    if (it != state.cache.end()) {
      ++state.hits;
      return it->second;
    }
  }

  std::string kernel_kind;
  switch (key.kind) {
    case KernelKind::Leaf:
      kernel_kind = "leaf";
      break;
    case KernelKind::LeafStrided:
      kernel_kind = "leaf_strided";
      break;
    case KernelKind::LeafPermutedStore:
      kernel_kind = "leaf_permuted_store";
      break;
    case KernelKind::LeafStridedPermutedStore:
      kernel_kind = "leaf_strided_permuted_store";
      break;
    case KernelKind::LeafR2C:
      kernel_kind = "leaf_r2c";
      break;
    case KernelKind::LeafR2CPermutedStore:
      kernel_kind = "leaf_r2c_permuted_store";
      break;
    case KernelKind::LeafPackedR2C:
      kernel_kind = "leaf_packed_r2c";
      break;
    case KernelKind::LeafC2R:
      kernel_kind = "leaf_c2r";
      break;
    case KernelKind::LeafPackedC2R:
      kernel_kind = "leaf_packed_c2r";
      break;
    case KernelKind::LeafBluestein:
      kernel_kind = key.perm_form == "outer" ? "leaf_bluestein" : "leaf_bluestein_" + key.perm_form;
      break;
    case KernelKind::LeafRaderFull:
      kernel_kind = "leaf_rader_full";
      break;
    case KernelKind::LeafRaderPrepare:
      kernel_kind = "leaf_rader_prepare";
      break;
    case KernelKind::LeafRaderFinish:
      kernel_kind = "leaf_rader_finish";
      break;
    case KernelKind::LeafBluesteinPrepare:
      kernel_kind = "leaf_bluestein_prepare";
      break;
    case KernelKind::LeafBluesteinFinish:
      kernel_kind = "leaf_bluestein_finish";
      break;
    case KernelKind::BluesteinFourStepPrepareRow:
      kernel_kind = key.perm_form == "outer" ? "bluestein_four_step_prepare_row"
                                             : "bluestein_four_step_prepare_row_" + key.perm_form;
      break;
    case KernelKind::BluesteinFourStepPointwiseRow:
      kernel_kind = "bluestein_four_step_pointwise_row";
      break;
    case KernelKind::BluesteinFourStepFinishCol:
      kernel_kind = key.perm_form == "outer" ? "bluestein_four_step_finish_col"
                                             : "bluestein_four_step_finish_col_" + key.perm_form;
      break;
    case KernelKind::DirectDft:
      kernel_kind = "direct_dft";
      break;
    case KernelKind::DirectDftStrided:
      kernel_kind = "direct_dft_strided";
      break;
    case KernelKind::DirectDftCube:
      kernel_kind = "direct_dft_cube";
      break;
    case KernelKind::DirectDftCubeTransposed:
      kernel_kind = "direct_dft_cube_transposed";
      break;
    case KernelKind::DirectDftCubeStrided:
      kernel_kind = "direct_dft_cube_strided";
      break;
    case KernelKind::DirectDftCube2DRow:
      kernel_kind = "direct_dft_cube_2d_row";
      break;
    case KernelKind::DirectDftCube2DCol:
      kernel_kind = "direct_dft_cube_2d_col";
      break;
    case KernelKind::DirectDftR2C:
      kernel_kind = "direct_dft_r2c";
      break;
    case KernelKind::DirectDftC2R:
      kernel_kind = "direct_dft_c2r";
      break;
    case KernelKind::StockhamStage:
      kernel_kind = "stockham_stage";
      break;
    case KernelKind::FourStepRow:
      kernel_kind = "four_step_row";
      break;
    case KernelKind::FourStepRowStrided:
      kernel_kind = "four_step_row_strided";
      break;
    case KernelKind::FourStepRealRow:
      kernel_kind = "four_step_real_row";
      break;
    case KernelKind::FourStepHermitianRow:
      kernel_kind = "four_step_hermitian_row";
      break;
    case KernelKind::FourStepCol:
      kernel_kind = "four_step_col";
      break;
    case KernelKind::FourStepColStrided:
      kernel_kind = "four_step_col_strided";
      break;
    case KernelKind::FourStepR2CCol:
      kernel_kind = "four_step_r2c_col";
      break;
    case KernelKind::FourStepC2RCol:
      kernel_kind = "four_step_c2r_col";
      break;
    case KernelKind::BluesteinPrepare:
      kernel_kind = "bluestein_prepare";
      break;
    case KernelKind::BluesteinPointwise:
      kernel_kind = "bluestein_pointwise";
      break;
    case KernelKind::BluesteinFinalize:
      kernel_kind = "bluestein_finalize";
      break;
    case KernelKind::RaderPrepare:
      kernel_kind = "rader_prepare";
      break;
    case KernelKind::RaderPointwise:
      kernel_kind = "rader_pointwise";
      break;
    case KernelKind::RaderFinalize:
      kernel_kind = "rader_finalize";
      break;
    case KernelKind::ReshapePack:
      kernel_kind = "reshape_pack";
      break;
    case KernelKind::TwiddleReshapePack:
      kernel_kind = "twiddle_reshape_pack";
      break;
    case KernelKind::RealPairR2CPack:
      kernel_kind = "r2c_pair_pack";
      break;
    case KernelKind::RealPairR2CUnpack:
      kernel_kind = "r2c_pair_unpack";
      break;
    case KernelKind::RealPairC2RPack:
      kernel_kind = "c2r_pair_pack";
      break;
    case KernelKind::RealPairC2RUnpack:
      kernel_kind = "c2r_pair_unpack";
      break;
    case KernelKind::RealToComplex:
      kernel_kind = "real_to_complex";
      break;
    case KernelKind::R2CHalfPack:
      kernel_kind = "r2c_half_pack";
      break;
    case KernelKind::R2CPackedPostprocess:
      kernel_kind = "r2c_packed_postprocess";
      break;
    case KernelKind::C2RPackedPreprocess:
      kernel_kind = "c2r_packed_preprocess";
      break;
    case KernelKind::CompactToHermitianFull:
      kernel_kind = "compact_to_hermitian_full";
      break;
    case KernelKind::ComplexToReal:
      kernel_kind = "complex_to_real";
      break;
    case KernelKind::TiledTranspose:
      kernel_kind = "tiled_transpose";
      break;
    case KernelKind::NpuGroupedFFT64:
      kernel_kind = "npu_grouped_fft64";
      break;
    case KernelKind::Transpose3D:
      kernel_kind = "transpose3d";
      break;
    case KernelKind::Hcu3DCTFirst:
      kernel_kind = "hcu_3d_ct_first";
      break;
    case KernelKind::Hcu3DCTLast:
      kernel_kind = "hcu_3d_ct_last";
      break;
    case KernelKind::Fused16Plane:
      kernel_kind = "fused_16_plane";
      break;
    case KernelKind::Fused16RealPlane:
      kernel_kind = "fused_16_real_plane";
      break;
    case KernelKind::Fused32Plane:
      kernel_kind = "fused_32_plane";
      break;
    case KernelKind::Fused32RealPlane:
      kernel_kind = "fused_32_real_plane";
      break;
    case KernelKind::FusedRectPlane:
      kernel_kind = "fused_rect_plane";
      break;
    case KernelKind::Fused16Cube:
      kernel_kind = "fused_16_cube";
      break;
    case KernelKind::Fused16RealCube:
      kernel_kind = "fused_16_real_cube";
      break;
    case KernelKind::Fused32Column:
      kernel_kind = "fused_32_column";
      break;
    case KernelKind::Fused2D:
      kernel_kind = "fused_2d";
      break;
    default:
      throw std::runtime_error("JIT backend does not support kernel kind: " + kernel_kind_name(key.kind));
  }
  const std::string id = request_id(cache_key);
  const auto request_dir = out_dir() / "requests" / id;
  // Keep the lock outside the output directory so it can survive a failed
  // generation or cache cleanup without changing the inode being locked.
  KernelFileLock request_lock(out_dir() / ".locks" / (id + ".lock"));
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.cache.find(cache_key);
    if (it != state.cache.end()) {
      ++state.hits;
      return it->second;
    }
    ++state.misses;
  }
  std::ostringstream jit_command;
  if (ix_bluestein_cta) {
    for (const auto &[name, value] : kIxBluesteinCtaDefaults) {
      if (!performance_env(name)) jit_command << name << "=" << value << " ";
    }
  }
  if (ix_packed_real_tle) {
    for (const auto &[name, value] : kIxPackedRealTleDefaults) {
      if (!performance_env(name)) jit_command << name << "=" << value << " ";
    }
    if (!performance_env("FLAGFFT_IX_WARPS")) jit_command << "FLAGFFT_IX_WARPS=" << key.num_warps << " ";
    if (!performance_env("FLAGFFT_IX_SMEM_SWIZZLE_SHIFT")) {
      jit_command << "FLAGFFT_IX_SMEM_SWIZZLE_SHIFT=" << ix_packed_smem_shift << " ";
    }
  }
  const bool ix_split_frequency = key.kind == KernelKind::Leaf &&
      (key.perm_form == "ix_split_frequency" || key.perm_form == "ix_split_c2r_frequency");
  if (npu_performance_defaults_active) {
    for (const auto &setting : kNpuPerformanceDefaults) {
      jit_command << setting.name << "=" << shell_quote(performance_env(setting.name)) << " ";
    }
  }
  if (hcu_performance_defaults_active) {
    for (const auto &setting : kHcuPerformanceDefaults) {
      jit_command << setting.name << "=" << shell_quote(performance_env(setting.name)) << " ";
    }
  }
  // These literals belong to this kernel key, not the process-wide policy.
  // They remove runtime layout division without changing the launch ABI.
  if (key.cuda_permute_span > 0 && key.cuda_permute_batch > 0) {
    jit_command << "FLAGFFT_CUDA_PERMUTE_SPAN=" << key.cuda_permute_span << " "
                << "FLAGFFT_CUDA_PERMUTE_BATCH=" << key.cuda_permute_batch << " ";
  }
  if (key.target.rfind("hcu:", 0) == 0) {
    if (key.kind == KernelKind::Transpose3D)
      jit_command << "FLAGFFT_HCU_TRANSPOSE_COMPACT="
                  << (key.perm_form == "compact_spectrum" ? "1 " : "0 ");
    jit_command << "FLAGFFT_HCU_PACKED_CTA_MAX_PACK="
                << (key.hcu_packed_max_pack > 0 ? std::to_string(key.hcu_packed_max_pack) : "auto")
                << " ";
    jit_command << "FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK="
                << (key.hcu_3d_middle_batch_pack > 0
                        ? std::to_string(key.hcu_3d_middle_batch_pack)
                        : std::string("auto"))
                << " ";
  }
  if (key.npu_batch_pack > 0) {
    jit_command << "FLAGFFT_NPU_2D_BATCH_PACK=" << key.npu_batch_pack << " ";
  }
  jit_command << shell_quote(python_executable()) << " " << triton_jit_source_entrypoint() << " --kernel "
              << kernel_kind << " --out-dir " << shell_quote(request_dir.string()) << " --dtype "
              << shell_quote(key.dtype) << " --target " << shell_quote(key.target) << " --device-profile "
              << shell_quote(device_profile) << " --execution-policy " << shell_quote(policy);
  if (ix_ct_single_policy_) jit_command << " --ix-ct-single";
  if (ix_ct_batch_policy_) jit_command << " --ix-ct-batch";
  if (ix_real_single_pack_) jit_command << " --ix-real-single-pack";
  if (ix_prime_3d_policy_) jit_command << " --ix-prime-3d";
  if (ix_fourstep_preset_) jit_command << " --ix-fourstep-preset " << ix_fourstep_preset_;
  if (key.ix_warp_fft_pack > 0) jit_command << " --ix-warp-fft-pack " << key.ix_warp_fft_pack;
  if (ix_ct_single_tle_policy_) jit_command << " --ix-ct-single-tle " << ix_ct_single_tle_policy_;
  if (musa_1d_batch_policy_) jit_command << " --musa-1d-batch";
#if defined(BACKEND_MACA)
  jit_command << " --compile-script "
              << shell_quote((triton_jit::get_script_dir() / "standalone_compile.py").string());
  if (maca_1d_single_policy_) {
    jit_command << " --maca-1d-single";
  }
  if (maca_1d_batch_policy_) {
    jit_command << " --maca-1d-batch";
  }
  jit_command << " --maca-tail-mode " << shell_quote(tail_mode);
  if (maca_2d_single_policy_) {
    jit_command << " --maca-2d-single";
  }
  if (maca_3d_policy_) {
    jit_command << " --maca-3d";
  }
  if (maca_3d_c2c32_single_cube_policy_) {
    jit_command << " --maca-3d-c2c32-single-cube";
  }
#endif
  if (key.kind == KernelKind::Leaf || key.kind == KernelKind::LeafStrided ||
      key.kind == KernelKind::LeafPermutedStore ||
      key.kind == KernelKind::LeafStridedPermutedStore ||
      key.kind == KernelKind::LeafR2C || key.kind == KernelKind::LeafR2CPermutedStore ||
      key.kind == KernelKind::LeafPackedR2C ||
      key.kind == KernelKind::LeafC2R ||
      key.kind == KernelKind::LeafPackedC2R ||
      key.kind == KernelKind::LeafBluestein || key.kind == KernelKind::LeafRaderFull ||
      key.kind == KernelKind::LeafRaderPrepare || key.kind == KernelKind::LeafRaderFinish ||
      key.kind == KernelKind::LeafBluesteinPrepare ||
      key.kind == KernelKind::LeafBluesteinFinish || key.kind == KernelKind::BluesteinFourStepPrepareRow ||
      key.kind == KernelKind::BluesteinFourStepPointwiseRow ||
      key.kind == KernelKind::BluesteinFourStepFinishCol || key.kind == KernelKind::FourStepRow ||
      key.kind == KernelKind::FourStepRowStrided || key.kind == KernelKind::FourStepRealRow ||
      key.kind == KernelKind::FourStepHermitianRow || key.kind == KernelKind::FourStepCol ||
      key.kind == KernelKind::FourStepColStrided || key.kind == KernelKind::FourStepR2CCol ||
      key.kind == KernelKind::FourStepC2RCol) {
    jit_command << " --length " << key.length << " --factors " << shell_quote(join_ints(key.factors))
                << " --lanes " << key.lanes << " --num-warps " << key.num_warps << " --generic-radices "
                << shell_quote(join_ints(key.generic_radices)) << " --smem-size " << key.smem_size
                << " --direction " << shell_quote(key.direction);
    if (key.npu_portable_leaf) {
      jit_command << " --npu-portable-leaf";
    }
  }
  if (ix_split_frequency || key.kind == KernelKind::LeafPermutedStore ||
      key.kind == KernelKind::LeafStridedPermutedStore ||
      key.kind == KernelKind::LeafR2CPermutedStore ||
      (key.kind == KernelKind::LeafPackedR2C && key.perm_form == "outer_packed")) {
    jit_command << " --perm-form " << shell_quote(key.perm_form);
    if (key.permutation_span > 0) jit_command << " --permutation-span " << key.permutation_span;
    if (key.permutation_batch > 0) jit_command << " --permutation-batch " << key.permutation_batch;
  }
  if (key.hcu_full_smem) jit_command << " --hcu-3d-full-smem";
  if (key.kind == KernelKind::DirectDft || key.kind == KernelKind::DirectDftStrided ||
      key.kind == KernelKind::DirectDftCube || key.kind == KernelKind::DirectDftCubeTransposed ||
      key.kind == KernelKind::DirectDftCubeStrided || key.kind == KernelKind::DirectDftCube2DRow ||
      key.kind == KernelKind::DirectDftCube2DCol ||
      key.kind == KernelKind::DirectDftR2C || key.kind == KernelKind::DirectDftC2R) {
    jit_command << " --length " << key.length << " --direction " << shell_quote(key.direction);
  }
  if (key.kind == KernelKind::StockhamStage) {
    jit_command << " --length " << key.length << " --factors " << shell_quote(join_ints(key.factors))
                << " --direction " << shell_quote(key.direction);
    if (key.perm_form == "tle_aiv") jit_command << " --npu-stockham-aiv";
  }
  if (key.kind == KernelKind::FourStepRow || key.kind == KernelKind::FourStepRowStrided ||
      key.kind == KernelKind::FourStepRealRow || key.kind == KernelKind::FourStepHermitianRow ||
      key.kind == KernelKind::FourStepCol || key.kind == KernelKind::FourStepColStrided ||
      key.kind == KernelKind::FourStepR2CCol || key.kind == KernelKind::FourStepC2RCol ||
      key.kind == KernelKind::BluesteinFourStepPrepareRow ||
      key.kind == KernelKind::BluesteinFourStepPointwiseRow ||
      key.kind == KernelKind::BluesteinFourStepFinishCol) {
    jit_command << " --four-step-n1 " << key.four_step_n1 << " --four-step-n2 " << key.four_step_n2;
  }
  if (key.kind == KernelKind::LeafBluestein || key.kind == KernelKind::LeafBluesteinPrepare ||
      key.kind == KernelKind::LeafBluesteinFinish || key.kind == KernelKind::BluesteinFourStepPrepareRow ||
      key.kind == KernelKind::BluesteinFourStepPointwiseRow ||
      key.kind == KernelKind::BluesteinFourStepFinishCol || key.kind == KernelKind::BluesteinPrepare ||
      key.kind == KernelKind::BluesteinPointwise || key.kind == KernelKind::BluesteinFinalize) {
    jit_command << " --bluestein-n " << key.bluestein_n << " --bluestein-m " << key.bluestein_m;
  }
  if (key.kind == KernelKind::LeafRaderFull || key.kind == KernelKind::LeafRaderPrepare ||
      key.kind == KernelKind::LeafRaderFinish || key.kind == KernelKind::RaderPrepare ||
      key.kind == KernelKind::RaderPointwise ||
      key.kind == KernelKind::RaderFinalize) {
    jit_command << " --rader-n " << key.rader_n << " --rader-m " << key.rader_m;
  }
  if (key.kind == KernelKind::ReshapePack || key.kind == KernelKind::TwiddleReshapePack ||
      key.kind == KernelKind::TiledTranspose) {
    jit_command << " --reshape-n1 " << key.reshape_n1 << " --reshape-n2 " << key.reshape_n2;
  }
  if (key.kind == KernelKind::TiledTranspose && key.perm_form == "npu_table") {
    if (key.target.rfind("npu:", 0) != 0 || key.dtype != "complex64") {
      throw std::runtime_error("NPU vector transpose requires NPU complex64");
    }
    jit_command << " --npu-vector-transpose";
  }
  if (key.kind == KernelKind::Hcu3DCTFirst || key.kind == KernelKind::Hcu3DCTLast) {
    jit_command << " --transpose3d-n0 " << key.transpose3d_n0
                << " --transpose3d-n1 " << key.transpose3d_n1
                << " --transpose3d-n2 " << key.transpose3d_n2
                << " --four-step-n1 " << key.four_step_n1
                << " --four-step-n2 " << key.four_step_n2
                << " --direction " << shell_quote(key.direction);
  }
  if (key.kind == KernelKind::Transpose3D) {
    jit_command << " --transpose3d-n0 " << key.transpose3d_n0 << " --transpose3d-n1 " << key.transpose3d_n1
                << " --transpose3d-n2 " << key.transpose3d_n2 << " --transpose3d-order "
                << shell_quote(key.transpose3d_order);
    if (key.perm_form == "npu_table_loop") {
      if (key.target.rfind("npu:", 0) != 0 || key.dtype != "complex64") {
        throw std::runtime_error("Table transpose3d requires NPU FP32");
      }
      jit_command << " --npu-table-transpose3d";
    }
  }
  if (key.kind == KernelKind::Fused16Plane || key.kind == KernelKind::Fused32Plane ||
      key.kind == KernelKind::Fused16RealPlane || key.kind == KernelKind::Fused32RealPlane ||
      key.kind == KernelKind::FusedRectPlane || key.kind == KernelKind::Fused16Cube ||
      key.kind == KernelKind::Fused16RealCube || key.kind == KernelKind::Fused32Column ||
      key.kind == KernelKind::Fused2D) {
    jit_command << " --direction " << shell_quote(key.direction);
  }
  if (key.kind == KernelKind::Fused16Plane || key.kind == KernelKind::Fused32RealPlane ||
      key.kind == KernelKind::Fused16Cube) {
    jit_command << " --length " << key.length;
  }
  if (key.kind == KernelKind::FusedRectPlane) {
    jit_command << " --fused-plane-n0 " << key.fused_plane_n0
                << " --fused-plane-n1 " << key.fused_plane_n1
                << " --fused-plane-middle " << key.fused_plane_middle;
  }
  if (key.kind == KernelKind::Fused32Column) {
    jit_command << " --length " << key.length;
  }
  if (key.kind == KernelKind::Fused2D) {
    jit_command << " --length " << key.length;
    if (key.perm_form == "transpose") {
      jit_command << " --fused-2d-transpose";
    }
  }
  if (key.kind == KernelKind::RealPairR2CPack || key.kind == KernelKind::RealPairR2CUnpack ||
      key.kind == KernelKind::RealPairC2RPack || key.kind == KernelKind::RealPairC2RUnpack ||
      key.kind == KernelKind::RealToComplex || key.kind == KernelKind::R2CHalfPack ||
      key.kind == KernelKind::R2CPackedPostprocess || key.kind == KernelKind::C2RPackedPreprocess ||
      key.kind == KernelKind::CompactToHermitianFull || key.kind == KernelKind::ComplexToReal) {
    jit_command << " --length " << key.length;
  }

  if (key.kind == KernelKind::NpuGroupedFFT64) {
    if (key.target.rfind("npu:", 0) != 0 || key.dtype != "complex64") {
      throw std::runtime_error("Grouped FFT64 body requires NPU FP32");
    }
    jit_command << " --npu-fft-group " << key.lanes
                << " --npu-fft-stride " << key.reshape_n1
                << " --npu-fft-mode " << key.reshape_n2
                << " --length " << key.length
                << " --direction " << shell_quote(key.direction);
    if (key.perm_form == "radix4_pair") jit_command << " --npu-fft-radix4-pair";
    if (key.perm_form == "npu_matrix_pack") jit_command << " --npu-matrix-phase pack";
    if (key.perm_form == "npu_matrix_cube") jit_command << " --npu-matrix-phase cube";
    if (key.perm_form == "structured_packed_real") jit_command << " --npu-packed-real";
    if (key.perm_form == "structured_tensor") {
      if (key.factors.size() != 2 && key.factors.size() != 3)
        throw std::runtime_error("Tensor FFT key requires two or three axes");
      jit_command << " --npu-tensor-shape " << shell_quote(join_ints(key.factors));
    }
    if (key.perm_form == "structured_power2" || key.perm_form == "structured_packed_real" ||
        key.perm_form == "structured_recursive_power2") {
      if (key.factors.size() != 1) throw std::runtime_error("Grouped FFT key requires an output stride");
      jit_command << " --npu-power2 --npu-fft-output-stride " << key.factors.front();
      if (key.perm_form == "structured_recursive_power2") jit_command << " --npu-recursive-power2";
    }
  }

  if (key.perm_form=="npu_frequency_dft") jit_command << " --npu-frequency-dft";
  if (key.perm_form.rfind("npu_vector_dif_",0)==0) {
    const bool coarse=key.perm_form.rfind("npu_vector_dif_coarse_",0)==0;
    const bool aligned=key.perm_form.rfind("npu_vector_dif_aligned_",0)==0;
    const bool flat=key.perm_form.rfind("npu_vector_dif_flat_power2_",0)==0;
    const bool cube=key.perm_form.rfind("npu_vector_dif_aligned_cube_",0)==0;
    std::string phase=key.perm_form.substr(coarse?22:flat?27:cube?28:aligned?23:15);
    if (phase.size()>=5 && phase.compare(phase.size()-5,5,"_auto")==0)
      phase.resize(phase.size()-5);
    const std::string chain_tag="_matrixchain";
    const auto chain_pos=phase.rfind(chain_tag);
    if (cube && chain_pos!=std::string::npos) {
      const auto columns=phase.substr(chain_pos+chain_tag.size());
      if (columns!="128" && columns!="256")
        throw std::runtime_error("Unsupported matrix-chain columns");
      phase.resize(chain_pos);
      jit_command << " --npu-vector-dif-matrix-chain-columns " << columns;
    }
    const std::string late_batch_tag="_latepairbatch8";
    if (phase.size()>=late_batch_tag.size() &&
        phase.compare(phase.size()-late_batch_tag.size(),late_batch_tag.size(),late_batch_tag)==0) {
      phase.resize(phase.size()-late_batch_tag.size());
      jit_command << " --npu-vector-dif-late-pair-batch 8";
    }
    const std::string late_pair_tag="_latepair91";
    if (phase.size()>=late_pair_tag.size() &&
        phase.compare(phase.size()-late_pair_tag.size(),late_pair_tag.size(),late_pair_tag)==0) {
      phase.resize(phase.size()-late_pair_tag.size());
      jit_command << " --npu-vector-dif-dense-late-pair";
    }
    const std::string terminal_tag="_terminalpair";
    if (phase.size()>=terminal_tag.size() &&
        phase.compare(phase.size()-terminal_tag.size(),terminal_tag.size(),terminal_tag)==0) {
      phase.resize(phase.size()-terminal_tag.size());
      jit_command << " --npu-vector-dif-terminal-pair";
    }
    const std::string basis_tag="_vectorbasis";
    if (phase.size()>=basis_tag.size() &&
        phase.compare(phase.size()-basis_tag.size(),basis_tag.size(),basis_tag)==0) {
      phase.resize(phase.size()-basis_tag.size());
      jit_command << " --npu-vector-dif-vector-basis";
    }
    const std::string finish_tag="_finishbatch4";
    if (phase.size()>=finish_tag.size() &&
        phase.compare(phase.size()-finish_tag.size(),finish_tag.size(),finish_tag)==0) {
      phase.resize(phase.size()-finish_tag.size());
      jit_command << " --npu-vector-dif-finish-batch4";
    }
    if (phase.size()>=13 && phase.compare(phase.size()-13,13,"_codeletreuse")==0) {
      phase.resize(phase.size()-13);
      jit_command << " --npu-vector-dif-codelet-reuse";
    }
    if (flat && phase.size()>=10 && phase.compare(phase.size()-10,10,"_terminal8")==0) {
      phase.resize(phase.size()-10);
      jit_command << " --npu-vector-dif-terminal8";
    }
    const std::string dense_batch_tag="_dense_batch_f";
    const auto dense_batch_pos=phase.rfind(dense_batch_tag);
    if (dense_batch_pos!=std::string::npos) {
      const auto tile=phase.substr(dense_batch_pos+dense_batch_tag.size());
      const auto separator=tile.find("_c");
      if (separator==std::string::npos)
        throw std::runtime_error("Malformed batch dense-prefix tile key");
      const auto frequencies=tile.substr(0,separator), columns=tile.substr(separator+2);
      if ((frequencies!="32" && frequencies!="64" && frequencies!="128") ||
          (columns!="16" && columns!="32" && columns!="64"))
        throw std::runtime_error("Unsupported batch dense-prefix tile key");
      phase.resize(dense_batch_pos);
      jit_command << " --npu-vector-dif-dense-prefix"
                  << " --npu-vector-dif-dense-prefix-columns " << columns
                  << " --npu-vector-dif-dense-prefix-frequencies " << frequencies;
    } else if (phase.size()>=12 && phase.compare(phase.size()-12,12,"_dense_short")==0) {
      phase.resize(phase.size()-12);
      jit_command << " --npu-vector-dif-dense-prefix";
    }
    if (phase.size()>=8 && phase.compare(phase.size()-8,8,"_pitch32")==0) {
      phase.resize(phase.size()-8);
      jit_command << " --npu-vector-dif-prefix-pitch 32";
    }
    if (phase.size()>=7 && phase.compare(phase.size()-7,7,"_half17")==0) {
      phase.resize(phase.size()-7);
      jit_command << " --npu-vector-dif-matrix-first17";
    }
    if (phase.size()>=11 && phase.compare(phase.size()-11,11,"_globalconj")==0) {
      phase.resize(phase.size()-11);
      jit_command << " --npu-vector-dif-global-inverse";
    }
    const std::string first_phase_tag="_firstphase";
    if (flat && phase.size()>=first_phase_tag.size() &&
        phase.compare(phase.size()-first_phase_tag.size(),first_phase_tag.size(),first_phase_tag)==0) {
      phase.resize(phase.size()-first_phase_tag.size());
      jit_command << " --npu-vector-dif-power-first-phase";
    }
    const std::string first_batch_tag="_firstbatch";
    const auto first_batch_pos=phase.rfind(first_batch_tag);
    if (flat && first_batch_pos!=std::string::npos) {
      const auto group=phase.substr(first_batch_pos+first_batch_tag.size());
      if (group!="8" && group!="16")
        throw std::runtime_error("Unsupported power first-axis batch group");
      phase.resize(first_batch_pos);
      jit_command << " --npu-vector-dif-power-first-batch " << group;
    }
    const std::string middle_tag="_middle8";
    if (flat && phase.size()>=middle_tag.size() &&
        phase.compare(phase.size()-middle_tag.size(),middle_tag.size(),middle_tag)==0) {
      phase.resize(phase.size()-middle_tag.size());
      jit_command << " --npu-vector-dif-power-middle-radix8";
    }
    const std::string cube_batch_tag="_cubebatch";
    const auto cube_batch_pos=phase.rfind(cube_batch_tag);
    if (flat && cube_batch_pos!=std::string::npos) {
      const auto group=phase.substr(cube_batch_pos+cube_batch_tag.size());
      if (group!="8" && group!="32")
        throw std::runtime_error("Unsupported power Cube batch group");
      phase.resize(cube_batch_pos);
      jit_command << " --npu-vector-dif-power-cube-batch " << group;
    }
    if (flat && phase.size()>=11 && phase.compare(phase.size()-11,11,"_cubefinish")==0) {
      phase.resize(phase.size()-11);
      jit_command << " --npu-vector-dif-power-cube-finish";
    }
    if (flat && phase.size()>=16 && phase.compare(phase.size()-16,16,"_workspacepad512")==0) {
      phase.resize(phase.size()-16);
      jit_command << " --npu-vector-dif-workspace-padding 512";
    }
    const std::string natural_tag="_natural126";
    if (coarse && phase.size()>=natural_tag.size() &&
        phase.compare(phase.size()-natural_tag.size(),natural_tag.size(),natural_tag)==0) {
      phase.resize(phase.size()-natural_tag.size());
      jit_command << " --npu-vector-dif-natural-prefix";
    }
    if (coarse && phase.size()>=6 && phase.compare(phase.size()-6,6,"_dense")==0) {
      phase.resize(phase.size()-6);
      jit_command << " --npu-vector-dif-coarse-dense";
    }
    if (coarse && phase.size()>=3 && phase.compare(phase.size()-3,3,"_gt")==0) {
      phase.resize(phase.size()-3);
      jit_command << " --npu-vector-dif-coarse-gt";
    }
    if (coarse && phase.size()>=7 && phase.compare(phase.size()-7,7,"_single")==0) {
      phase.resize(phase.size()-7);
      jit_command << " --npu-vector-dif-coarse-single";
    }
    if (phase.size()>=10 && phase.compare(phase.size()-10,10,"_batchpack")==0) {
      phase.resize(phase.size()-10);
      jit_command << " --npu-vector-dif-batch-pack";
    }
    if (phase.size()>=9 && phase.compare(phase.size()-9,9,"_vector17")==0) {
      phase.resize(phase.size()-9);
      jit_command << " --npu-vector-dif-vector-first17";
    }
    if (phase.size()>=9 && phase.compare(phase.size()-9,9,"_vector15")==0) {
      phase.resize(phase.size()-9);
      jit_command << " --npu-vector-dif-vector-second15";
    }
    jit_command << " --npu-vector-dif-phase " << shell_quote(phase);
    if (aligned) jit_command << " --npu-vector-dif-aligned";
    if (cube) jit_command << " --npu-vector-dif-cube-prefix";
    if (flat) jit_command << " --npu-vector-dif-flat-power2";
    if (coarse) jit_command << " --npu-vector-dif-coarse";
  }
  if (key.perm_form.rfind("npu_matrix_fourstep_",0)==0) {
    const std::string phase=key.perm_form.substr(20);
    const bool wide=phase.size()>5 && phase.substr(phase.size()-5)=="_wide";
    jit_command << " --npu-matrix-fourstep-phase " << shell_quote(wide?phase.substr(0,phase.size()-5):phase);
    if (wide) jit_command << " --npu-matrix-fourstep-wide";
  }
  if (key.perm_form == "npu_packed_mixed_planar") jit_command << " --npu-packed-mixed-axis";
  const auto structured_pack=key.perm_form.find("_pack_");
  std::string structured_form=key.perm_form.substr(0,structured_pack);
  const bool structured_global=structured_form.size()>=11 &&
      structured_form.compare(structured_form.size()-11,11,"_globalconj")==0;
  if (structured_global) structured_form.resize(structured_form.size()-11);
  if (structured_form=="npu_structured" || structured_form=="npu_structured_planar" ||
      structured_form=="npu_structured_recursive_axis_planar") {
    jit_command << " --npu-structured-fft";
    if (structured_global) jit_command << " --npu-structured-global-inverse";
    if (structured_form=="npu_structured_recursive_axis_planar")
      jit_command << " --npu-structured-recursive";
    if (structured_pack!=std::string::npos)
      jit_command << " --npu-structured-pack " << shell_quote(key.perm_form.substr(structured_pack+6));
  }

  std::string artifact_json = run_command_capture_stdout(jit_command.str());
  auto kernel = std::make_shared<JitKernel>();
  kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
  kernel->module_path = json_string_field(artifact_json, "module_path");
#if defined(BACKEND_MACA)
  kernel->binary_dir = json_string_field(artifact_json, "binary_dir");
#endif
  if (adaptor::backend_name() == "npu") {
    if (key.perm_form.rfind("npu_vector_dif_aligned_cube_",0)==0)
      cache_key += ";npu-dif-cube-prefix-v2";
    const char *join = performance_env("FLAGFFT_NPU_JOIN_MODE");
    kernel->npu_join_lowering = join != nullptr ? join : "auto";
    if (kernel->npu_join_lowering != "auto" && kernel->npu_join_lowering != "native" &&
        kernel->npu_join_lowering != "packed")
      throw std::runtime_error("FLAGFFT_NPU_JOIN_MODE must be auto, native or packed");
    kernel->npu_auto_blockify = npu_auto_blockify_for(key);
  }
  kernel->signature = json_string_field(artifact_json, "signature");
  kernel->num_warps = json_int_field(artifact_json, "num_warps");
  kernel->warp_size = json_int_field(artifact_json, "warp_size");
  kernel->profile_id = json_string_field(artifact_json, "profile_id");
  kernel->num_stages = json_int_field(artifact_json, "num_stages");
  kernel->npu_single_buffer = adaptor::backend_name() == "npu" && kernel->num_stages == 1 &&
      ((kernel->kernel_name == "dense_prefix_half" &&
        key.perm_form.find("_dense_batch_f") != std::string::npos) ||
       kernel->kernel_name == "coarse_dense_prefix" ||
       kernel->kernel_name == "matrix_chain_finish_batch" ||
       kernel->kernel_name == "power_cube_finish" ||
       kernel->kernel_name == "power_cube_finish_batch");
  kernel->batch_per_block = json_int_field(artifact_json, "batch_per_block");
  if (ix_split_frequency) {
    if (kernel->batch_per_block != 1 ||
        kernel->kernel_name.find("_split_frequency") == std::string::npos)
      throw std::runtime_error("invalid IX split-frequency leaf");
    kernel->grid_x_override = 2;
  }
  if (key.kind == KernelKind::LeafStridedPermutedStore &&
      (kernel->kernel_name == "ix_prime_r2c_last" ||
       (kernel->kernel_name.rfind("hcu_warp", 0) == 0 &&
        kernel->kernel_name.find("_stridedinput") != std::string::npos))) {
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
    if (kernel->grid_x_override <= 0)
      throw std::runtime_error("strided wave requires a positive tile-derived launch grid");
  }
  if (key.kind == KernelKind::Hcu3DCTFirst || key.kind == KernelKind::Hcu3DCTLast) {
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
    if (kernel->grid_x_override <= 0) {
      throw std::runtime_error("HCU CT requires a positive, tile-derived launch grid");
    }
  }
  if (key.kind == KernelKind::DirectDftCubeTransposed ||
      key.kind == KernelKind::DirectDftCube2DRow || key.kind == KernelKind::DirectDftCube2DCol) {
    kernel->grid_y_override = json_int_field(artifact_json, "grid_y_override");
  }
  if (key.kind == KernelKind::StockhamStage) {
    kernel->butterflies_per_block = json_int_field(artifact_json, "butterflies_per_block");
  }
  if (key.kind == KernelKind::Transpose3D) {
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
    kernel->npu_permutation_table = key.perm_form == "npu_table_loop";
  }
  if (key.kind == KernelKind::TiledTranspose && key.perm_form == "npu_table") {
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
    kernel->grid_y_override = json_int_field(artifact_json, "grid_y_override");
    kernel->transpose_tile_size = json_int_field(artifact_json, "tile_size");
  }
  if (key.kind == KernelKind::RealPairR2CPack || key.kind == KernelKind::RealPairR2CUnpack ||
      key.kind == KernelKind::RealPairC2RPack || key.kind == KernelKind::RealPairC2RUnpack ||
      key.kind == KernelKind::RealToComplex || key.kind == KernelKind::R2CHalfPack ||
      key.kind == KernelKind::R2CPackedPostprocess || key.kind == KernelKind::C2RPackedPreprocess ||
      key.kind == KernelKind::CompactToHermitianFull || key.kind == KernelKind::ComplexToReal) {
    kernel->rows_per_block = json_int_field(artifact_json, "rows_per_block");
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
    if (key.kind==KernelKind::R2CPackedPostprocess || key.kind==KernelKind::C2RPackedPreprocess) {
      kernel->column_block=json_int_field(artifact_json,"block");
      if (kernel->column_block<=0) throw std::runtime_error("invalid packed-real column block");
    }
  }
  if (key.kind == KernelKind::FourStepRow || key.kind == KernelKind::FourStepRowStrided ||
      key.kind == KernelKind::FourStepRealRow || key.kind == KernelKind::FourStepHermitianRow ||
      key.kind == KernelKind::FourStepCol || key.kind == KernelKind::FourStepColStrided ||
      key.kind == KernelKind::FourStepR2CCol || key.kind == KernelKind::FourStepC2RCol ||
      key.kind == KernelKind::BluesteinFourStepPrepareRow ||
      key.kind == KernelKind::BluesteinFourStepPointwiseRow ||
      key.kind == KernelKind::BluesteinFourStepFinishCol) {
    kernel->inner_pack = json_int_field(artifact_json, "inner_pack");
    kernel->tle_fused_twiddle = json_bool_field(artifact_json, "tle_fused_twiddle");
  }
  kernel->compile();

  std::lock_guard<std::mutex> lock(state.mutex);
  auto [it, inserted] = state.cache.emplace(cache_key, kernel);
  return inserted ? kernel : it->second;
}

void TritonCompiler::clear_kernel_cache() {
  KernelCacheState &state = kernel_cache_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.cache.clear();
  state.hits = 0;
  state.misses = 0;
}

std::filesystem::path TritonCompiler::out_dir() const {
  return default_cache_dir();
}

std::string TritonCompiler::python_executable() const {
  const char *override_path = std::getenv("FLAGFFT_PYTHON");
  if (override_path != nullptr && std::strlen(override_path) > 0) {
    return override_path;
  }
  return "python3";
}

std::string TritonCompiler::triton_jit_source_entrypoint() const {
  return "-m flagfft_codegen.jit_source";
}

}  // namespace flagfft
