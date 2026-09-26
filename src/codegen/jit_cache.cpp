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

#if defined(BACKEND_MACA)
#include "triton_jit/jit_utils.h"
#endif

namespace flagfft {
namespace {

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
      const char *value = std::getenv(name);
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
  reject_legacy_backend_env();
  const auto device_profile = adaptor::device_capabilities_json();
  const char *policy_env = std::getenv("FLAGFFT_EXECUTION_POLICY");
  const std::string default_policy = adaptor::backend_name() == "ix"
                                         ? "balanced"
                                         : (adaptor::backend_name() == "hcu" ? "native" : "legacy");
  const std::string policy = policy_env ? policy_env : default_policy;
  const auto tail_mode = maca_tail_kernel_mode(maca_tail_policy_, key.kind, key.dtype,
                                               key.length, key.four_step_n1, key.four_step_n2);
  const std::string cache_key = key.repr() + device_profile + policy + ";profile-v1;maca-1d-single=" +
                                (maca_1d_single_policy_ ? "1" : "0") + ";maca-2d-single=" +
                                (maca_2d_single_policy_ ? "1" : "0") + ";ix-ct-single=" +
                                (ix_ct_single_policy_ ? "1" : "0") + ";ix-ct-single-tle=" +
                                std::to_string(ix_ct_single_tle_policy_) + ";ix-ct-batch=" +
                                (ix_ct_batch_policy_ ? "1" : "0") + ";ix-real-single-pack=" +
                                (ix_real_single_pack_ ? "1" : "0") +
                                (adaptor::backend_name() == "maca"
                                     ? maca_tail_codegen_identity(tail_mode) : "");
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
    case KernelKind::LeafR2C:
      kernel_kind = "leaf_r2c";
      break;
    case KernelKind::LeafPackedR2C:
      kernel_kind = "leaf_packed_r2c";
      break;
    case KernelKind::LeafC2R:
      kernel_kind = "leaf_c2r";
      break;
    case KernelKind::LeafBluestein:
      kernel_kind = "leaf_bluestein";
      break;
    case KernelKind::LeafBluesteinPrepare:
      kernel_kind = "leaf_bluestein_prepare";
      break;
    case KernelKind::LeafBluesteinFinish:
      kernel_kind = "leaf_bluestein_finish";
      break;
    case KernelKind::BluesteinFourStepPrepareRow:
      kernel_kind = "bluestein_four_step_prepare_row";
      break;
    case KernelKind::BluesteinFourStepPointwiseRow:
      kernel_kind = "bluestein_four_step_pointwise_row";
      break;
    case KernelKind::BluesteinFourStepFinishCol:
      kernel_kind = "bluestein_four_step_finish_col";
      break;
    case KernelKind::DirectDft:
      kernel_kind = "direct_dft";
      break;
    case KernelKind::DirectDftStrided:
      kernel_kind = "direct_dft_strided";
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
    case KernelKind::Transpose3D:
      kernel_kind = "transpose3d";
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
  jit_command << shell_quote(python_executable()) << " " << triton_jit_source_entrypoint() << " --kernel "
              << kernel_kind << " --out-dir " << shell_quote(request_dir.string()) << " --dtype "
              << shell_quote(key.dtype) << " --target " << shell_quote(key.target) << " --device-profile "
              << shell_quote(device_profile) << " --execution-policy " << shell_quote(policy);
  if (ix_ct_single_policy_) jit_command << " --ix-ct-single";
  if (ix_ct_batch_policy_) jit_command << " --ix-ct-batch";
  if (ix_real_single_pack_) jit_command << " --ix-real-single-pack";
  if (ix_ct_single_tle_policy_) jit_command << " --ix-ct-single-tle " << ix_ct_single_tle_policy_;
#if defined(BACKEND_MACA)
  jit_command << " --compile-script "
              << shell_quote((triton_jit::get_script_dir() / "standalone_compile.py").string());
  if (maca_1d_single_policy_) {
    jit_command << " --maca-1d-single";
  }
  jit_command << " --maca-tail-mode " << shell_quote(tail_mode);
  if (maca_2d_single_policy_) {
    jit_command << " --maca-2d-single";
  }
#endif
  if (key.kind == KernelKind::Leaf || key.kind == KernelKind::LeafStrided ||
      key.kind == KernelKind::LeafPermutedStore ||
      key.kind == KernelKind::LeafR2C || key.kind == KernelKind::LeafPackedR2C ||
      key.kind == KernelKind::LeafC2R ||
      key.kind == KernelKind::LeafBluestein || key.kind == KernelKind::LeafBluesteinPrepare ||
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
  }
  if (key.kind == KernelKind::LeafPermutedStore) {
    jit_command << " --perm-form " << shell_quote(key.perm_form);
  }
  if (key.kind == KernelKind::DirectDft || key.kind == KernelKind::DirectDftStrided ||
      key.kind == KernelKind::DirectDftR2C || key.kind == KernelKind::DirectDftC2R) {
    jit_command << " --length " << key.length << " --direction " << shell_quote(key.direction);
  }
  if (key.kind == KernelKind::StockhamStage) {
    jit_command << " --length " << key.length << " --factors " << shell_quote(join_ints(key.factors))
                << " --direction " << shell_quote(key.direction);
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
  if (key.kind == KernelKind::RaderPrepare || key.kind == KernelKind::RaderPointwise ||
      key.kind == KernelKind::RaderFinalize) {
    jit_command << " --rader-n " << key.rader_n << " --rader-m " << key.rader_m;
  }
  if (key.kind == KernelKind::ReshapePack || key.kind == KernelKind::TwiddleReshapePack ||
      key.kind == KernelKind::TiledTranspose) {
    jit_command << " --reshape-n1 " << key.reshape_n1 << " --reshape-n2 " << key.reshape_n2;
  }
  if (key.kind == KernelKind::Transpose3D) {
    jit_command << " --transpose3d-n0 " << key.transpose3d_n0 << " --transpose3d-n1 " << key.transpose3d_n1
                << " --transpose3d-n2 " << key.transpose3d_n2 << " --transpose3d-order "
                << shell_quote(key.transpose3d_order);
  }
  if (key.kind == KernelKind::RealToComplex || key.kind == KernelKind::R2CHalfPack ||
      key.kind == KernelKind::R2CPackedPostprocess || key.kind == KernelKind::C2RPackedPreprocess ||
      key.kind == KernelKind::CompactToHermitianFull || key.kind == KernelKind::ComplexToReal) {
    jit_command << " --length " << key.length;
  }

  std::string artifact_json = run_command_capture_stdout(jit_command.str());
  auto kernel = std::make_shared<JitKernel>();
  kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
  kernel->module_path = json_string_field(artifact_json, "module_path");
#if defined(BACKEND_MACA)
  kernel->binary_dir = json_string_field(artifact_json, "binary_dir");
#endif
  kernel->signature = json_string_field(artifact_json, "signature");
  kernel->num_warps = json_int_field(artifact_json, "num_warps");
  kernel->warp_size = json_int_field(artifact_json, "warp_size");
  kernel->profile_id = json_string_field(artifact_json, "profile_id");
  kernel->num_stages = json_int_field(artifact_json, "num_stages");
  kernel->batch_per_block = json_int_field(artifact_json, "batch_per_block");
  if (key.kind == KernelKind::StockhamStage) {
    kernel->butterflies_per_block = json_int_field(artifact_json, "butterflies_per_block");
  }
  if (key.kind == KernelKind::Transpose3D) {
    kernel->grid_x_override = json_int_field(artifact_json, "grid_x_override");
  }
  if (key.kind == KernelKind::RealToComplex || key.kind == KernelKind::R2CHalfPack ||
      key.kind == KernelKind::R2CPackedPostprocess || key.kind == KernelKind::C2RPackedPreprocess ||
      key.kind == KernelKind::CompactToHermitianFull || key.kind == KernelKind::ComplexToReal) {
    kernel->rows_per_block = json_int_field(artifact_json, "rows_per_block");
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
