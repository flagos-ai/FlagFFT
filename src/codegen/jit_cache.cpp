#include "flagfft/core.hpp"

namespace flagfft {
namespace {

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

  struct KernelCacheState {
    std::mutex mutex;
    std::unordered_map<KernelKey, std::shared_ptr<JitKernel>, KernelKeyHash> cache;
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
  KernelCacheState &state = kernel_cache_state();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.cache.find(key);
    if (it != state.cache.end()) {
      ++state.hits;
      return it->second;
    }
    ++state.misses;
  }

  std::string kernel_kind;
  switch (key.kind) {
    case KernelKind::Leaf:
      kernel_kind = "leaf";
      break;
    case KernelKind::FourStepRow:
      kernel_kind = "four_step_row";
      break;
    case KernelKind::FourStepCol:
      kernel_kind = "four_step_col";
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
    case KernelKind::CompactToHermitianFull:
      kernel_kind = "compact_to_hermitian_full";
      break;
    case KernelKind::ComplexToReal:
      kernel_kind = "complex_to_real";
      break;
    default:
      throw std::runtime_error("JIT backend does not support kernel kind: " + kernel_kind_name(key.kind));
  }
  std::ostringstream jit_command;
  jit_command << shell_quote(python_executable()) << " " << triton_jit_source_entrypoint() << " --kernel "
              << kernel_kind << " --out-dir " << shell_quote(out_dir().string()) << " --dtype "
              << shell_quote(key.dtype);
  if (key.kind == KernelKind::Leaf || key.kind == KernelKind::FourStepRow ||
      key.kind == KernelKind::FourStepCol) {
    jit_command << " --length " << key.length << " --factors " << shell_quote(join_ints(key.factors))
                << " --lanes " << key.lanes << " --num-warps " << key.num_warps << " --generic-radices "
                << shell_quote(join_ints(key.generic_radices)) << " --smem-size " << key.smem_size
                << " --direction " << shell_quote(key.direction);
  }
  if (key.kind == KernelKind::FourStepRow || key.kind == KernelKind::FourStepCol) {
    jit_command << " --four-step-n1 " << key.four_step_n1 << " --four-step-n2 " << key.four_step_n2;
  }
  if (key.kind == KernelKind::BluesteinPrepare || key.kind == KernelKind::BluesteinPointwise ||
      key.kind == KernelKind::BluesteinFinalize) {
    jit_command << " --bluestein-n " << key.bluestein_n << " --bluestein-m " << key.bluestein_m;
  }
  if (key.kind == KernelKind::ReshapePack || key.kind == KernelKind::TwiddleReshapePack) {
    jit_command << " --reshape-n1 " << key.reshape_n1 << " --reshape-n2 " << key.reshape_n2;
  }
  if (key.kind == KernelKind::RealToComplex || key.kind == KernelKind::R2CHalfPack ||
      key.kind == KernelKind::CompactToHermitianFull || key.kind == KernelKind::ComplexToReal) {
    jit_command << " --length " << key.length;
  }

  std::string artifact_json = run_command_capture_stdout(jit_command.str());
  auto kernel = std::make_shared<JitKernel>();
  kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
  kernel->module_path = json_string_field(artifact_json, "module_path");
  kernel->signature = json_string_field(artifact_json, "signature");
  kernel->num_warps = json_int_field(artifact_json, "num_warps");
  kernel->num_stages = json_int_field(artifact_json, "num_stages");
  kernel->batch_per_block = json_int_field(artifact_json, "batch_per_block");
  kernel->compile();

  std::lock_guard<std::mutex> lock(state.mutex);
  auto [it, inserted] = state.cache.emplace(key, kernel);
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
