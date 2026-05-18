#include "flagfft/core.hpp"

namespace flagfft {
namespace {

RuntimeKernelBackend configured_kernel_backend() {
    const char *env = std::getenv("FLAGFFT_KERNEL_BACKEND");
    std::string backend =
        (env != nullptr && std::strlen(env) > 0) ? env : FLAGFFT_KERNEL_BACKEND_DEFAULT;
    if (backend == "AOT") {
        return RuntimeKernelBackend::Aot;
    }
    if (backend == "LIBTRITON_JIT") {
#if defined(FLAGFFT_ENABLE_LIBTRITON_JIT)
        return RuntimeKernelBackend::LibTritonJit;
#else
        throw std::runtime_error(
            "FLAGFFT_KERNEL_BACKEND=LIBTRITON_JIT requires a build with FLAGFFT_ENABLE_LIBTRITON_JIT=ON");
#endif
    }
    throw std::runtime_error("unsupported FLAGFFT_KERNEL_BACKEND: " + backend);
}

std::string run_command_capture_stdout(const std::string &command) {
    std::array<char, 4096> buffer{};
    std::string output;
    int status = 0;
    auto run_pipe = [&]() {
        nb::gil_scoped_release release;
        FILE *pipe = popen(command.c_str(), "r");
        if (pipe == nullptr) {
            throw std::runtime_error("failed to start command: " + command);
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output += buffer.data();
        }
        status = pclose(pipe);
    };
    auto run_pipe_without_gil = [&]() {
        FILE *pipe = popen(command.c_str(), "r");
        if (pipe == nullptr) {
            throw std::runtime_error("failed to start command: " + command);
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output += buffer.data();
        }
        status = pclose(pipe);
    };
    if (Py_IsInitialized()) {
        run_pipe();
    } else {
        run_pipe_without_gil();
    }
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

std::vector<unsigned char> hex_to_bytes(const std::string &hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex string has odd length");
    }
    std::vector<unsigned char> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        unsigned int value = 0;
        std::stringstream stream;
        stream << std::hex << hex.substr(i, 2);
        stream >> value;
        bytes.push_back(static_cast<unsigned char>(value));
    }
    return bytes;
}

struct KernelCacheState {
    std::mutex mutex;
    std::unordered_map<KernelKey, std::shared_ptr<AotKernel>, KernelKeyHash> cache;
    int64_t hits = 0;
    int64_t misses = 0;
};

KernelCacheState &kernel_cache_state() {
    static KernelCacheState *state = new KernelCacheState();
    return *state;
}

}  // namespace

std::shared_ptr<AotKernel> TritonCompiler::compile_kernel(const KernelKey &key, const std::string &command) const {
    if (configured_kernel_backend() == RuntimeKernelBackend::LibTritonJit &&
        (key.kind == KernelKind::Leaf || key.kind == KernelKind::FourStepRow ||
         key.kind == KernelKind::FourStepCol)) {
        return compile_jit_kernel(key, command);
    }
    return compile_aot_kernel(key, command);
}

std::shared_ptr<AotKernel> TritonCompiler::compile_aot_kernel(const KernelKey &key,
                                                              const std::string &command) const {
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

    std::string artifact_json = run_command_capture_stdout(command);
    auto kernel = std::make_shared<AotKernel>();
    kernel->backend = RuntimeKernelBackend::Aot;
    kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
    kernel->cubin = hex_to_bytes(json_string_field(artifact_json, "cubin_hex"));
    kernel->shared = json_int_field(artifact_json, "shared");
    kernel->num_warps = json_int_field(artifact_json, "num_warps");
    kernel->num_stages = json_int_field(artifact_json, "num_stages");
    kernel->batch_per_block = json_int_field(artifact_json, "batch_per_block");

    std::lock_guard<std::mutex> lock(state.mutex);
    auto [it, inserted] = state.cache.emplace(key, kernel);
    return inserted ? kernel : it->second;
}

std::shared_ptr<AotKernel> TritonCompiler::compile_jit_kernel(const KernelKey &key,
                                                              const std::string &command) const {
#if !defined(FLAGFFT_ENABLE_LIBTRITON_JIT)
    (void)key;
    (void)command;
    throw std::runtime_error("libtriton_jit backend was not enabled at build time");
#else
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

    (void)command;
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
        default:
            throw std::runtime_error("libtriton_jit backend supports only leaf and fused four-step kernels");
    }
    std::ostringstream jit_command;
    jit_command << shell_quote(python_executable())
                << " " << triton_jit_source_entrypoint()
                << " --kernel " << kernel_kind
                << " --length " << key.length
                << " --factors " << shell_quote(join_ints(key.factors))
                << " --lanes " << key.lanes
                << " --num-warps " << key.num_warps
                << " --generic-radices " << shell_quote(join_ints(key.generic_radices))
                << " --smem-size " << key.smem_size
                << " --direction " << shell_quote(key.direction)
                << " --out-dir " << shell_quote(out_dir().string());
    if (key.kind == KernelKind::FourStepRow || key.kind == KernelKind::FourStepCol) {
        jit_command << " --four-step-n1 " << key.four_step_n1
                    << " --four-step-n2 " << key.four_step_n2;
    }

    std::string artifact_json = run_command_capture_stdout(jit_command.str());
    auto kernel = std::make_shared<AotKernel>();
    kernel->backend = RuntimeKernelBackend::LibTritonJit;
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
#endif
}

void TritonCompiler::clear_kernel_cache() {
    KernelCacheState &state = kernel_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.cache.clear();
    state.hits = 0;
    state.misses = 0;
}

nb::dict TritonCompiler::kernel_cache_info() {
    KernelCacheState &state = kernel_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    nb::dict out;
    out["kernel_size"] = state.cache.size();
    out["kernel_hits"] = state.hits;
    out["kernel_misses"] = state.misses;
    return out;
}

nb::list TritonCompiler::kernel_cache_keys() {
    KernelCacheState &state = kernel_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    nb::list out;
    for (const auto &entry : state.cache) {
        out.append(kernel_key_to_dict(entry.first));
    }
    return out;
}

std::filesystem::path TritonCompiler::out_dir() const {
    return default_cache_dir();
}

std::string TritonCompiler::python_executable() const {
    const char *override_path = std::getenv("FLAGFFT_PYTHON");
    if (override_path != nullptr && std::strlen(override_path) > 0) {
        return override_path;
    }
    if (!Py_IsInitialized()) {
        return "python3";
    }
    std::wstring executable_w = Py_GetProgramFullPath();
    if (executable_w.empty()) {
        return "python3";
    }
    return std::string(executable_w.begin(), executable_w.end());
}

std::string TritonCompiler::triton_aot_entrypoint() const {
    std::filesystem::path local_script =
        std::filesystem::current_path() / "src" / "triton_aot.py";
    if (std::filesystem::exists(local_script)) {
        return shell_quote(local_script.string());
    }
    return "-m src.triton_aot";
}

std::string TritonCompiler::triton_jit_source_entrypoint() const {
    std::filesystem::path local_script =
        std::filesystem::current_path() / "src" / "codegen" / "jit_source.py";
    if (std::filesystem::exists(local_script)) {
        return shell_quote(local_script.string());
    }
    return "-m src.codegen.jit_source";
}

}  // namespace flagfft
