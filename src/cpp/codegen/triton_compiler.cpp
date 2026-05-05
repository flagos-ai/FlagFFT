#include "flagfft/core.hpp"

namespace flagfft {
namespace {

std::string run_command_capture_stdout(const std::string &command) {
    std::array<char, 4096> buffer{};
    std::string output;
    int status = 0;
    {
        nb::gil_scoped_release release;
        FILE *pipe = popen(command.c_str(), "r");
        if (pipe == nullptr) {
            throw std::runtime_error("failed to start command: " + command);
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output += buffer.data();
        }
        status = pclose(pipe);
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

}  // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage) {
    std::vector<int64_t> prev_freq;
    int64_t rem = codelet;
    for (int64_t axis = 0; axis < stage; ++axis) {
        prev_freq.push_back(rem % radices[static_cast<std::size_t>(axis)]);
        rem /= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<int64_t> future_time(radices.size(), 0);
    for (int64_t axis = static_cast<int64_t>(radices.size()) - 1; axis > stage; --axis) {
        future_time[static_cast<std::size_t>(axis)] = rem % radices[static_cast<std::size_t>(axis)];
        rem /= radices[static_cast<std::size_t>(axis)];
    }
    if (rem != 0) {
        throw std::runtime_error("stage codelet decode overflow");
    }
    return {prev_freq, future_time};
}

int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit) {
    int64_t value = 0;
    int64_t stride = 1;
    for (std::size_t axis = 0; axis < limit; ++axis) {
        value += digits[axis] * stride;
        stride *= radices[axis];
    }
    return value;
}

std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(
    const std::vector<int64_t> &radices, int64_t stage, int64_t lanes) {
    int64_t n = product(radices);
    int64_t elems_per_lane = n / lanes;
    int64_t radix = radices[static_cast<std::size_t>(stage)];
    int64_t denom = 1;
    for (int64_t axis = 0; axis <= stage; ++axis) {
        denom *= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<float> tw_r(static_cast<std::size_t>(n));
    std::vector<float> tw_i(static_cast<std::size_t>(n));
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t elem = 0; elem < elems_per_lane; ++elem) {
            int64_t group = elem / radix;
            int64_t digit = elem % radix;
            int64_t codelet = lane + lanes * group;
            auto decoded = decode_stage_codelet(codelet, radices, stage);
            int64_t prefix = mixed_radix_value(decoded.first, radices, decoded.first.size());
            double angle = -2.0 * kPi * static_cast<double>(prefix * digit) / static_cast<double>(denom);
            std::size_t index = static_cast<std::size_t>(lane + lanes * elem);
            tw_r[index] = static_cast<float>(std::cos(angle));
            tw_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {tw_r, tw_i};
}

std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix) {
    std::vector<float> dft_r(static_cast<std::size_t>(radix * radix));
    std::vector<float> dft_i(static_cast<std::size_t>(radix * radix));
    for (int64_t k = 0; k < radix; ++k) {
        for (int64_t n = 0; n < radix; ++n) {
            double angle = -2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(radix);
            std::size_t index = static_cast<std::size_t>(k * radix + n);
            dft_r[index] = static_cast<float>(std::cos(angle));
            dft_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {dft_r, dft_i};
}

nb::object build_four_step_twiddle_tensor(const FFTRequest &request, int64_t n1, int64_t n2) {
    std::vector<float> real(static_cast<std::size_t>(n1 * n2));
    std::vector<float> imag(static_cast<std::size_t>(n1 * n2));
    for (int64_t row = 0; row < n2; ++row) {
        for (int64_t col = 0; col < n1; ++col) {
            double angle = -2.0 * kPi * static_cast<double>(row * col) /
                           static_cast<double>(n1 * n2);
            std::size_t index = static_cast<std::size_t>(row * n1 + col);
            real[index] = static_cast<float>(std::cos(angle));
            imag[index] = static_cast<float>(std::sin(angle));
        }
    }
    return tensor_from_complex_vectors(real, imag, request, nb::make_tuple(n2, n1));
}

std::shared_ptr<CompiledNode> TritonCompiler::compile_node(const PlanNodePtr &node,
                                                           const FFTRequest &request,
                                                           int64_t batch) {
    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        return compile_leaf(*leaf, request);
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        std::shared_ptr<CompiledNode> row =
            compile_node(four_step->row_plan, request, batch * four_step->n2);
        std::shared_ptr<CompiledNode> col =
            compile_node(four_step->col_plan, request, batch * four_step->n1);
        nb::object twiddle = build_four_step_twiddle_tensor(request, four_step->n1, four_step->n2);
        nb::object stage0 =
            empty_complex64_tensor(request, nb::make_tuple(batch, four_step->n2, four_step->n1));
        nb::object stage2 =
            empty_complex64_tensor(request, nb::make_tuple(batch, four_step->n1, four_step->n2));
        return std::make_shared<CompiledFourStepNode>(
            four_step->length, four_step->n1, four_step->n2, std::move(row), std::move(col),
            compile_transpose_kernel(request), compile_twiddle_transpose_kernel(request),
            std::move(twiddle), std::move(stage0), std::move(stage2));
    }
    raise_python(PyExc_NotImplementedError,
                 "flagfft.fft C++ AOT backend does not support plan node kind: " + node->kind);
}

std::shared_ptr<CompiledNode> TritonCompiler::compile_leaf(const LeafPlanNode &leaf, const FFTRequest &request) {
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel leaf"
            << " --length " << leaf.length
            << " --factors " << shell_quote(join_ints(leaf.factors))
            << " --lanes " << leaf.lanes
            << " --num-warps " << leaf.num_warps
            << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
            << " --smem-size " << leaf.smem_size
            << " --target " << shell_quote(target_for_request(request))
            << " --out-dir " << shell_quote(out_dir().string());

    std::shared_ptr<AotKernel> kernel = compile_kernel_from_command(command.str());

    std::vector<nb::object> tables;
    for (std::size_t stage = 1; stage < leaf.factors.size(); ++stage) {
        auto twiddles = build_stage_twiddles(leaf.factors, static_cast<int64_t>(stage), leaf.lanes);
        tables.push_back(tensor_from_float_vector(twiddles.first, request));
        tables.push_back(tensor_from_float_vector(twiddles.second, request));
    }
    for (int64_t radix : leaf.generic_radices) {
        auto dft = build_dft_matrix(radix);
        tables.push_back(tensor_from_float_vector(dft.first, request));
        tables.push_back(tensor_from_float_vector(dft.second, request));
    }
    return std::make_shared<CompiledLeafNode>(leaf.length, std::move(kernel), std::move(tables));
}

std::shared_ptr<AotKernel> TritonCompiler::compile_transpose_kernel(const FFTRequest &request) {
    if (transpose_kernel != nullptr) {
        return transpose_kernel;
    }
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel transpose"
            << " --target " << shell_quote(target_for_request(request))
            << " --out-dir " << shell_quote(out_dir().string());
    transpose_kernel = compile_kernel_from_command(command.str());
    return transpose_kernel;
}

std::shared_ptr<AotKernel> TritonCompiler::compile_twiddle_transpose_kernel(const FFTRequest &request) {
    if (twiddle_transpose_kernel != nullptr) {
        return twiddle_transpose_kernel;
    }
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel twiddle_transpose"
            << " --target " << shell_quote(target_for_request(request))
            << " --out-dir " << shell_quote(out_dir().string());
    twiddle_transpose_kernel = compile_kernel_from_command(command.str());
    return twiddle_transpose_kernel;
}

std::shared_ptr<AotKernel> TritonCompiler::compile_kernel_from_command(const std::string &command) const {
    static std::mutex kernel_cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<AotKernel>> kernel_cache;

    {
        std::lock_guard<std::mutex> lock(kernel_cache_mutex);
        auto it = kernel_cache.find(command);
        if (it != kernel_cache.end()) {
            return it->second;
        }
    }

    std::string artifact_json = run_command_capture_stdout(command);
    auto kernel = std::make_shared<AotKernel>();
    kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
    kernel->cubin = hex_to_bytes(json_string_field(artifact_json, "cubin_hex"));
    kernel->shared = json_int_field(artifact_json, "shared");
    kernel->num_warps = json_int_field(artifact_json, "num_warps");

    std::lock_guard<std::mutex> lock(kernel_cache_mutex);
    auto [it, inserted] = kernel_cache.emplace(command, kernel);
    return inserted ? kernel : it->second;
}

std::string TritonCompiler::target_for_request(const FFTRequest &request) const {
    std::string arch = request.device_arch;
    if (arch.rfind("sm_", 0) == 0) {
        arch.erase(0, 3);
    }
    return "cuda:" + arch + ":32";
}

std::filesystem::path TritonCompiler::out_dir() const {
    return std::filesystem::current_path() / ".flagfft";
}

std::string TritonCompiler::python_executable() const {
    std::wstring executable_w = Py_GetProgramFullPath();
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

}  // namespace flagfft
