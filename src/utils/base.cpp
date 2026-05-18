#include "flagfft/core.hpp"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace flagfft {

[[noreturn]] void raise_python(PyObject *type, const std::string &message) {
    if (!Py_IsInitialized()) {
        throw std::runtime_error(message);
    }
    PyErr_SetString(type, message.c_str());
    throw nb::python_error();
}

bool contains(const std::vector<int64_t> &values, int64_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::string py_str(nb::handle object) {
    return nb::cast<std::string>(nb::str(object));
}

std::string strip_torch_prefix(std::string value) {
    constexpr const char *prefix = "torch.";
    if (value.rfind(prefix, 0) == 0) {
        value.erase(0, std::char_traits<char>::length(prefix));
    }
    return value;
}

std::vector<int64_t> int64_vector_from_sequence(nb::handle sequence) {
    std::vector<int64_t> result;
    for (nb::handle item : nb::iter(sequence)) {
        result.push_back(nb::cast<int64_t>(item));
    }
    return result;
}

int64_t product(const std::vector<int64_t> &values) {
    int64_t result = 1;
    for (int64_t value : values) {
        result *= value;
    }
    return result;
}

std::string join_ints(const std::vector<int64_t> &values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    return out.str();
}

int64_t ceil_power_of_two(int64_t value) {
    if (value <= 1) {
        return 1;
    }
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

int64_t lane_block_for(int64_t lanes) {
    return lanes <= 1 ? 1 : ceil_power_of_two(lanes);
}

std::string shell_quote(const std::string &value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out += ch;
        }
    }
    out += "'";
    return out;
}

void hash_combine(std::size_t &seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::filesystem::path executable_directory() {
#if defined(__linux__)
    std::array<char, 4096> buffer{};
    ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length > 0) {
        return std::filesystem::path(std::string(buffer.data(), static_cast<std::size_t>(length)))
            .parent_path();
    }
#endif
    if (Py_IsInitialized()) {
        std::wstring executable_w = Py_GetProgramFullPath();
        if (!executable_w.empty()) {
            return std::filesystem::path(std::string(executable_w.begin(), executable_w.end()))
                .parent_path();
        }
    }
    return std::filesystem::current_path();
}

std::filesystem::path default_cache_dir() {
    return executable_directory() / ".flagfft";
}

std::string plan_node_kind_name(PlanNodeKind kind) {
    switch (kind) {
        case PlanNodeKind::CtLeaf:
            return "ct_leaf";
        case PlanNodeKind::FourStep:
            return "four_step";
        case PlanNodeKind::DirectDft:
            return "direct_dft";
        case PlanNodeKind::StockhamAutosort:
            return "stockham_autosort";
        case PlanNodeKind::Bluestein:
            return "bluestein";
    }
    return "unknown";
}

std::string kernel_kind_name(KernelKind kind) {
    switch (kind) {
        case KernelKind::Leaf:
            return "leaf";
        case KernelKind::FourStepRow:
            return "four_step_row";
        case KernelKind::FourStepCol:
            return "four_step_col";
        case KernelKind::Transpose:
            return "transpose";
        case KernelKind::TwiddleTranspose:
            return "twiddle_transpose";
        case KernelKind::BluesteinPrepare:
            return "bluestein_prepare";
        case KernelKind::BluesteinPointwise:
            return "bluestein_pointwise";
        case KernelKind::BluesteinFinalize:
            return "bluestein_finalize";
    }
    return "unknown";
}

}  // namespace flagfft
