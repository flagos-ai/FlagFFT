#include "flagfft/core.hpp"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace flagfft {

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

bool contains(const std::vector<int64_t> &values, int64_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
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
        case KernelKind::ReshapePack:
            return "reshape_pack";
        case KernelKind::TwiddleReshapePack:
            return "twiddle_reshape_pack";
        case KernelKind::RealToComplex:
            return "real_to_complex";
        case KernelKind::R2CHalfPack:
            return "r2c_half_pack";
        case KernelKind::CompactToHermitianFull:
            return "compact_to_hermitian_full";
        case KernelKind::ComplexToReal:
            return "complex_to_real";
    }
    return "unknown";
}

int64_t complex_element_bytes(const std::string &input_dtype) {
    if (input_dtype == "complex128" || input_dtype == "float64") {
        return 16;
    }
    return 8;
}

std::string complex_dtype_for(const std::string &input_dtype) {
    if (input_dtype == "complex128" || input_dtype == "float64") {
        return "complex128";
    }
    return "complex64";
}

}  // namespace flagfft
