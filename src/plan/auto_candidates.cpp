#include "flagfft/core.hpp"

namespace flagfft {

std::vector<int64_t> PlanBuilder::enumerate_divisors(int64_t n) {
    auto it = divisor_cache_.find(n);
    if (it != divisor_cache_.end()) {
        return it->second;
    }
    std::vector<int64_t> divisors;
    int64_t root = static_cast<int64_t>(std::sqrt(static_cast<double>(n)));
    for (int64_t divisor = 1; divisor <= root; ++divisor) {
        if (n % divisor != 0) {
            continue;
        }
        divisors.push_back(divisor);
        int64_t mate = n / divisor;
        if (mate != divisor) {
            divisors.push_back(mate);
        }
    }
    std::sort(divisors.begin(), divisors.end());
    divisor_cache_[n] = divisors;
    return divisors;
}

int64_t PlanBuilder::next_supported_convolution_length(int64_t minimum) {
    if (minimum <= 1) {
        return 1;
    }
    int64_t power = ceil_power_of_two(minimum);
    for (int64_t candidate = minimum; candidate <= power; ++candidate) {
        Factorization factorization = factorize_supported_radices(candidate);
        if (factorization.remainder == 1 && !factorization.factors.empty()) {
            return candidate;
        }
    }
    return power;
}

PlanNodePtr PlanBuilder::make_bluestein_plan(int64_t n) {
    int64_t conv_length = next_supported_convolution_length(2 * n - 1);
    PlanNodePtr fft_plan = build_auto_node(conv_length);
    return std::make_shared<BluesteinPlanNode>(n, conv_length, std::move(fft_plan));
}

std::vector<PlanCandidate> PlanBuilder::build_auto_candidates(int64_t n) {
    if (n <= 0) {
        raise_python(PyExc_ValueError, "FFT length must be positive");
    }

    std::vector<PlanCandidate> candidates;
    Factorization factorization = factorize_supported_radices(n);
    if (factorization.remainder == 1 && !factorization.factors.empty() &&
        should_use_leaf(n, factorization.factors)) {
        PlanNodePtr node = make_leaf_plan(n, select_leaf_factors(n));
        candidates.push_back({node, estimate_leaf_warm_cost(n), priority(node)});
    }

    if (n <= kDirectDftMaxN) {
        PlanNodePtr node = std::make_shared<DirectDFTPlanNode>(n);
        candidates.push_back({node, estimate_direct_dft_cost(n), priority(node)});
    }

    if (n == 16384 && should_use_leaf(64, std::vector<int64_t>{4, 4, 4}) &&
        should_use_leaf(256, std::vector<int64_t>{4, 8, 8})) {
        PlanNodePtr row = make_leaf_plan(64, std::vector<int64_t>{4, 4, 4});
        PlanNodePtr col = make_leaf_plan(256, std::vector<int64_t>{4, 8, 8});
        PlanNodePtr node = std::make_shared<FourStepPlanNode>(n, 64, 256, row, col);
        candidates.push_back({node, four_step_cost(64, 256) * 0.5, priority(node)});
    }

    for (int64_t n1 : enumerate_divisors(n)) {
        if (n1 <= 1 || n1 >= n) {
            continue;
        }
        int64_t n2 = n / n1;
        try {
            PlanNodePtr row = build_auto_node(n1);
            PlanNodePtr col = build_auto_node(n2);
            PlanNodePtr node = std::make_shared<FourStepPlanNode>(n, n1, n2, row, col);
            double balance =
                std::abs(std::log(static_cast<double>(n1)) - std::log(static_cast<double>(n2)));
            candidates.push_back({node, four_step_cost(n1, n2) + balance, priority(node)});
        } catch (const nb::python_error &) {
            PyErr_Clear();
        }
    }
    if (candidates.empty() && n > kDirectDftMaxN) {
        PlanNodePtr node = make_bluestein_plan(n);
        auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node);
        candidates.push_back({node, bluestein_cost(n, bluestein->conv_length), priority(node)});
    }
    return candidates;
}

}  // namespace flagfft
