#include "flagfft/core.hpp"

namespace flagfft {

Factorization PlanBuilder::factorize_supported_radices(int64_t n) {
    if (n <= 0) {
        raise_python(PyExc_ValueError, "FFT length must be positive");
    }
    Factorization result;
    int64_t rem = n;
    while (rem > 1) {
        bool found = false;
        for (int64_t radix : kSupportedRadices) {
            if (rem % radix == 0) {
                result.factors.push_back(radix);
                rem /= radix;
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }
    result.remainder = rem;
    return result;
}

std::vector<std::vector<int64_t>> PlanBuilder::enumerate_supported_factorizations(int64_t n) {
    auto it = factorization_cache_.find(n);
    if (it != factorization_cache_.end()) {
        return it->second;
    }

    std::vector<std::vector<int64_t>> result;
    for (int64_t radix : kSupportedRadices) {
        if (n % radix != 0) {
            continue;
        }
        if (n == radix) {
            result.push_back({radix});
            continue;
        }
        for (auto tail : enumerate_supported_factorizations(n / radix)) {
            std::vector<int64_t> factors = {radix};
            factors.insert(factors.end(), tail.begin(), tail.end());
            result.push_back(std::move(factors));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    factorization_cache_[n] = result;
    return result;
}

std::vector<int64_t> PlanBuilder::factorize_or_raise(int64_t n) {
    Factorization factorization = factorize_supported_radices(n);
    if (factorization.remainder != 1) {
        std::ostringstream message;
        message << "length " << n << " is not fully factorable by supported radices, "
                << "remainder=" << factorization.remainder;
        raise_python(PyExc_ValueError, message.str());
    }
    if (factorization.factors.empty()) {
        raise_python(PyExc_ValueError, "at least one radix stage is required");
    }
    return factorization.factors;
}

int64_t PlanBuilder::choose_lanes(int64_t n, const std::vector<int64_t> &factors) {
    int64_t gcd_all = n / factors.front();
    for (std::size_t i = 1; i < factors.size(); ++i) {
        gcd_all = std::gcd(gcd_all, n / factors[i]);
    }
    int64_t upper = std::min(gcd_all, kMaxLanes);
    for (int64_t candidate = upper; candidate > 0; --candidate) {
        if (gcd_all % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

std::vector<int64_t> PlanBuilder::score_leaf_factorization(int64_t n, const std::vector<int64_t> &factors) {
    int64_t lanes = choose_lanes(n, factors);
    int64_t lane_block = lane_block_for(lanes);
    int64_t generic_stage_count = 0;
    for (int64_t radix : factors) {
        if (!contains(kSpecializedButterflyRadices, radix) &&
            !contains(kSpecializedDirectCodeletRadices, radix)) {
            ++generic_stage_count;
        }
    }
    int64_t warm_cost = static_cast<int64_t>(std::llround(estimate_leaf_warm_cost(n, factors) * 1024.0));
    std::vector<int64_t> score = {-warm_cost,
                                  -static_cast<int64_t>(factors.size()),
                                  lanes,
                                  -lane_block,
                                  -generic_stage_count};
    score.insert(score.end(), factors.begin(), factors.end());
    return score;
}

std::vector<int64_t> PlanBuilder::select_leaf_factors(int64_t n) {
    auto it = best_leaf_factors_cache_.find(n);
    if (it != best_leaf_factors_cache_.end()) {
        return it->second;
    }

    std::vector<std::vector<int64_t>> candidates = enumerate_supported_factorizations(n);
    std::vector<int64_t> best;
    if (candidates.empty()) {
        best = factorize_or_raise(n);
    } else {
        best = candidates.front();
        auto best_score = score_leaf_factorization(n, best);
        for (const auto &candidate : candidates) {
            auto score = score_leaf_factorization(n, candidate);
            if (score > best_score) {
                best = candidate;
                best_score = std::move(score);
            }
        }
    }
    best_leaf_factors_cache_[n] = best;
    return best;
}

int64_t PlanBuilder::choose_num_warps(int64_t lanes) {
    int64_t warps = std::max<int64_t>(1, (lane_block_for(lanes) + 31) / 32);
    int64_t choice = 1;
    while (choice < warps) {
        choice *= 2;
    }
    return std::min<int64_t>(choice, 8);
}

std::optional<int64_t> PlanBuilder::leaf_smem_elements(int64_t n,
                                                       const std::vector<int64_t> &factors,
                                                       const std::string &input_dtype) {
    if (input_dtype != "complex64" && input_dtype != "float32") {
        return std::nullopt;
    }
    if (factors.size() <= 1) {
        return 0;
    }
    return ceil_power_of_two(n);
}

std::optional<int64_t> PlanBuilder::leaf_smem_bytes(int64_t n,
                                                    const std::vector<int64_t> &factors,
                                                    const std::string &input_dtype) {
    auto elements = leaf_smem_elements(n, factors, input_dtype);
    if (!elements.has_value()) {
        return std::nullopt;
    }
    return 4 * *elements * static_cast<int64_t>(sizeof(float));
}

bool PlanBuilder::should_use_leaf(int64_t n, const std::vector<int64_t> &factors) {
    const RequestContext &context = request_context();
    auto smem_bytes = leaf_smem_bytes(n, factors, context.input_dtype);
    return n <= kLeafMaxN && smem_bytes.has_value() &&
           *smem_bytes <= context.max_dynamic_smem_bytes;
}

PlanNodePtr PlanBuilder::make_leaf_plan(int64_t n, const std::vector<int64_t> &factors, int64_t rem) {
    const RequestContext &context = request_context();
    auto smem_elements = leaf_smem_elements(n, factors, context.input_dtype);
    if (!smem_elements.has_value()) {
        raise_python(PyExc_NotImplementedError,
                     "ct_leaf shared-memory planner currently supports only float32 and complex64 inputs");
    }
    int64_t lanes = choose_lanes(n, factors);
    std::vector<int64_t> generic_radices;
    for (int64_t radix : factors) {
        if (!contains(kSpecializedButterflyRadices, radix) &&
            !contains(kSpecializedDirectCodeletRadices, radix) &&
            std::find(generic_radices.begin(), generic_radices.end(), radix) ==
                generic_radices.end()) {
            generic_radices.push_back(radix);
        }
    }
    std::sort(generic_radices.begin(), generic_radices.end());
    return std::make_shared<LeafPlanNode>(n, factors, rem, lanes, choose_num_warps(lanes),
                                          generic_radices, *smem_elements);
}

}  // namespace flagfft
