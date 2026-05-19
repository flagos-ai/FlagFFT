#include "flagfft/core.hpp"

namespace flagfft {

std::vector<PlanCandidate> PlanBuilder::top_candidates(std::vector<PlanCandidate> candidates, int64_t limit) {
    std::sort(candidates.begin(), candidates.end(), [](const PlanCandidate &a, const PlanCandidate &b) {
        if (a.cost != b.cost) {
            return a.cost < b.cost;
        }
        if (a.priority != b.priority) {
            return a.priority < b.priority;
        }
        return PlanKey::from_node(a.node).repr() < PlanKey::from_node(b.node).repr();
    });
    std::vector<PlanCandidate> unique;
    std::vector<std::string> seen;
    for (auto &candidate : candidates) {
        std::string repr = PlanKey::from_node(candidate.node).repr();
        if (std::find(seen.begin(), seen.end(), repr) != seen.end()) {
            continue;
        }
        seen.push_back(std::move(repr));
        unique.push_back(std::move(candidate));
        if (static_cast<int64_t>(unique.size()) >= limit) {
            break;
        }
    }
    return unique;
}

std::vector<PlanCandidate> PlanBuilder::build_leaf_tune_candidates(int64_t n) {
    std::vector<PlanCandidate> candidates;
    Factorization factorization = factorize_supported_radices(n);
    if (factorization.remainder != 1 || factorization.factors.empty()) {
        return candidates;
    }
    std::vector<std::vector<int64_t>> factorizations = enumerate_supported_factorizations(n);
    std::vector<std::string> seen_multisets;
    for (auto factors : factorizations) {
        if (!should_use_leaf(n, factors)) {
            continue;
        }
        std::vector<int64_t> multiset = factors;
        std::sort(multiset.begin(), multiset.end());
        std::string multiset_key = join_ints(multiset);
        if (std::find(seen_multisets.begin(), seen_multisets.end(), multiset_key) !=
            seen_multisets.end()) {
            continue;
        }
        seen_multisets.push_back(std::move(multiset_key));

        std::vector<std::vector<int64_t>> orders;
        std::vector<int64_t> asc = multiset;
        std::vector<int64_t> desc = asc;
        std::reverse(desc.begin(), desc.end());
        orders.push_back(desc);
        orders.push_back(asc);
        orders.push_back(factors);

        int64_t emitted_orders = 0;
        for (auto &order : orders) {
            if (emitted_orders >= kTuneOrdersPerFactorMultiset) {
                break;
            }
            PlanNodePtr node = make_leaf_plan(n, order);
            double cost = estimate_leaf_warm_cost(n, order);
            int64_t lanes = choose_lanes(n, order);
            double lane_bonus = lanes > 0 ? static_cast<double>(n) / static_cast<double>(lanes) : 0.0;
            candidates.push_back({node, cost - lane_bonus, priority(node)});
            ++emitted_orders;
        }
    }
    int64_t limit = n >= kLeafTuneLargeN ? kLeafTuneLargeTopK : kLeafTuneTopK;
    return top_candidates(std::move(candidates), limit);
}

std::vector<PlanCandidate> PlanBuilder::build_tune_candidates(int64_t n, int64_t depth) {
    auto cached = tune_candidate_cache_.find(n);
    if (cached != tune_candidate_cache_.end()) {
        return cached->second;
    }
    if (depth > 3) {
        return build_leaf_tune_candidates(n);
    }

    std::vector<PlanCandidate> candidates = build_leaf_tune_candidates(n);
    if (n <= kDirectDftMaxN) {
        PlanNodePtr node = std::make_shared<DirectDFTPlanNode>(n);
        candidates.push_back({node, estimate_direct_dft_cost(n), priority(node)});
    }

    struct DivisorPair {
        int64_t n1 = 0;
        int64_t n2 = 0;
        double score = 0.0;
    };
    std::vector<DivisorPair> divisor_pairs;
    for (int64_t n1 : enumerate_divisors(n)) {
        if (n1 <= 1 || n1 >= n) {
            continue;
        }
        int64_t n2 = n / n1;
        double balance =
            std::abs(std::log(static_cast<double>(n1)) - std::log(static_cast<double>(n2)));
        divisor_pairs.push_back({n1, n2, balance + static_cast<double>(n1 + n2) / static_cast<double>(n)});
    }
    std::sort(divisor_pairs.begin(), divisor_pairs.end(), [](const DivisorPair &a, const DivisorPair &b) {
        if (a.score != b.score) {
            return a.score < b.score;
        }
        return a.n1 < b.n1;
    });
    if (static_cast<int64_t>(divisor_pairs.size()) > kTuneFourStepPairTopK) {
        divisor_pairs.resize(static_cast<std::size_t>(kTuneFourStepPairTopK));
    }

    std::vector<PlanCandidate> pair_candidates;
    for (const auto &pair : divisor_pairs) {
        int64_t n1 = pair.n1;
        int64_t n2 = pair.n2;
        auto rows = build_tune_candidates(n1, depth + 1);
        auto cols = build_tune_candidates(n2, depth + 1);
        if (rows.empty() || cols.empty()) {
            continue;
        }
        int64_t combos = 0;
        for (const auto &row : rows) {
            for (const auto &col : cols) {
                if (combos >= kTuneFourStepCombosPerPair) {
                    break;
                }
                PlanNodePtr node = std::make_shared<FourStepPlanNode>(n, n1, n2, row.node, col.node);
                double balance = pair.score;
                double cost = static_cast<double>(n2) * row.cost + static_cast<double>(n1) * col.cost +
                              static_cast<double>(n1 * n2) + balance;
                pair_candidates.push_back({node, cost, priority(node)});
                ++combos;
            }
            if (combos >= kTuneFourStepCombosPerPair) {
                break;
            }
        }
    }
    auto four_step = top_candidates(std::move(pair_candidates), kTuneFourStepTopK);
    candidates.insert(candidates.end(), four_step.begin(), four_step.end());
    auto result = top_candidates(std::move(candidates), kTuneStaticPlanTopK);
    tune_candidate_cache_[n] = result;
    return result;
}

PlanCandidate PlanBuilder::select_candidate(const std::vector<PlanCandidate> &candidates) {
    if (candidates.empty()) {
        throw std::runtime_error("no FFT plan candidates were generated");
    }
    return *std::min_element(candidates.begin(), candidates.end(),
                             [](const PlanCandidate &a, const PlanCandidate &b) {
                                 if (a.cost != b.cost) {
                                     return a.cost < b.cost;
                                 }
                                 return a.priority < b.priority;
                             });
}

PlanNodePtr PlanBuilder::build_auto_node(int64_t n) {
    auto it = node_cache_.find(n);
    if (it != node_cache_.end()) {
        return it->second;
    }
    PlanCandidate candidate = select_candidate(build_auto_candidates(n));
    node_cache_[n] = candidate.node;
    cost_cache_[n] = candidate.cost;
    return candidate.node;
}

}  // namespace flagfft
