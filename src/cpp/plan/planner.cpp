#include "flagfft/core.hpp"

namespace flagfft {

PlanNode::PlanNode(int64_t length, PlanNodeKind kind) : length(length), kind(kind) {}
PlanNode::~PlanNode() = default;

LeafPlanNode::LeafPlanNode(int64_t length,
                           std::vector<int64_t> factors,
                           int64_t remainder,
                           int64_t lanes,
                           int64_t num_warps,
                           std::vector<int64_t> generic_radices,
                           int64_t smem_size)
    : PlanNode(length, PlanNodeKind::CtLeaf),
      factors(std::move(factors)),
      remainder(remainder),
      lanes(lanes),
      num_warps(num_warps),
      generic_radices(std::move(generic_radices)),
      smem_size(smem_size) {}

nb::dict LeafPlanNode::to_dict() const {
    nb::dict out;
    out["kind"] = plan_node_kind_name(kind);
    out["length"] = length;
    out["factors"] = nb::cast(factors);
    out["remainder"] = remainder;
    out["lanes"] = lanes;
    out["num_warps"] = num_warps;
    out["generic_radices"] = nb::cast(generic_radices);
    out["smem_size"] = smem_size;
    return out;
}

DirectDFTPlanNode::DirectDFTPlanNode(int64_t length) : PlanNode(length, PlanNodeKind::DirectDft) {}

nb::dict DirectDFTPlanNode::to_dict() const {
    nb::dict out;
    out["kind"] = plan_node_kind_name(kind);
    out["length"] = length;
    out["impl"] = "torch_matmul";
    return out;
}

StockhamPlanNode::StockhamPlanNode(int64_t length, std::vector<int64_t> factors)
    : PlanNode(length, PlanNodeKind::StockhamAutosort), factors(std::move(factors)) {}

nb::dict StockhamPlanNode::to_dict() const {
    nb::dict out;
    out["kind"] = plan_node_kind_name(kind);
    out["length"] = length;
    out["factors"] = nb::cast(factors);
    out["stages"] = nb::list();
    return out;
}

FourStepPlanNode::FourStepPlanNode(int64_t length, int64_t n1, int64_t n2, PlanNodePtr row, PlanNodePtr col)
    : PlanNode(length, PlanNodeKind::FourStep),
      n1(n1),
      n2(n2),
      row_plan(std::move(row)),
      col_plan(std::move(col)) {}

nb::dict FourStepPlanNode::to_dict() const {
    nb::dict out;
    out["kind"] = plan_node_kind_name(kind);
    out["length"] = length;
    out["n1"] = n1;
    out["n2"] = n2;
    out["row"] = row_plan->to_dict();
    out["col"] = col_plan->to_dict();
    return out;
}

PlanKey PlanKey::from_node(const PlanNodePtr &node) {
    if (node == nullptr) {
        raise_python(PyExc_ValueError, "cannot build a plan key from a null plan node");
    }

    PlanKey key;
    key.schema_version = kPlanSchemaVersion;
    key.root_kind = node->kind;
    key.length = node->length;

    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        key.factors = leaf->factors;
        key.remainder = leaf->remainder;
        key.lanes = leaf->lanes;
        key.num_warps = leaf->num_warps;
        key.generic_radices = leaf->generic_radices;
        key.smem_size = leaf->smem_size;
        return key;
    }
    if (auto direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node)) {
        (void)direct;
        return key;
    }
    if (auto stockham = std::dynamic_pointer_cast<StockhamPlanNode>(node)) {
        key.factors = stockham->factors;
        return key;
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        key.n1 = four_step->n1;
        key.n2 = four_step->n2;
        key.child_keys.push_back(PlanKey::from_node(four_step->row_plan).repr());
        key.child_keys.push_back(PlanKey::from_node(four_step->col_plan).repr());
        return key;
    }

    raise_python(PyExc_NotImplementedError,
                 "unsupported plan node kind for key generation: " + plan_node_kind_name(node->kind));
}

PlanNodePtr PlanBuilder::build(int64_t n) {
    if (n <= 0) {
        raise_python(PyExc_ValueError, "FFT length must be positive");
    }
    return build_auto_node(n);
}

double PlanBuilder::cost_for(int64_t n) {
    auto it = cost_cache_.find(n);
    if (it != cost_cache_.end()) {
        return it->second;
    }
    std::vector<PlanCandidate> candidates = build_auto_candidates(n);
    if (candidates.empty()) {
        raise_python(PyExc_ValueError,
                     "length " + std::to_string(n) +
                         " has no supported FFT implementation route");
    }
    auto best = select_candidate(candidates);
    cost_cache_[n] = best.cost;
    return best.cost;
}

nb::dict PlanBuilder::wrap_plan_dict(const PlanNodePtr &root, const FFTRequest &request) {
    return wrap_forced_plan_dict(root, request, "cpp_auto");
}

nb::dict PlanBuilder::wrap_forced_plan_dict(const PlanNodePtr &root,
                                            const FFTRequest &request,
                                            std::string source) {
    nb::dict out;
    out["schema_version"] = kPlanSchemaVersion;
    out["source"] = std::move(source);
    out["request"] = request_to_dict(request);
    out["estimated_cost"] = cost_for(root->length);
    out["plan_key"] = plan_key_to_dict(PlanKey::from_node(root));
    out["tags"] = nb::dict();
    out["root"] = root->to_dict();
    return out;
}

PlanNodePtr PlanBuilder::node_from_dict(nb::dict node) {
    std::string kind = nb::cast<std::string>(node["kind"]);
    int64_t length = nb::cast<int64_t>(node["length"]);
    if (kind == "ct_leaf") {
        std::vector<int64_t> factors = int64_vector_from_sequence(node["factors"]);
        int64_t remainder = node.contains("remainder") ? nb::cast<int64_t>(node["remainder"]) : 1;
        int64_t lanes = node.contains("lanes") ? nb::cast<int64_t>(node["lanes"]) : choose_lanes(length, factors);
        int64_t num_warps =
            node.contains("num_warps") ? nb::cast<int64_t>(node["num_warps"]) : choose_num_warps(lanes);
        std::vector<int64_t> generic_radices;
        if (node.contains("generic_radices")) {
            generic_radices = int64_vector_from_sequence(node["generic_radices"]);
        } else {
            for (int64_t radix : factors) {
                if (!contains(kSpecializedButterflyRadices, radix) &&
                    !contains(kSpecializedDirectCodeletRadices, radix) &&
                    !contains(generic_radices, radix)) {
                    generic_radices.push_back(radix);
                }
            }
            std::sort(generic_radices.begin(), generic_radices.end());
        }
        int64_t smem_size =
            node.contains("smem_size") ? nb::cast<int64_t>(node["smem_size"]) : ceil_power_of_two(length);
        return std::make_shared<LeafPlanNode>(length, std::move(factors), remainder, lanes,
                                              num_warps, std::move(generic_radices), smem_size);
    }
    if (kind == "four_step") {
        int64_t n1 = nb::cast<int64_t>(node["n1"]);
        int64_t n2 = nb::cast<int64_t>(node["n2"]);
        return std::make_shared<FourStepPlanNode>(
            length, n1, n2, node_from_dict(nb::cast<nb::dict>(node["row"])),
            node_from_dict(nb::cast<nb::dict>(node["col"])));
    }
    if (kind == "direct_dft") {
        return std::make_shared<DirectDFTPlanNode>(length);
    }
    if (kind == "stockham_autosort") {
        return std::make_shared<StockhamPlanNode>(length, int64_vector_from_sequence(node["factors"]));
    }
    raise_python(PyExc_ValueError, "unsupported plan node kind in forced plan: " + kind);
}

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
    std::vector<int64_t> score = {lanes, -lane_block, -generic_stage_count,
                                  -static_cast<int64_t>(factors.size())};
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

int64_t PlanBuilder::estimate_leaf_smem_bytes(int64_t n, const std::vector<int64_t> &factors) {
    if (factors.size() <= 1) {
        return 0;
    }
    int64_t smem_n = ceil_power_of_two(n);
    return 4 * smem_n * 4;
}

bool PlanBuilder::should_use_leaf(int64_t n, const std::vector<int64_t> &factors) {
    return n <= kLeafMaxN && estimate_leaf_smem_bytes(n, factors) <= kLeafSmemBudgetBytes;
}

PlanNodePtr PlanBuilder::make_leaf_plan(int64_t n, const std::vector<int64_t> &factors, int64_t rem) {
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
                                          generic_radices, ceil_power_of_two(n));
}

double PlanBuilder::estimate_leaf_warm_cost(int64_t n) {
    auto factors = select_leaf_factors(n);
    return estimate_leaf_warm_cost(n, factors);
}

double PlanBuilder::estimate_leaf_warm_cost(int64_t n, const std::vector<int64_t> &factors) {
    int64_t lanes = choose_lanes(n, factors);
    int64_t warps = choose_num_warps(lanes);
    int64_t generic_stage_count = 0;
    for (int64_t radix : factors) {
        if (!contains(kSpecializedButterflyRadices, radix) &&
            !contains(kSpecializedDirectCodeletRadices, radix)) {
            ++generic_stage_count;
        }
    }
    double base = static_cast<double>(n * static_cast<int64_t>(factors.size()) * warps) /
                  static_cast<double>(lanes);
    return base + static_cast<double>(generic_stage_count * n);
}

double PlanBuilder::estimate_direct_dft_cost(int64_t n) {
    return static_cast<double>(n * n);
}

double PlanBuilder::four_step_cost(int64_t n1, int64_t n2) {
    return static_cast<double>(n2) * cost_for(n1) +
           static_cast<double>(n1) * cost_for(n2) + static_cast<double>(n1 * n2);
}

int PlanBuilder::priority(const PlanNodePtr &node) {
    if (node->kind == PlanNodeKind::CtLeaf) {
        return 0;
    }
    if (node->kind == PlanNodeKind::FourStep) {
        return 1;
    }
    if (node->kind == PlanNodeKind::StockhamAutosort) {
        return 2;
    }
    if (node->kind == PlanNodeKind::DirectDft) {
        return 3;
    }
    return 99;
}

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
    return candidates;
}

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

nb::list PlanBuilder::enumerate_candidate_plans(int64_t n, const FFTRequest &request) {
    nb::list out;
    for (const auto &candidate : build_tune_candidates(n, 0)) {
        nb::dict item = wrap_forced_plan_dict(candidate.node, request, "cpp_tune_candidate");
        item["estimated_cost"] = candidate.cost;
        item["priority"] = candidate.priority;
        out.append(std::move(item));
    }
    return out;
}

PlanCandidate PlanBuilder::select_candidate(const std::vector<PlanCandidate> &candidates) {
    if (candidates.empty()) {
        raise_python(PyExc_ValueError, "no FFT plan candidates were generated");
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
