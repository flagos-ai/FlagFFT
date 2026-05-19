#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"

namespace flagfft {

PlanNodePtr plan_node_from_json(PlanBuilder &builder, const nlohmann::json &node) {
    std::string kind = node.at("kind").get<std::string>();
    int64_t length = node.at("length").get<int64_t>();
    if (kind == "ct_leaf") {
        std::vector<int64_t> factors = node.at("factors").get<std::vector<int64_t>>();
        int64_t remainder = node.value("remainder", 1);
        int64_t lanes = node.contains("lanes")
                            ? node["lanes"].get<int64_t>()
                            : builder.choose_lanes(length, factors);
        int64_t num_warps = node.contains("num_warps")
                                ? node["num_warps"].get<int64_t>()
                                : builder.choose_num_warps(lanes);
        std::vector<int64_t> generic_radices;
        if (node.contains("generic_radices")) {
            generic_radices = node["generic_radices"].get<std::vector<int64_t>>();
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
        int64_t smem_size = node.contains("smem_size")
                                ? node["smem_size"].get<int64_t>()
                                : (factors.size() <= 1 ? 0 : ceil_power_of_two(length));
        return std::make_shared<LeafPlanNode>(length, std::move(factors), remainder, lanes,
                                               num_warps, std::move(generic_radices), smem_size);
    }
    if (kind == "four_step") {
        int64_t n1 = node.at("n1").get<int64_t>();
        int64_t n2 = node.at("n2").get<int64_t>();
        return std::make_shared<FourStepPlanNode>(
            length, n1, n2,
            plan_node_from_json(builder, node.at("row")),
            plan_node_from_json(builder, node.at("col")));
    }
    if (kind == "bluestein") {
        int64_t conv_length = node.at("conv_length").get<int64_t>();
        return std::make_shared<BluesteinPlanNode>(
            length, conv_length, plan_node_from_json(builder, node.at("fft_plan")));
    }
    if (kind == "direct_dft") {
        return std::make_shared<DirectDFTPlanNode>(length);
    }
    if (kind == "stockham_autosort") {
        return std::make_shared<StockhamPlanNode>(
            length, node.at("factors").get<std::vector<int64_t>>());
    }
    throw std::runtime_error("unsupported plan node kind in forced plan: " + kind);
}

}  // namespace flagfft
