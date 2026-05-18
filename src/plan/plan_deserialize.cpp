#include "flagfft/core.hpp"

namespace flagfft {

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
        int64_t smem_size = node.contains("smem_size")
                                 ? nb::cast<int64_t>(node["smem_size"])
                                 : (factors.size() <= 1 ? 0 : ceil_power_of_two(length));
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
    if (kind == "bluestein") {
        int64_t conv_length = nb::cast<int64_t>(node["conv_length"]);
        return std::make_shared<BluesteinPlanNode>(
            length, conv_length, node_from_dict(nb::cast<nb::dict>(node["fft_plan"])));
    }
    if (kind == "direct_dft") {
        return std::make_shared<DirectDFTPlanNode>(length);
    }
    if (kind == "stockham_autosort") {
        return std::make_shared<StockhamPlanNode>(length, int64_vector_from_sequence(node["factors"]));
    }
    raise_python(PyExc_ValueError, "unsupported plan node kind in forced plan: " + kind);
}

}  // namespace flagfft
