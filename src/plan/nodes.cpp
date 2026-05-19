#include "flagfft/core.hpp"

#include <sstream>

namespace flagfft {

static std::string indent_str(int indent) {
    return std::string(static_cast<std::size_t>(indent), ' ');
}

static std::string factors_str(const std::vector<int64_t> &factors) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (i > 0) oss << ",";
        oss << factors[i];
    }
    oss << "]";
    return oss.str();
}

PlanNode::PlanNode(int64_t length, PlanNodeKind kind) : length(length), kind(kind) {}
PlanNode::~PlanNode() = default;

std::string LeafPlanNode::describe(int indent) const {
    std::ostringstream oss;
    oss << indent_str(indent)
        << "LeafPlan(n=" << length
        << ", factors=" << factors_str(factors)
        << ", lanes=" << lanes
        << ", num_warps=" << num_warps;
    if (!generic_radices.empty()) {
        oss << ", generic_radices=" << factors_str(generic_radices);
    }
    oss << ", smem_size=" << smem_size << ")";
    return oss.str();
}

std::string DirectDFTPlanNode::describe(int indent) const {
    std::ostringstream oss;
    oss << indent_str(indent) << "DirectDFT(n=" << length << ")";
    return oss.str();
}

std::string StockhamPlanNode::describe(int indent) const {
    std::ostringstream oss;
    oss << indent_str(indent) << "Stockham(n=" << length
        << ", factors=" << factors_str(factors) << ")";
    return oss.str();
}

std::string FourStepPlanNode::describe(int indent) const {
    std::ostringstream oss;
    oss << indent_str(indent) << "FourStep(n=" << length
        << ", n1=" << n1 << ", n2=" << n2 << ")\n";
    oss << row_plan->describe(indent + 2) << "\n";
    oss << col_plan->describe(indent + 2);
    return oss.str();
}

std::string BluesteinPlanNode::describe(int indent) const {
    std::ostringstream oss;
    oss << indent_str(indent) << "Bluestein(n=" << length
        << ", conv_length=" << conv_length << ")\n";
    oss << fft_plan->describe(indent + 2);
    return oss.str();
}

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

DirectDFTPlanNode::DirectDFTPlanNode(int64_t length) : PlanNode(length, PlanNodeKind::DirectDft) {}

StockhamPlanNode::StockhamPlanNode(int64_t length, std::vector<int64_t> factors)
    : PlanNode(length, PlanNodeKind::StockhamAutosort), factors(std::move(factors)) {}

FourStepPlanNode::FourStepPlanNode(int64_t length, int64_t n1, int64_t n2, PlanNodePtr row, PlanNodePtr col)
    : PlanNode(length, PlanNodeKind::FourStep),
      n1(n1),
      n2(n2),
      row_plan(std::move(row)),
      col_plan(std::move(col)) {}

BluesteinPlanNode::BluesteinPlanNode(int64_t length, int64_t conv_length, PlanNodePtr fft_plan)
    : PlanNode(length, PlanNodeKind::Bluestein),
      conv_length(conv_length),
      fft_plan(std::move(fft_plan)) {}

PlanKey PlanKey::from_node(const PlanNodePtr &node) {
    if (node == nullptr) {
        throw std::runtime_error("cannot build a plan key from a null plan node");
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
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        key.conv_length = bluestein->conv_length;
        key.child_keys.push_back(PlanKey::from_node(bluestein->fft_plan).repr());
        return key;
    }

    throw std::runtime_error(
        "unsupported plan node kind for key generation: " + plan_node_kind_name(node->kind));
}

}  // namespace flagfft
