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

BluesteinPlanNode::BluesteinPlanNode(int64_t length, int64_t conv_length, PlanNodePtr fft_plan)
    : PlanNode(length, PlanNodeKind::Bluestein),
      conv_length(conv_length),
      fft_plan(std::move(fft_plan)) {}

nb::dict BluesteinPlanNode::to_dict() const {
    nb::dict out;
    out["kind"] = plan_node_kind_name(kind);
    out["length"] = length;
    out["conv_length"] = conv_length;
    out["fft_plan"] = fft_plan->to_dict();
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
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        key.conv_length = bluestein->conv_length;
        key.child_keys.push_back(PlanKey::from_node(bluestein->fft_plan).repr());
        return key;
    }

    raise_python(PyExc_NotImplementedError,
                 "unsupported plan node kind for key generation: " + plan_node_kind_name(node->kind));
}

}  // namespace flagfft
