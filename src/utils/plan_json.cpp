#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "sqlite_wrapper.hpp"

#include <nlohmann/json.hpp>

namespace flagfft {

TuneFingerprints tune_fingerprints() {
  TuneFingerprints fps;
  fps.planner = "planner-schema-2-device-smem-leaf";
  fps.codegen = "codegen-schema-3-directional-ifft";
  fps.runtime = "runtime-schema-3-directional-ifft";
  fps.benchmark = "benchmark-schema-2-api-dispatch";
  return fps;
}

nlohmann::json plan_node_to_json(const PlanNodePtr &node) {
  nlohmann::json out;
  out["kind"] = plan_node_kind_name(node->kind);
  out["length"] = node->length;

  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    out["factors"] = leaf->factors;
    out["remainder"] = leaf->remainder;
    out["lanes"] = leaf->lanes;
    out["num_warps"] = leaf->num_warps;
    out["generic_radices"] = leaf->generic_radices;
    out["smem_size"] = leaf->smem_size;
  } else if (auto direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node)) {
    (void)direct;
    out["impl"] = "torch_matmul";
  } else if (auto stockham = std::dynamic_pointer_cast<StockhamPlanNode>(node)) {
    out["factors"] = stockham->factors;
    out["stages"] = nlohmann::json::array();
  } else if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    out["n1"] = four_step->n1;
    out["n2"] = four_step->n2;
    out["row"] = plan_node_to_json(four_step->row_plan);
    out["col"] = plan_node_to_json(four_step->col_plan);
  } else if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
    out["conv_length"] = bluestein->conv_length;
    out["fft_plan"] = plan_node_to_json(bluestein->fft_plan);
  } else if (auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(node)) {
    out["n0"] = two_dim->n0;
    out["n1"] = two_dim->n1;
    out["strategy"] = two_dim_strategy_name(two_dim->strategy);
    out["row"] = plan_node_to_json(two_dim->row_plan);
    out["col"] = plan_node_to_json(two_dim->col_plan);
  }
  return out;
}

nlohmann::json request_to_json(const FFTRequest &request) {
  nlohmann::json out;
  out["op"] = request.direction == "inverse" ? "ifft" : "fft";
  out["length"] = request.fft_length;
  out["input_shape"] = request.input_shape;
  out["n"] = request.n.has_value() ? nlohmann::json(*request.n) : nlohmann::json();
  out["requested_n"] = request.requested_n;
  out["dim"] = request.normalized_dim;
  out["raw_dim"] = request.raw_dim;
  out["norm"] = request.norm;
  out["input_dtype"] = request.input_dtype;
  out["dtype"] = request.input_dtype;
  out["output_dtype"] = request.output_dtype;
  out["device"] = request.device_type;
  out["device_type"] = request.device_type;
  out["device_index"] = request.device_index;
  out["device_arch"] = request.device_arch;
  out["input_strides"] = request.input_strides;
  out["input_layout"] = request.input_layout;
  out["requires_contiguous_copy"] = request.requires_contiguous_copy;
  out["direction"] = request.direction;
  out["batch"] = request.batch;
  out["output_order"] = "natural";
  return out;
}

nlohmann::json plan_key_to_json(const PlanKey &key) {
  nlohmann::json out;
  out["repr"] = key.repr();
  out["schema_version"] = key.schema_version;
  out["kind"] = plan_node_kind_name(key.root_kind);
  out["length"] = key.length;
  out["factors"] = key.factors;
  out["remainder"] = key.remainder;
  out["lanes"] = key.lanes;
  out["num_warps"] = key.num_warps;
  out["generic_radices"] = key.generic_radices;
  out["smem_size"] = key.smem_size;
  out["n1"] = key.n1;
  out["n2"] = key.n2;
  out["conv_length"] = key.conv_length;
  out["child_keys"] = key.child_keys;
  out["hash"] = static_cast<uint64_t>(PlanKeyHash {}(key));
  return out;
}

nlohmann::json problem_key_to_json(const ProblemKey &key) {
  nlohmann::json out;
  out["repr"] = key.repr();
  out["fft_length"] = key.fft_length;
  out["input_shape"] = key.input_shape;
  out["n"] = key.n.has_value() ? nlohmann::json(*key.n) : nlohmann::json();
  out["requested_n"] = key.requested_n;
  out["dim"] = key.normalized_dim;
  out["norm"] = key.norm;
  out["input_dtype"] = key.input_dtype;
  out["output_dtype"] = key.output_dtype;
  out["device_type"] = key.device_type;
  out["device_index"] = key.device_index;
  out["device_arch"] = key.device_arch;
  out["input_strides"] = key.input_strides;
  out["input_layout"] = key.input_layout;
  out["requires_contiguous_copy"] = key.requires_contiguous_copy;
  out["direction"] = key.direction;
  out["hash"] = static_cast<uint64_t>(ProblemKeyHash {}(key));
  return out;
}

nlohmann::json kernel_key_to_json(const KernelKey &key) {
  nlohmann::json out;
  out["repr"] = key.repr();
  out["kind"] = kernel_kind_name(key.kind);
  out["target"] = key.target;
  out["direction"] = key.direction;
  out["length"] = key.length;
  out["factors"] = key.factors;
  out["lanes"] = key.lanes;
  out["num_warps"] = key.num_warps;
  out["generic_radices"] = key.generic_radices;
  out["smem_size"] = key.smem_size;
  out["four_step_n1"] = key.four_step_n1;
  out["four_step_n2"] = key.four_step_n2;
  out["bluestein_n"] = key.bluestein_n;
  out["bluestein_m"] = key.bluestein_m;
  out["hash"] = static_cast<uint64_t>(KernelKeyHash {}(key));
  return out;
}

nlohmann::json wrap_plan_json(const PlanNodePtr &root, const FFTRequest &request, const std::string &source) {
  auto fps = tune_fingerprints();
  nlohmann::json out;
  out["schema_version"] = kPlanSchemaVersion;
  out["source"] = source;
  out["request"] = request_to_json(request);
  out["plan_key"] = plan_key_to_json(PlanKey::from_node(root));
  out["tags"] = nlohmann::json::object();
  out["root"] = plan_node_to_json(root);
  return out;
}

std::optional<nlohmann::json> lookup_tuned_plan_json(const FFTRequest &request) {
  auto db_path = tuned_db_path();
  if (!db_path.has_value()) {
    return std::nullopt;
  }
  std::error_code ec;
  if (!std::filesystem::is_regular_file(*db_path, ec)) {
    return std::nullopt;
  }

  try {
    auto fps = tune_fingerprints();
    SqliteDb db(db_path->string());

    SqliteStmt stmt(db,
                    "SELECT plan_json FROM tuned_measurements "
                    "WHERE schema_version=? AND status='valid' AND rank=0 "
                    "AND device_arch=? AND fft_length=? AND batch_bucket=? AND dtype=? "
                    "AND direction=? AND norm=? AND input_layout=? "
                    "AND planner_fingerprint=? AND codegen_fingerprint=? "
                    "AND runtime_fingerprint=? "
                    "ORDER BY measured_at DESC LIMIT 1");

    stmt.bind_int64(1, kPlanSchemaVersion);
    stmt.bind_text(2, request.device_arch);
    stmt.bind_int64(3, request.requested_n);
    stmt.bind_text(4, batch_bucket(request.batch));
    stmt.bind_text(5, request.input_dtype);
    stmt.bind_text(6, request.direction);
    stmt.bind_text(7, request.norm);
    stmt.bind_text(8, request.input_layout);
    stmt.bind_text(9, fps.planner);
    stmt.bind_text(10, fps.codegen);
    stmt.bind_text(11, fps.runtime);

    if (!stmt.step()) {
      return std::nullopt;
    }

    std::string plan_json_str = stmt.column_text(0);
    return nlohmann::json::parse(plan_json_str);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

}  // namespace flagfft
