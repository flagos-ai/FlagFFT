#include "flagfft/core.hpp"

namespace flagfft {

nb::dict request_to_dict(const FFTRequest &request) {
    nb::dict out;
    out["op"] = request.direction == "inverse" ? "ifft" : "fft";
    out["length"] = request.fft_length;
    out["input_shape"] = nb::cast(request.input_shape);
    out["n"] = request.n.has_value() ? nb::cast(*request.n) : nb::none();
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
    out["input_strides"] = nb::cast(request.input_strides);
    out["input_layout"] = request.input_layout;
    out["requires_contiguous_copy"] = request.requires_contiguous_copy;
    out["direction"] = request.direction;
    out["batch"] = request.batch;
    out["output_order"] = "natural";
    return out;
}

nb::dict problem_key_to_dict(const ProblemKey &key) {
    nb::dict out;
    out["repr"] = key.repr();
    out["fft_length"] = key.fft_length;
    out["input_shape"] = nb::cast(key.input_shape);
    out["n"] = key.n.has_value() ? nb::cast(*key.n) : nb::none();
    out["requested_n"] = key.requested_n;
    out["dim"] = key.normalized_dim;
    out["norm"] = key.norm;
    out["input_dtype"] = key.input_dtype;
    out["output_dtype"] = key.output_dtype;
    out["device_type"] = key.device_type;
    out["device_index"] = key.device_index;
    out["device_arch"] = key.device_arch;
    out["input_strides"] = nb::cast(key.input_strides);
    out["input_layout"] = key.input_layout;
    out["requires_contiguous_copy"] = key.requires_contiguous_copy;
    out["direction"] = key.direction;
    out["hash"] = static_cast<uint64_t>(ProblemKeyHash{}(key));
    return out;
}

nb::dict plan_key_to_dict(const PlanKey &key) {
    nb::dict out;
    out["repr"] = key.repr();
    out["schema_version"] = key.schema_version;
    out["kind"] = plan_node_kind_name(key.root_kind);
    out["length"] = key.length;
    out["factors"] = nb::cast(key.factors);
    out["remainder"] = key.remainder;
    out["lanes"] = key.lanes;
    out["num_warps"] = key.num_warps;
    out["generic_radices"] = nb::cast(key.generic_radices);
    out["smem_size"] = key.smem_size;
    out["n1"] = key.n1;
    out["n2"] = key.n2;
    out["conv_length"] = key.conv_length;
    out["child_keys"] = nb::cast(key.child_keys);
    out["hash"] = static_cast<uint64_t>(PlanKeyHash{}(key));
    return out;
}

nb::dict kernel_key_to_dict(const KernelKey &key) {
    nb::dict out;
    out["repr"] = key.repr();
    out["kind"] = kernel_kind_name(key.kind);
    out["target"] = key.target;
    out["direction"] = key.direction;
    out["length"] = key.length;
    out["factors"] = nb::cast(key.factors);
    out["lanes"] = key.lanes;
    out["num_warps"] = key.num_warps;
    out["generic_radices"] = nb::cast(key.generic_radices);
    out["smem_size"] = key.smem_size;
    out["four_step_n1"] = key.four_step_n1;
    out["four_step_n2"] = key.four_step_n2;
    out["bluestein_n"] = key.bluestein_n;
    out["bluestein_m"] = key.bluestein_m;
    out["hash"] = static_cast<uint64_t>(KernelKeyHash{}(key));
    return out;
}

}  // namespace flagfft
