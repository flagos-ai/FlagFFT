#include "flagfft/core.hpp"

namespace flagfft {

nb::object ExecutablePlan::execute(nb::object input) const {
    if (backend == ExecutionBackend::TorchFFT) {
        nb::module_ torch = nb::module_::import_("torch");
        const char *op_name = request.direction == "inverse" ? "ifft" : "fft";
        return torch.attr("fft").attr(op_name)(input, "n"_a = request.requested_n,
                                               "dim"_a = request.normalized_dim,
                                               "norm"_a = request.norm);
    }

    nb::object exec_input = input;
    if (request.input_dtype == "float32") {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object zeros = torch.attr("zeros_like")(input);
        exec_input = torch.attr("complex")(input, zeros);
        if (request.requires_contiguous_copy) {
            exec_input = exec_input.attr("contiguous")();
        }
    } else if (request.requires_contiguous_copy) {
        exec_input = exec_input.attr("contiguous")();
    }
    ExecutionContext context{request, current_cuda_stream(request)};
    nb::object result = compiled_root->execute(exec_input, context);
    if (!request_has_flat_batch_shape(request)) {
        result = result.attr("reshape")(nb::cast(request.input_shape));
    }

    double scale = normalization_scale(request);
    if (scale != 1.0) {
        return result.attr("mul")(scale);
    }
    return result;
}

}  // namespace flagfft
