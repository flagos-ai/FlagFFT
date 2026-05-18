#include "flagfft/core.hpp"

namespace flagfft {

std::string output_dtype_for(const std::string &input_dtype) {
    if (input_dtype == "complex64" || input_dtype == "float32") {
        return "complex64";
    }
    if (input_dtype == "complex128" || input_dtype == "float64") {
        return "complex128";
    }
    return "unsupported";
}

FFTRequest build_request(nb::object input,
                         nb::object n_obj,
                         int64_t dim,
                         nb::object norm_obj,
                         std::string direction) {
    FFTRequest request;
    request.direction = std::move(direction);
    request.input_shape = int64_vector_from_sequence(input.attr("size")());
    request.input_strides = int64_vector_from_sequence(input.attr("stride")());
    request.raw_dim = dim;

    const int64_t ndim = static_cast<int64_t>(request.input_shape.size());
    if (ndim > 0 && -ndim <= dim && dim < ndim) {
        request.normalized_dim = dim < 0 ? dim + ndim : dim;
        request.fft_length = request.input_shape[static_cast<std::size_t>(request.normalized_dim)];
    } else {
        request.normalized_dim = dim;
        request.fft_length = 0;
    }

    if (n_obj.is_none()) {
        request.n = std::nullopt;
        request.requested_n = request.fft_length;
    } else {
        request.n = nb::cast<int64_t>(n_obj);
        request.requested_n = *request.n;
    }

    if (norm_obj.is_none()) {
        request.norm = "backward";
    } else {
        request.norm = nb::cast<std::string>(norm_obj);
    }

    request.input_dtype = strip_torch_prefix(py_str(input.attr("dtype")));
    request.output_dtype = output_dtype_for(request.input_dtype);

    nb::object device = input.attr("device");
    request.device_type = py_str(device.attr("type"));
    nb::object index_obj = device.attr("index");
    if (!index_obj.is_none()) {
        request.device_index = nb::cast<int64_t>(index_obj);
    } else if (request.device_type == "cuda") {
        nb::module_ torch = nb::module_::import_("torch");
        request.device_index = nb::cast<int64_t>(torch.attr("cuda").attr("current_device")());
    }

    if (request.device_type == "cuda") {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object capability =
            torch.attr("cuda").attr("get_device_capability")(request.device_index);
        int64_t major = nb::cast<int64_t>(capability.attr("__getitem__")(0));
        int64_t minor = nb::cast<int64_t>(capability.attr("__getitem__")(1));
        request.device_arch = "sm_" + std::to_string(major) + std::to_string(minor);
    } else {
        request.device_arch = "none";
    }

    bool contiguous = nb::cast<bool>(input.attr("is_contiguous")());
    request.input_layout = contiguous ? "contiguous" : "strided";
    request.requires_contiguous_copy = !contiguous;

    int64_t numel = product(request.input_shape);
    request.batch = request.requested_n > 0 ? numel / request.requested_n : 0;
    return request;
}

void validate_request(const FFTRequest &request) {
    const std::string op_name = request.direction == "inverse" ? "ifft" : "fft";
    const int64_t ndim = static_cast<int64_t>(request.input_shape.size());
    if (request.direction != "forward" && request.direction != "inverse") {
        raise_python(PyExc_ValueError, "flagfft.fft direction must be 'forward' or 'inverse'");
    }
    if (ndim == 0) {
        raise_python(PyExc_ValueError, "flagfft." + op_name + " expected at least a 1-D tensor");
    }
    if (!(-ndim <= request.raw_dim && request.raw_dim < ndim)) {
        std::ostringstream message;
        message << "Dimension out of range (expected to be in range of [" << -ndim << ", "
                << (ndim - 1) << "], but got " << request.raw_dim << ")";
        raise_python(PyExc_IndexError, message.str());
    }
    if (request.requested_n <= 0) {
        raise_python(PyExc_ValueError,
                     "flagfft." + op_name + " expected n to be a positive integer");
    }
    if (request.n.has_value() && request.requested_n != request.fft_length) {
        raise_python(PyExc_NotImplementedError,
                     "flagfft." + op_name +
                         " currently does not support padding or trimming with n");
    }
    if (request.normalized_dim != ndim - 1) {
        raise_python(PyExc_NotImplementedError,
                     "flagfft." + op_name + " currently supports only the last dimension");
    }
    if (request.norm != "backward" && request.norm != "forward" && request.norm != "ortho") {
        raise_python(PyExc_ValueError,
                     "flagfft." + op_name +
                         " norm must be None, 'backward', 'forward', or 'ortho'");
    }
    if (request.device_type != "cuda") {
        raise_python(PyExc_NotImplementedError,
                     "flagfft." + op_name + " currently supports only CUDA tensors");
    }
    if (request.output_dtype == "unsupported") {
        raise_python(PyExc_TypeError,
                     "flagfft." + op_name +
                         " supports float32, float64, complex64, and complex128 inputs");
    }
}

}  // namespace flagfft
