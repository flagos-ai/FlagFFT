#include "flagfft/core.hpp"

namespace flagfft {

[[noreturn]] void raise_python(PyObject *type, const std::string &message) {
    PyErr_SetString(type, message.c_str());
    throw nb::python_error();
}

bool contains(const std::vector<int64_t> &values, int64_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::string py_str(nb::handle object) {
    return nb::cast<std::string>(nb::str(object));
}

std::string strip_torch_prefix(std::string value) {
    constexpr const char *prefix = "torch.";
    if (value.rfind(prefix, 0) == 0) {
        value.erase(0, std::char_traits<char>::length(prefix));
    }
    return value;
}

std::vector<int64_t> int64_vector_from_sequence(nb::handle sequence) {
    std::vector<int64_t> result;
    for (nb::handle item : nb::iter(sequence)) {
        result.push_back(nb::cast<int64_t>(item));
    }
    return result;
}

int64_t product(const std::vector<int64_t> &values) {
    int64_t result = 1;
    for (int64_t value : values) {
        result *= value;
    }
    return result;
}

std::string join_ints(const std::vector<int64_t> &values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    return out.str();
}

int64_t ceil_power_of_two(int64_t value) {
    if (value <= 1) {
        return 1;
    }
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

int64_t lane_block_for(int64_t lanes) {
    return lanes <= 1 ? 1 : ceil_power_of_two(lanes);
}

std::string shell_quote(const std::string &value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out += ch;
        }
    }
    out += "'";
    return out;
}

void hash_combine(std::size_t &seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::string plan_node_kind_name(PlanNodeKind kind) {
    switch (kind) {
        case PlanNodeKind::CtLeaf:
            return "ct_leaf";
        case PlanNodeKind::FourStep:
            return "four_step";
        case PlanNodeKind::DirectDft:
            return "direct_dft";
        case PlanNodeKind::StockhamAutosort:
            return "stockham_autosort";
        case PlanNodeKind::Bluestein:
            return "bluestein";
    }
    return "unknown";
}

std::string kernel_kind_name(KernelKind kind) {
    switch (kind) {
        case KernelKind::Leaf:
            return "leaf";
        case KernelKind::FourStepRow:
            return "four_step_row";
        case KernelKind::FourStepCol:
            return "four_step_col";
        case KernelKind::Transpose:
            return "transpose";
        case KernelKind::TwiddleTranspose:
            return "twiddle_transpose";
        case KernelKind::BluesteinPrepare:
            return "bluestein_prepare";
        case KernelKind::BluesteinPointwise:
            return "bluestein_pointwise";
        case KernelKind::BluesteinFinalize:
            return "bluestein_finalize";
    }
    return "unknown";
}

ProblemKey ProblemKey::from_request(const FFTRequest &request) {
    return ProblemKey{
        request.fft_length,
        request.input_shape,
        request.n,
        request.requested_n,
        request.normalized_dim,
        request.norm,
        request.input_dtype,
        request.output_dtype,
        request.device_type,
        request.device_index,
        request.device_arch,
        request.input_strides,
        request.input_layout,
        request.requires_contiguous_copy,
        request.direction,
    };
}

bool ProblemKey::operator==(const ProblemKey &other) const {
    return fft_length == other.fft_length && input_shape == other.input_shape && n == other.n &&
           requested_n == other.requested_n && normalized_dim == other.normalized_dim &&
           norm == other.norm && input_dtype == other.input_dtype &&
           output_dtype == other.output_dtype && device_type == other.device_type &&
           device_index == other.device_index && device_arch == other.device_arch &&
           input_strides == other.input_strides && input_layout == other.input_layout &&
           requires_contiguous_copy == other.requires_contiguous_copy &&
           direction == other.direction;
}

std::string ProblemKey::repr() const {
    std::ostringstream out;
    out << "fft_length=" << fft_length << ";n=";
    if (n.has_value()) {
        out << *n;
    } else {
        out << "None";
    }
    out << ";requested_n=" << requested_n << ";dim=" << normalized_dim << ";norm=" << norm
        << ";input_dtype=" << input_dtype << ";output_dtype=" << output_dtype
        << ";device=" << device_type << ":" << device_index << ";arch=" << device_arch
        << ";layout=" << input_layout << ";requires_contiguous_copy="
        << (requires_contiguous_copy ? "true" : "false") << ";shape=[";
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << input_shape[i];
    }
    out << "];strides=[";
    for (std::size_t i = 0; i < input_strides.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << input_strides[i];
    }
    out << "];direction=" << direction;
    return out.str();
}

std::size_t ProblemKeyHash::operator()(const ProblemKey &key) const {
    std::size_t seed = 0;
    hash_value(seed, key.fft_length);
    hash_vector(seed, key.input_shape);
    hash_value(seed, key.n.has_value());
    if (key.n.has_value()) {
        hash_value(seed, *key.n);
    }
    hash_value(seed, key.requested_n);
    hash_value(seed, key.normalized_dim);
    hash_value(seed, key.norm);
    hash_value(seed, key.input_dtype);
    hash_value(seed, key.output_dtype);
    hash_value(seed, key.device_type);
    hash_value(seed, key.device_index);
    hash_value(seed, key.device_arch);
    hash_vector(seed, key.input_strides);
    hash_value(seed, key.input_layout);
    hash_value(seed, key.requires_contiguous_copy);
    hash_value(seed, key.direction);
    return seed;
}

bool PlanKey::operator==(const PlanKey &other) const {
    return schema_version == other.schema_version && root_kind == other.root_kind &&
           length == other.length && factors == other.factors && remainder == other.remainder &&
           lanes == other.lanes && num_warps == other.num_warps &&
           generic_radices == other.generic_radices && smem_size == other.smem_size &&
           n1 == other.n1 && n2 == other.n2 && conv_length == other.conv_length &&
           child_keys == other.child_keys;
}

std::string PlanKey::repr() const {
    std::ostringstream out;
    out << "schema=" << schema_version << ";kind=" << plan_node_kind_name(root_kind)
        << ";length=" << length;
    if (!factors.empty()) {
        out << ";factors=[" << join_ints(factors) << "]";
    }
    if (root_kind == PlanNodeKind::CtLeaf) {
        out << ";remainder=" << remainder << ";lanes=" << lanes
            << ";num_warps=" << num_warps << ";generic_radices=["
            << join_ints(generic_radices) << "];smem_size=" << smem_size;
    }
    if (root_kind == PlanNodeKind::FourStep) {
        out << ";n1=" << n1 << ";n2=" << n2;
    }
    if (root_kind == PlanNodeKind::Bluestein) {
        out << ";conv_length=" << conv_length;
    }
    if (!child_keys.empty()) {
        out << ";children=[";
        for (std::size_t i = 0; i < child_keys.size(); ++i) {
            if (i != 0) {
                out << "|";
            }
            out << "{" << child_keys[i] << "}";
        }
        out << "]";
    }
    return out.str();
}

std::size_t PlanKeyHash::operator()(const PlanKey &key) const {
    std::size_t seed = 0;
    hash_value(seed, key.schema_version);
    hash_value(seed, static_cast<int64_t>(key.root_kind));
    hash_value(seed, key.length);
    hash_vector(seed, key.factors);
    hash_value(seed, key.remainder);
    hash_value(seed, key.lanes);
    hash_value(seed, key.num_warps);
    hash_vector(seed, key.generic_radices);
    hash_value(seed, key.smem_size);
    hash_value(seed, key.n1);
    hash_value(seed, key.n2);
    hash_value(seed, key.conv_length);
    hash_vector(seed, key.child_keys);
    return seed;
}

KernelKey KernelKey::leaf(std::string target,
                          int64_t length,
                          std::vector<int64_t> factors,
                          int64_t lanes,
                          int64_t num_warps,
                          std::vector<int64_t> generic_radices,
                          int64_t smem_size) {
    KernelKey key;
    key.kind = KernelKind::Leaf;
    key.target = std::move(target);
    key.length = length;
    key.factors = std::move(factors);
    key.lanes = lanes;
    key.num_warps = num_warps;
    key.generic_radices = std::move(generic_radices);
    key.smem_size = smem_size;
    return key;
}

KernelKey KernelKey::four_step_row(std::string target,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
    KernelKey key = KernelKey::leaf(std::move(target),
                                    length,
                                    std::move(factors),
                                    lanes,
                                    num_warps,
                                    std::move(generic_radices),
                                    smem_size);
    key.kind = KernelKind::FourStepRow;
    key.four_step_n1 = n1;
    key.four_step_n2 = n2;
    return key;
}

KernelKey KernelKey::four_step_col(std::string target,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
    KernelKey key = KernelKey::leaf(std::move(target),
                                    length,
                                    std::move(factors),
                                    lanes,
                                    num_warps,
                                    std::move(generic_radices),
                                    smem_size);
    key.kind = KernelKind::FourStepCol;
    key.four_step_n1 = n1;
    key.four_step_n2 = n2;
    return key;
}

KernelKey KernelKey::transpose(std::string target) {
    KernelKey key;
    key.kind = KernelKind::Transpose;
    key.target = std::move(target);
    return key;
}

KernelKey KernelKey::twiddle_transpose(std::string target) {
    KernelKey key;
    key.kind = KernelKind::TwiddleTranspose;
    key.target = std::move(target);
    return key;
}

KernelKey KernelKey::bluestein_prepare(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinPrepare;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

KernelKey KernelKey::bluestein_pointwise(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinPointwise;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

KernelKey KernelKey::bluestein_finalize(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinFinalize;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

bool KernelKey::operator==(const KernelKey &other) const {
    return kind == other.kind && target == other.target && length == other.length &&
           factors == other.factors && lanes == other.lanes &&
           num_warps == other.num_warps && generic_radices == other.generic_radices &&
           smem_size == other.smem_size && four_step_n1 == other.four_step_n1 &&
           four_step_n2 == other.four_step_n2 && bluestein_n == other.bluestein_n &&
           bluestein_m == other.bluestein_m;
}

std::string KernelKey::repr() const {
    std::ostringstream out;
    out << "kind=" << kernel_kind_name(kind) << ";target=" << target;
    if (kind == KernelKind::Leaf || kind == KernelKind::FourStepRow ||
        kind == KernelKind::FourStepCol) {
        out << ";length=" << length << ";factors=[" << join_ints(factors) << "]"
            << ";lanes=" << lanes << ";num_warps=" << num_warps
            << ";generic_radices=[" << join_ints(generic_radices)
            << "];smem_size=" << smem_size;
        if (kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol) {
            out << ";four_step_n1=" << four_step_n1 << ";four_step_n2=" << four_step_n2;
        }
    }
    if (kind == KernelKind::BluesteinPrepare || kind == KernelKind::BluesteinPointwise ||
        kind == KernelKind::BluesteinFinalize) {
        out << ";bluestein_n=" << bluestein_n << ";bluestein_m=" << bluestein_m;
    }
    return out.str();
}

std::size_t KernelKeyHash::operator()(const KernelKey &key) const {
    std::size_t seed = 0;
    hash_value(seed, static_cast<int64_t>(key.kind));
    hash_value(seed, key.target);
    hash_value(seed, key.length);
    hash_vector(seed, key.factors);
    hash_value(seed, key.lanes);
    hash_value(seed, key.num_warps);
    hash_vector(seed, key.generic_radices);
    hash_value(seed, key.smem_size);
    hash_value(seed, key.four_step_n1);
    hash_value(seed, key.four_step_n2);
    hash_value(seed, key.bluestein_n);
    hash_value(seed, key.bluestein_m);
    return seed;
}

nb::dict request_to_dict(const FFTRequest &request) {
    nb::dict out;
    out["op"] = "fft";
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

std::string output_dtype_for(const std::string &input_dtype) {
    if (input_dtype == "complex64" || input_dtype == "float32") {
        return "complex64";
    }
    if (input_dtype == "complex128" || input_dtype == "float64") {
        return "complex128";
    }
    return "unsupported";
}

FFTRequest build_request(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    FFTRequest request;
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
    const int64_t ndim = static_cast<int64_t>(request.input_shape.size());
    if (ndim == 0) {
        raise_python(PyExc_ValueError, "flagfft.fft expected at least a 1-D tensor");
    }
    if (!(-ndim <= request.raw_dim && request.raw_dim < ndim)) {
        std::ostringstream message;
        message << "Dimension out of range (expected to be in range of [" << -ndim << ", "
                << (ndim - 1) << "], but got " << request.raw_dim << ")";
        raise_python(PyExc_IndexError, message.str());
    }
    if (request.requested_n <= 0) {
        raise_python(PyExc_ValueError, "flagfft.fft expected n to be a positive integer");
    }
    if (request.n.has_value() && request.requested_n != request.fft_length) {
        raise_python(PyExc_NotImplementedError,
                     "flagfft.fft currently does not support padding or trimming with n");
    }
    if (request.normalized_dim != ndim - 1) {
        raise_python(PyExc_NotImplementedError,
                     "flagfft.fft currently supports only the last dimension");
    }
    if (request.norm != "backward" && request.norm != "forward" && request.norm != "ortho") {
        raise_python(PyExc_ValueError,
                     "flagfft.fft norm must be None, 'backward', 'forward', or 'ortho'");
    }
    if (request.device_type != "cuda") {
        raise_python(PyExc_NotImplementedError, "flagfft.fft currently supports only CUDA tensors");
    }
    if (request.output_dtype == "unsupported") {
        raise_python(PyExc_TypeError,
                     "flagfft.fft supports float32, float64, complex64, and complex128 inputs");
    }
}

}  // namespace flagfft
