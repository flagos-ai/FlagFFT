#include <Python.h>
#include <cuda.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr int64_t kPlanSchemaVersion = 1;
constexpr int64_t kDirectDftMaxN = 64;
constexpr int64_t kLeafMaxN = 4096;
constexpr int64_t kLeafSmemBudgetBytes = 48 * 1024;
constexpr int64_t kMaxLanes = 128;
constexpr double kPi = 3.14159265358979323846264338327950288;

const std::vector<int64_t> kSupportedRadices = {17, 16, 13, 11, 9, 8, 7, 6, 5, 4, 3, 2};
const std::vector<int64_t> kSpecializedButterflyRadices = {2, 4, 8, 16};
const std::vector<int64_t> kSpecializedDirectCodeletRadices = {3, 5, 6, 7, 9, 11, 13};

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

std::string run_command_capture_stdout(const std::string &command) {
    std::array<char, 4096> buffer{};
    std::string output;
    int status = 0;
    {
        nb::gil_scoped_release release;
        FILE *pipe = popen(command.c_str(), "r");
        if (pipe == nullptr) {
            throw std::runtime_error("failed to start command: " + command);
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output += buffer.data();
        }
        status = pclose(pipe);
    }
    if (status != 0) {
        std::ostringstream message;
        message << "command failed with status " << status << ": " << command << "\n" << output;
        throw std::runtime_error(message.str());
    }
    return output;
}

std::string json_string_field(const std::string &json, const std::string &field) {
    const std::string key = "\"" + field + "\"";
    std::size_t pos = json.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing JSON field: " + field);
    }
    pos = json.find(':', pos);
    pos = json.find('"', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid JSON string field: " + field);
    }
    std::size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated JSON string field: " + field);
    }
    return json.substr(pos + 1, end - pos - 1);
}

int64_t json_int_field(const std::string &json, const std::string &field) {
    const std::string key = "\"" + field + "\"";
    std::size_t pos = json.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing JSON field: " + field);
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid JSON integer field: " + field);
    }
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    std::size_t end = pos;
    while (end < json.size() && (json[end] == '-' || std::isdigit(static_cast<unsigned char>(json[end])))) {
        ++end;
    }
    return std::stoll(json.substr(pos, end - pos));
}

std::vector<unsigned char> hex_to_bytes(const std::string &hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex string has odd length");
    }
    std::vector<unsigned char> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        unsigned int value = 0;
        std::stringstream stream;
        stream << std::hex << hex.substr(i, 2);
        stream >> value;
        bytes.push_back(static_cast<unsigned char>(value));
    }
    return bytes;
}

void cuda_check(CUresult result, const std::string &context) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << context << " failed";
    if (name != nullptr) {
        out << " (" << name << ")";
    }
    if (message != nullptr) {
        out << ": " << message;
    }
    throw std::runtime_error(out.str());
}

void hash_combine(std::size_t &seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

template <typename T>
void hash_value(std::size_t &seed, const T &value) {
    hash_combine(seed, std::hash<T>{}(value));
}

template <typename T>
void hash_vector(std::size_t &seed, const std::vector<T> &values) {
    hash_value(seed, values.size());
    for (const T &value : values) {
        hash_value(seed, value);
    }
}

struct FFTRequest {
    int64_t fft_length = 0;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    std::optional<int64_t> n;
    int64_t requested_n = 0;
    int64_t raw_dim = -1;
    int64_t normalized_dim = -1;
    std::string norm = "backward";
    std::string input_dtype;
    std::string output_dtype;
    std::string device_type;
    int64_t device_index = -1;
    std::string device_arch;
    std::string input_layout;
    bool requires_contiguous_copy = false;
    std::string direction = "forward";
    int64_t batch = 0;
};

struct PlanKey {
    int64_t fft_length = 0;
    std::vector<int64_t> input_shape;
    std::optional<int64_t> n;
    int64_t requested_n = 0;
    int64_t normalized_dim = -1;
    std::string norm;
    std::string input_dtype;
    std::string output_dtype;
    std::string device_type;
    int64_t device_index = -1;
    std::string device_arch;
    std::vector<int64_t> input_strides;
    std::string input_layout;
    bool requires_contiguous_copy = false;
    std::string direction;

    static PlanKey from_request(const FFTRequest &request) {
        return PlanKey{
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

    bool operator==(const PlanKey &other) const {
        return fft_length == other.fft_length && input_shape == other.input_shape && n == other.n &&
               requested_n == other.requested_n && normalized_dim == other.normalized_dim &&
               norm == other.norm && input_dtype == other.input_dtype &&
               output_dtype == other.output_dtype && device_type == other.device_type &&
               device_index == other.device_index && device_arch == other.device_arch &&
               input_strides == other.input_strides && input_layout == other.input_layout &&
               requires_contiguous_copy == other.requires_contiguous_copy &&
               direction == other.direction;
    }

    std::string repr() const {
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
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey &key) const {
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
};

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

nb::dict key_to_dict(const PlanKey &key) {
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
    out["hash"] = static_cast<uint64_t>(PlanKeyHash{}(key));
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

struct Factorization {
    std::vector<int64_t> factors;
    int64_t remainder = 1;
};

struct PlanNode {
    explicit PlanNode(int64_t length, std::string kind) : length(length), kind(std::move(kind)) {}
    virtual ~PlanNode() = default;
    virtual nb::dict to_dict() const = 0;

    int64_t length;
    std::string kind;
};

using PlanNodePtr = std::shared_ptr<PlanNode>;

struct LeafPlanNode final : PlanNode {
    LeafPlanNode(int64_t length,
                 std::vector<int64_t> factors,
                 int64_t remainder,
                 int64_t lanes,
                 int64_t num_warps,
                 std::vector<int64_t> generic_radices,
                 int64_t smem_size)
        : PlanNode(length, "ct_leaf"),
          factors(std::move(factors)),
          remainder(remainder),
          lanes(lanes),
          num_warps(num_warps),
          generic_radices(std::move(generic_radices)),
          smem_size(smem_size) {}

    nb::dict to_dict() const override {
        nb::dict out;
        out["kind"] = kind;
        out["length"] = length;
        out["factors"] = nb::cast(factors);
        out["remainder"] = remainder;
        out["lanes"] = lanes;
        out["num_warps"] = num_warps;
        out["generic_radices"] = nb::cast(generic_radices);
        out["smem_size"] = smem_size;
        return out;
    }

    std::vector<int64_t> factors;
    int64_t remainder;
    int64_t lanes;
    int64_t num_warps;
    std::vector<int64_t> generic_radices;
    int64_t smem_size;
};

struct DirectDFTPlanNode final : PlanNode {
    explicit DirectDFTPlanNode(int64_t length) : PlanNode(length, "direct_dft") {}

    nb::dict to_dict() const override {
        nb::dict out;
        out["kind"] = kind;
        out["length"] = length;
        out["impl"] = "torch_matmul";
        return out;
    }
};

struct StockhamPlanNode final : PlanNode {
    StockhamPlanNode(int64_t length, std::vector<int64_t> factors)
        : PlanNode(length, "stockham_autosort"), factors(std::move(factors)) {}

    nb::dict to_dict() const override {
        nb::dict out;
        out["kind"] = kind;
        out["length"] = length;
        out["factors"] = nb::cast(factors);
        out["stages"] = nb::list();
        return out;
    }

    std::vector<int64_t> factors;
};

struct FourStepPlanNode final : PlanNode {
    FourStepPlanNode(int64_t length, int64_t n1, int64_t n2, PlanNodePtr row, PlanNodePtr col)
        : PlanNode(length, "four_step"),
          n1(n1),
          n2(n2),
          row_plan(std::move(row)),
          col_plan(std::move(col)) {}

    nb::dict to_dict() const override {
        nb::dict out;
        out["kind"] = kind;
        out["length"] = length;
        out["n1"] = n1;
        out["n2"] = n2;
        out["row"] = row_plan->to_dict();
        out["col"] = col_plan->to_dict();
        return out;
    }

    int64_t n1;
    int64_t n2;
    PlanNodePtr row_plan;
    PlanNodePtr col_plan;
};

struct PlanCandidate {
    PlanNodePtr node;
    double cost = 0.0;
    int priority = 99;
};

class PlanBuilder {
public:
    PlanNodePtr build(int64_t n) {
        if (n <= 0) {
            raise_python(PyExc_ValueError, "FFT length must be positive");
        }
        return build_auto_node(n);
    }

    double cost_for(int64_t n) {
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

    nb::dict wrap_plan_dict(const PlanNodePtr &root, const FFTRequest &request) {
        nb::dict out;
        out["schema_version"] = kPlanSchemaVersion;
        out["source"] = "cpp_auto";
        out["request"] = request_to_dict(request);
        out["estimated_cost"] = cost_for(root->length);
        out["tags"] = nb::dict();
        out["root"] = root->to_dict();
        return out;
    }

private:
    Factorization factorize_supported_radices(int64_t n) {
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

    std::vector<std::vector<int64_t>> enumerate_supported_factorizations(int64_t n) {
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

    std::vector<int64_t> factorize_or_raise(int64_t n) {
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

    int64_t choose_lanes(int64_t n, const std::vector<int64_t> &factors) {
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

    std::vector<int64_t> score_leaf_factorization(int64_t n, const std::vector<int64_t> &factors) {
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

    std::vector<int64_t> select_leaf_factors(int64_t n) {
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

    int64_t choose_num_warps(int64_t lanes) {
        int64_t warps = std::max<int64_t>(1, (lane_block_for(lanes) + 31) / 32);
        int64_t choice = 1;
        while (choice < warps) {
            choice *= 2;
        }
        return std::min<int64_t>(choice, 8);
    }

    int64_t estimate_leaf_smem_bytes(int64_t n, const std::vector<int64_t> &factors) {
        if (factors.size() <= 1) {
            return 0;
        }
        int64_t smem_n = ceil_power_of_two(n);
        return 4 * smem_n * 4;
    }

    bool should_use_leaf(int64_t n, const std::vector<int64_t> &factors) {
        return n <= kLeafMaxN && estimate_leaf_smem_bytes(n, factors) <= kLeafSmemBudgetBytes;
    }

    PlanNodePtr make_leaf_plan(int64_t n, const std::vector<int64_t> &factors, int64_t rem = 1) {
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

    double estimate_leaf_warm_cost(int64_t n) {
        auto factors = select_leaf_factors(n);
        int64_t lanes = choose_lanes(n, factors);
        int64_t warps = choose_num_warps(lanes);
        return static_cast<double>(n * static_cast<int64_t>(factors.size()) * warps) /
               static_cast<double>(lanes);
    }

    double estimate_direct_dft_cost(int64_t n) { return static_cast<double>(n * n); }

    double four_step_cost(int64_t n1, int64_t n2) {
        return static_cast<double>(n2) * cost_for(n1) +
               static_cast<double>(n1) * cost_for(n2) + static_cast<double>(n1 * n2);
    }

    int priority(const PlanNodePtr &node) {
        if (node->kind == "ct_leaf") {
            return 0;
        }
        if (node->kind == "four_step") {
            return 1;
        }
        if (node->kind == "stockham_autosort") {
            return 2;
        }
        if (node->kind == "direct_dft") {
            return 3;
        }
        return 99;
    }

    std::vector<int64_t> enumerate_divisors(int64_t n) {
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

    std::vector<PlanCandidate> build_auto_candidates(int64_t n) {
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

    PlanCandidate select_candidate(const std::vector<PlanCandidate> &candidates) {
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

    PlanNodePtr build_auto_node(int64_t n) {
        auto it = node_cache_.find(n);
        if (it != node_cache_.end()) {
            return it->second;
        }
        PlanCandidate candidate = select_candidate(build_auto_candidates(n));
        node_cache_[n] = candidate.node;
        cost_cache_[n] = candidate.cost;
        return candidate.node;
    }

    std::unordered_map<int64_t, PlanNodePtr> node_cache_;
    std::unordered_map<int64_t, double> cost_cache_;
    std::unordered_map<int64_t, std::vector<int64_t>> divisor_cache_;
    std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> factorization_cache_;
    std::unordered_map<int64_t, std::vector<int64_t>> best_leaf_factors_cache_;
};

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage) {
    std::vector<int64_t> prev_freq;
    int64_t rem = codelet;
    for (int64_t axis = 0; axis < stage; ++axis) {
        prev_freq.push_back(rem % radices[static_cast<std::size_t>(axis)]);
        rem /= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<int64_t> future_time(radices.size(), 0);
    for (int64_t axis = static_cast<int64_t>(radices.size()) - 1; axis > stage; --axis) {
        future_time[static_cast<std::size_t>(axis)] = rem % radices[static_cast<std::size_t>(axis)];
        rem /= radices[static_cast<std::size_t>(axis)];
    }
    if (rem != 0) {
        throw std::runtime_error("stage codelet decode overflow");
    }
    return {prev_freq, future_time};
}

int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit) {
    int64_t value = 0;
    int64_t stride = 1;
    for (std::size_t axis = 0; axis < limit; ++axis) {
        value += digits[axis] * stride;
        stride *= radices[axis];
    }
    return value;
}

std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(
    const std::vector<int64_t> &radices, int64_t stage, int64_t lanes) {
    int64_t n = product(radices);
    int64_t elems_per_lane = n / lanes;
    int64_t radix = radices[static_cast<std::size_t>(stage)];
    int64_t denom = 1;
    for (int64_t axis = 0; axis <= stage; ++axis) {
        denom *= radices[static_cast<std::size_t>(axis)];
    }

    std::vector<float> tw_r(static_cast<std::size_t>(n));
    std::vector<float> tw_i(static_cast<std::size_t>(n));
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t elem = 0; elem < elems_per_lane; ++elem) {
            int64_t group = elem / radix;
            int64_t digit = elem % radix;
            int64_t codelet = lane + lanes * group;
            auto decoded = decode_stage_codelet(codelet, radices, stage);
            int64_t prefix = mixed_radix_value(decoded.first, radices, decoded.first.size());
            double angle = -2.0 * kPi * static_cast<double>(prefix * digit) / static_cast<double>(denom);
            std::size_t index = static_cast<std::size_t>(lane + lanes * elem);
            tw_r[index] = static_cast<float>(std::cos(angle));
            tw_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {tw_r, tw_i};
}

std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix) {
    std::vector<float> dft_r(static_cast<std::size_t>(radix * radix));
    std::vector<float> dft_i(static_cast<std::size_t>(radix * radix));
    for (int64_t k = 0; k < radix; ++k) {
        for (int64_t n = 0; n < radix; ++n) {
            double angle = -2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(radix);
            std::size_t index = static_cast<std::size_t>(k * radix + n);
            dft_r[index] = static_cast<float>(std::cos(angle));
            dft_i[index] = static_cast<float>(std::sin(angle));
        }
    }
    return {dft_r, dft_i};
}

nb::object tensor_from_float_vector(const std::vector<float> &values, const FFTRequest &request) {
    nb::module_ torch = nb::module_::import_("torch");
    std::string device = request.device_type + ":" + std::to_string(request.device_index);
    return torch.attr("tensor")(nb::cast(values), "device"_a = device, "dtype"_a = torch.attr("float32"));
}

struct AotKernel {
    ~AotKernel() {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }

    void load() {
        std::lock_guard<std::mutex> lock(mutex);
        if (function != nullptr) {
            return;
        }
        cuda_check(cuInit(0), "cuInit");
        cuda_check(cuModuleLoadData(&module, cubin.data()), "cuModuleLoadData");
        cuda_check(cuModuleGetFunction(&function, module, kernel_name.c_str()), "cuModuleGetFunction");
        if (shared > 49152) {
            cuda_check(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED),
                       "cuFuncSetCacheConfig");
            cuda_check(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          shared),
                       "cuFuncSetAttribute");
        }
    }

    void launch(CUstream stream, std::vector<CUdeviceptr> &device_args, int32_t nbatch) {
        load();
        CUdeviceptr global_scratch = 0;
        CUdeviceptr profile_scratch = 0;
        std::vector<void *> args;
        args.reserve(device_args.size() + 3);
        for (CUdeviceptr &arg : device_args) {
            args.push_back(&arg);
        }
        args.push_back(&nbatch);
        args.push_back(&global_scratch);
        args.push_back(&profile_scratch);
        cuda_check(cuLaunchKernel(function,
                                  static_cast<unsigned int>(nbatch),
                                  1,
                                  1,
                                  static_cast<unsigned int>(num_warps * 32),
                                  1,
                                  1,
                                  static_cast<unsigned int>(shared),
                                  stream,
                                  args.data(),
                                  nullptr),
                   "cuLaunchKernel");
    }

    std::string kernel_name;
    std::vector<unsigned char> cubin;
    int64_t shared = 0;
    int64_t num_warps = 1;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::mutex mutex;
};

struct CompiledLeafExecutable {
    std::shared_ptr<AotKernel> kernel;
    std::vector<nb::object> tables;
};

class TritonCompiler {
public:
    CompiledLeafExecutable compile_leaf(const LeafPlanNode &leaf, const FFTRequest &request) const {
        std::filesystem::path out_dir = std::filesystem::current_path() / ".flagfft_aot_cache";
        std::string arch = request.device_arch;
        if (arch.rfind("sm_", 0) == 0) {
            arch.erase(0, 3);
        }
        std::string target = "cuda:" + arch + ":32";

        std::wstring executable_w = Py_GetProgramFullPath();
        std::string python_executable(executable_w.begin(), executable_w.end());
        std::ostringstream command;
        command << shell_quote(python_executable)
                << " -m src.triton_aot"
                << " --length " << leaf.length
                << " --factors " << shell_quote(join_ints(leaf.factors))
                << " --lanes " << leaf.lanes
                << " --num-warps " << leaf.num_warps
                << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
                << " --smem-size " << leaf.smem_size
                << " --target " << shell_quote(target)
                << " --out-dir " << shell_quote(out_dir.string());

        std::string artifact_json = run_command_capture_stdout(command.str());
        auto kernel = std::make_shared<AotKernel>();
        kernel->kernel_name = json_string_field(artifact_json, "kernel_name");
        kernel->cubin = hex_to_bytes(json_string_field(artifact_json, "cubin_hex"));
        kernel->shared = json_int_field(artifact_json, "shared");
        kernel->num_warps = json_int_field(artifact_json, "num_warps");

        std::vector<nb::object> tables;
        for (std::size_t stage = 1; stage < leaf.factors.size(); ++stage) {
            auto twiddles = build_stage_twiddles(leaf.factors, static_cast<int64_t>(stage), leaf.lanes);
            tables.push_back(tensor_from_float_vector(twiddles.first, request));
            tables.push_back(tensor_from_float_vector(twiddles.second, request));
        }
        for (int64_t radix : leaf.generic_radices) {
            auto dft = build_dft_matrix(radix);
            tables.push_back(tensor_from_float_vector(dft.first, request));
            tables.push_back(tensor_from_float_vector(dft.second, request));
        }
        return {kernel, tables};
    }
};

enum class ExecutionBackend { AotCudaLeaf, TorchFFT };

struct ExecutablePlan {
    PlanKey key;
    FFTRequest request;
    PlanNodePtr root;
    nb::dict plan_dict;
    ExecutionBackend backend;
    std::shared_ptr<AotKernel> aot_kernel;
    std::vector<nb::object> aot_tables;

    nb::object execute(nb::object input) const {
        if (backend == ExecutionBackend::TorchFFT) {
            nb::module_ torch = nb::module_::import_("torch");
            return torch.attr("fft").attr("fft")(input, "n"_a = request.requested_n,
                                                 "dim"_a = request.normalized_dim,
                                                 "norm"_a = request.norm);
        }

        nb::object exec_input = input;
        if (request.input_dtype == "float32") {
            nb::module_ torch = nb::module_::import_("torch");
            nb::object zeros = torch.attr("zeros_like")(input);
            exec_input = torch.attr("complex")(input, zeros);
        }

        exec_input = exec_input.attr("contiguous")();
        nb::module_ torch = nb::module_::import_("torch");
        nb::object result = torch.attr("empty_like")(exec_input);

        std::vector<CUdeviceptr> device_args;
        device_args.reserve(2 + aot_tables.size());
        device_args.push_back(static_cast<CUdeviceptr>(nb::cast<uint64_t>(exec_input.attr("data_ptr")())));
        device_args.push_back(static_cast<CUdeviceptr>(nb::cast<uint64_t>(result.attr("data_ptr")())));
        for (const nb::object &table : aot_tables) {
            device_args.push_back(static_cast<CUdeviceptr>(nb::cast<uint64_t>(table.attr("data_ptr")())));
        }

        nb::object stream_obj = torch.attr("cuda").attr("current_stream")(request.device_index);
        CUstream stream =
            reinterpret_cast<CUstream>(nb::cast<uint64_t>(stream_obj.attr("cuda_stream")));
        aot_kernel->launch(stream, device_args, static_cast<int32_t>(request.batch));

        if (request.norm == "forward") {
            return result.attr("mul")(1.0 / static_cast<double>(request.requested_n));
        }
        if (request.norm == "ortho") {
            return result.attr("mul")(1.0 / std::sqrt(static_cast<double>(request.requested_n)));
        }
        return result;
    }
};

class PlanCache {
public:
    std::shared_ptr<ExecutablePlan> get_or_create(const PlanKey &key, const FFTRequest &request) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                ++hits_;
                return it->second;
            }
            ++misses_;
        }

        PlanBuilder builder;
        PlanNodePtr root = builder.build(request.requested_n);
        nb::dict plan_dict = builder.wrap_plan_dict(root, request);

        ExecutionBackend backend = ExecutionBackend::TorchFFT;
        std::shared_ptr<AotKernel> aot_kernel;
        std::vector<nb::object> aot_tables;
        if (request.output_dtype == "complex64") {
            auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(root);
            if (leaf == nullptr) {
                raise_python(PyExc_NotImplementedError,
                             "flagfft.fft C++ AOT backend currently supports only ct_leaf plans");
            }
            backend = ExecutionBackend::AotCudaLeaf;
            TritonCompiler compiler;
            CompiledLeafExecutable compiled = compiler.compile_leaf(*leaf, request);
            aot_kernel = std::move(compiled.kernel);
            aot_tables = std::move(compiled.tables);
        }

        auto executable = std::make_shared<ExecutablePlan>(ExecutablePlan{
            key, request, root, plan_dict, backend, std::move(aot_kernel), std::move(aot_tables)});

        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = cache_.emplace(key, executable);
        return inserted ? executable : it->second;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        hits_ = 0;
        misses_ = 0;
    }

    nb::dict info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        nb::dict out;
        out["size"] = cache_.size();
        out["hits"] = hits_;
        out["misses"] = misses_;
        return out;
    }

    nb::list keys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        nb::list out;
        for (const auto &entry : cache_) {
            out.append(key_to_dict(entry.first));
        }
        return out;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<PlanKey, std::shared_ptr<ExecutablePlan>, PlanKeyHash> cache_;
    int64_t hits_ = 0;
    int64_t misses_ = 0;
};

PlanCache &plan_cache() {
    static PlanCache *cache = new PlanCache();
    return *cache;
}

std::shared_ptr<ExecutablePlan> resolve_plan(nb::object input,
                                             nb::object n_obj,
                                             int64_t dim,
                                             nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    PlanKey key = PlanKey::from_request(request);
    validate_request(request);
    return plan_cache().get_or_create(key, request);
}

nb::object fft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    std::shared_ptr<ExecutablePlan> executable = resolve_plan(input, n_obj, dim, norm_obj);
    return executable->execute(input);
}

nb::dict debug_request(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    return request_to_dict(build_request(input, n_obj, dim, norm_obj));
}

nb::dict debug_plan_key(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    return key_to_dict(PlanKey::from_request(request));
}

nb::dict debug_plan(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = builder.build(request.requested_n);
    return builder.wrap_plan_dict(root, request);
}

}  // namespace

NB_MODULE(_flagfft_core, m) {
    m.doc() = "C++ request, plan, and cache core for FlagFFT";

    m.def("fft", &fft, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none(), "Execute a 1-D FFT through the C++ plan cache");
    m.def("debug_request", &debug_request, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("debug_plan_key", &debug_plan_key, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("debug_plan", &debug_plan, "input"_a, "n"_a = nb::none(), "dim"_a = -1,
          "norm"_a = nb::none());
    m.def("clear_plan_cache", []() { plan_cache().clear(); });
    m.def("cache_info", []() { return plan_cache().info(); });
    m.def("cache_keys", []() { return plan_cache().keys(); });
}
