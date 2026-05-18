#include "flagfft/core.hpp"

namespace flagfft {

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

}  // namespace flagfft
