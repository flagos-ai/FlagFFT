#pragma once

#include "flagfft/plan.hpp"

namespace flagfft {

nb::object tensor_from_float_vector(const std::vector<float> &values, const FFTRequest &request);
std::string torch_device_string(const FFTRequest &request);
nb::object tensor_from_complex_vectors(const std::vector<float> &real,
                                       const std::vector<float> &imag,
                                       const FFTRequest &request,
                                       nb::tuple shape);
nb::object empty_complex64_tensor(const FFTRequest &request, nb::tuple shape);
int64_t ceil_div(int64_t numerator, int64_t denominator);
int64_t tensor_numel(const nb::object &tensor);
int64_t tensor_size(const nb::object &tensor, int64_t dim);
int64_t tensor_stride(const nb::object &tensor, int64_t dim);
CUdeviceptr tensor_data_ptr(const nb::object &tensor);
CUstream current_cuda_stream(const FFTRequest &request);
int64_t cuda_device_max_dynamic_shared_memory_bytes(int64_t device_index);

}  // namespace flagfft
