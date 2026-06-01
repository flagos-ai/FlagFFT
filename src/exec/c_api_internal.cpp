#include "c_api_internal.hpp"

namespace flagfft {

flagfftResult type_metadata(flagfftType type,
                            FlagFFTPrecision &precision,
                            FlagFFTTransformKind &kind,
                            bool &real_input,
                            bool &real_output) {
  switch (type) {
    case FLAGFFT_C2C:
      precision = FlagFFTPrecision::Float32;
      kind = FlagFFTTransformKind::C2C;
      real_input = false;
      real_output = false;
      return FLAGFFT_SUCCESS;
    case FLAGFFT_Z2Z:
      precision = FlagFFTPrecision::Float64;
      kind = FlagFFTTransformKind::Z2Z;
      real_input = false;
      real_output = false;
      return FLAGFFT_SUCCESS;
    case FLAGFFT_R2C:
      precision = FlagFFTPrecision::Float32;
      kind = FlagFFTTransformKind::R2C;
      real_input = true;
      real_output = false;
      return FLAGFFT_SUCCESS;
    case FLAGFFT_D2Z:
      precision = FlagFFTPrecision::Float64;
      kind = FlagFFTTransformKind::D2Z;
      real_input = true;
      real_output = false;
      return FLAGFFT_SUCCESS;
    case FLAGFFT_C2R:
      precision = FlagFFTPrecision::Float32;
      kind = FlagFFTTransformKind::C2R;
      real_input = false;
      real_output = true;
      return FLAGFFT_SUCCESS;
    case FLAGFFT_Z2D:
      precision = FlagFFTPrecision::Float64;
      kind = FlagFFTTransformKind::Z2D;
      real_input = false;
      real_output = true;
      return FLAGFFT_SUCCESS;
  }
  return FLAGFFT_INVALID_TYPE;
}

std::vector<int64_t> copy_dims(const int *values, int rank) {
  std::vector<int64_t> out;
  out.reserve(static_cast<std::size_t>(rank));
  for (int i = 0; i < rank; ++i) {
    out.push_back(static_cast<int64_t>(values[i]));
  }
  return out;
}

bool is_supported_minimal_desc(const FlagFFTPlanDesc &desc) {
  if (desc.rank != 1 || desc.batch <= 0) {
    return false;
  }
  if (desc.type != FLAGFFT_C2C && desc.type != FLAGFFT_Z2Z && desc.type != FLAGFFT_R2C &&
      desc.type != FLAGFFT_D2Z && desc.type != FLAGFFT_C2R && desc.type != FLAGFFT_Z2D) {
    return false;
  }
  if (desc.n.size() != 1 || desc.n[0] <= 0) {
    return false;
  }
  const int64_t n = desc.n[0];
  const bool real_forward = desc.type == FLAGFFT_R2C || desc.type == FLAGFFT_D2Z;
  const bool real_inverse = desc.type == FLAGFFT_C2R || desc.type == FLAGFFT_Z2D;
  const int64_t compact_length = n / 2 + 1;
  const int64_t padded_real_length = 2 * compact_length;
  const int64_t input_length = real_inverse ? compact_length : n;
  const int64_t output_length = real_forward ? compact_length : n;
  const bool valid_input_distance =
      desc.idist == input_length || (real_forward && desc.idist == padded_real_length);
  const bool valid_output_distance =
      desc.odist == output_length || (real_inverse && desc.odist == padded_real_length);
  const bool valid_input_embed =
      desc.inembed.size() == 1 &&
      (desc.inembed[0] == input_length || (real_forward && desc.inembed[0] == padded_real_length));
  const bool valid_output_embed =
      desc.onembed.size() == 1 &&
      (desc.onembed[0] == output_length || (real_inverse && desc.onembed[0] == padded_real_length));
  return desc.istride == 1 && desc.ostride == 1 && valid_input_distance && valid_output_distance &&
         valid_input_embed && valid_output_embed;
}

bool is_supported_2d_desc(const FlagFFTPlanDesc &desc) {
  if (desc.rank != 2 || desc.batch <= 0) {
    return false;
  }
  // Only C2C and Z2Z are supported for 2D FFT
  // R2C/C2R/D2Z/Z2D require special handling that is not yet implemented
  if (desc.type != FLAGFFT_C2C && desc.type != FLAGFFT_Z2Z) {
    return false;
  }
  if (desc.n.size() != 2 || desc.n[0] <= 0 || desc.n[1] <= 0) {
    return false;
  }
  if (desc.istride != 1 || desc.ostride != 1) {
    return false;
  }
  const int64_t n0 = desc.n[0];
  const int64_t n1 = desc.n[1];
  // Pre-plumbed for future 2D real FFT support (R2C/C2R/D2Z/Z2D).
  // The early-return above rejects all real types, so these currently
  // always evaluate to the C2C/Z2Z path.
  const bool real_forward = desc.type == FLAGFFT_R2C || desc.type == FLAGFFT_D2Z;
  const bool real_inverse = desc.type == FLAGFFT_C2R || desc.type == FLAGFFT_Z2D;
  const int64_t half_n1 = n1 / 2 + 1;
  const int64_t input_logical = real_inverse ? n0 * half_n1 : n0 * n1;
  const int64_t output_logical = real_forward ? n0 * half_n1 : n0 * n1;
  if (desc.idist < input_logical || desc.odist < output_logical) {
    return false;
  }
  return true;
}

bool raw_supported_node(const PlanNodePtr &node) {
  if (std::dynamic_pointer_cast<LeafPlanNode>(node) != nullptr) {
    return true;
  }
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    return raw_supported_node(four_step->row_plan) && raw_supported_node(four_step->col_plan);
  }
  if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
    return raw_supported_node(bluestein->fft_plan);
  }
  if (auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(node)) {
    return raw_supported_node(two_dim->row_plan) && raw_supported_node(two_dim->col_plan);
  }
  return false;
}

FFTRequest request_from_desc(const FlagFFTPlanDesc &desc, std::string direction) {
  FFTRequest request;
  const int64_t logical_size = product(desc.n);
  request.fft_length = desc.rank == 1 ? desc.n[0] : logical_size;
  if (desc.rank == 1) {
    request.input_shape = {desc.batch, desc.n[0]};
    request.input_strides = {desc.idist, desc.istride};
    request.raw_dim = 1;
    request.normalized_dim = 1;
  } else {
    request.input_shape = {desc.batch, desc.n[0], desc.n[1]};
    request.input_strides = {desc.idist, desc.n[1] * desc.istride, desc.istride};
    request.raw_dim = 2;
    request.normalized_dim = 2;
  }
  request.n = std::nullopt;
  request.requested_n = request.fft_length;
  request.norm = "backward";
  const bool is_double = desc.precision == FlagFFTPrecision::Float64;
  request.input_dtype = is_double ? "complex128" : "complex64";
  request.output_dtype = is_double ? "complex128" : "complex64";
  request.device_type = adaptor::backend_name();
  request.device_index = desc.device_index;
  request.device_arch = desc.device_arch;
  request.input_layout = "contiguous";
  request.requires_contiguous_copy = false;
  request.direction = std::move(direction);
  request.batch = desc.batch;
  return request;
}

FlagFFTPlan *checked_plan(flagfftHandle handle) {
  if (handle == nullptr || handle->impl == nullptr) {
    return nullptr;
  }
  return static_cast<FlagFFTPlan *>(handle->impl);
}

}  // namespace flagfft
