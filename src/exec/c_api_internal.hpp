#pragma once

#include "flagfft/core.hpp"

struct flagfftPlan_t {
  void *impl = nullptr;
};

namespace flagfft {

enum class FlagFFTPrecision { Float32, Float64 };
enum class FlagFFTTransformKind { C2C, Z2Z, R2C, D2Z, C2R, Z2D };

struct FlagFFTPlanDesc {
  int rank = 0;
  std::vector<int64_t> n;
  std::vector<int64_t> inembed;
  int64_t istride = 1;
  int64_t idist = 0;
  std::vector<int64_t> onembed;
  int64_t ostride = 1;
  int64_t odist = 0;
  int64_t batch = 1;
  flagfftType type = FLAGFFT_C2C;
  FlagFFTPrecision precision = FlagFFTPrecision::Float32;
  FlagFFTTransformKind kind = FlagFFTTransformKind::C2C;
  bool real_input = false;
  bool real_output = false;
  int device_index = 0;
  std::string device_arch;
};

struct FlagFFTPlanState {
  bool initialized = false;
  bool destroyed = false;
  adaptor::StreamHandle stream = nullptr;
  flagfftResult last_error = FLAGFFT_SUCCESS;
};

struct FlagFFTExecutable {
  FFTRequest forward_request;
  FFTRequest inverse_request;
  ProblemKey forward_problem_key;
  ProblemKey inverse_problem_key;
  PlanKey plan_key;
  PlanNodePtr root;
  std::shared_ptr<CompiledRawNode> forward;
  std::shared_ptr<CompiledRawNode> inverse;
};

struct FlagFFTPlan {
  FlagFFTPlanDesc desc;
  FlagFFTPlanState state;
  FlagFFTExecutable executable;
  mutable std::string description_cache;
  std::mutex mutex;
};

flagfftResult type_metadata(flagfftType type,
                            FlagFFTPrecision &precision,
                            FlagFFTTransformKind &kind,
                            bool &real_input,
                            bool &real_output);
std::vector<int64_t> copy_dims(const int *values, int rank);
bool is_supported_minimal_desc(const FlagFFTPlanDesc &desc);
bool is_supported_2d_desc(const FlagFFTPlanDesc &desc);
bool raw_supported_node(const PlanNodePtr &node);
FFTRequest request_from_desc(const FlagFFTPlanDesc &desc, std::string direction);
flagfftResult build_plan(flagfftHandle *out, FlagFFTPlanDesc desc);
FlagFFTPlan *checked_plan(flagfftHandle handle);

}  // namespace flagfft
