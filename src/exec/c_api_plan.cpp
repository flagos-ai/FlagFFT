#include "adaptor/adaptor.h"
#include "c_api_internal.hpp"
#include "flagfft/tune_json.hpp"
#include "plan/rader_utils.hpp"

namespace flagfft {
namespace {

  PlanNodePtr raw_compatible_bluestein_plan(int64_t length, PlanBuilder &builder, const FFTRequest &request) {
    int64_t conv_length = ceil_power_of_two(2 * length - 1);
    PlanNodePtr child = builder.build(conv_length, request);
    PlanNodePtr candidate = std::make_shared<BluesteinPlanNode>(length, conv_length, std::move(child));
    return raw_supported_node(candidate) ? candidate : nullptr;
  }

  PlanNodePtr raw_compatible_rader_plan(int64_t length, PlanBuilder &builder, const FFTRequest &request) {
    if (!is_prime_length(length) || length <= kDirectDftMaxN) {
      return nullptr;
    }
    int64_t root = find_primitive_root(length);
    std::vector<int64_t> idx = build_rader_index_table(length, root);
    PlanNodePtr child = builder.build(length - 1, request);
    PlanNodePtr candidate = std::make_shared<RaderPlanNode>(length, root, std::move(idx), std::move(child));
    return raw_supported_node(candidate) ? candidate : nullptr;
  }

  PlanNodePtr lookup_or_build_root(PlanBuilder &builder, const FFTRequest &request) {
    auto tuned = lookup_tuned_plan_json(request);
    if (tuned.has_value()) {
      try {
        PlanNodePtr root = plan_node_from_json(builder, tuned->at("root"));
        if (raw_supported_node(root)) {
          return root;
        }
      } catch (const std::exception &) {
      }
    }
    return builder.build(request.requested_n, request);
  }

}  // namespace

flagfftResult build_plan(flagfftHandle *out, FlagFFTPlanDesc desc) {
  if (out == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  *out = nullptr;

  // Check if this is a supported 2D descriptor
  const bool is_2d = is_supported_2d_desc(desc);
  if (!is_supported_minimal_desc(desc) && !is_2d) {
    return FLAGFFT_NOT_SUPPORTED;
  }

  flagfftResult device_result = adaptor::ensure_device(desc.device_index, desc.device_arch);
  if (device_result != FLAGFFT_SUCCESS) {
    return device_result;
  }

  try {
    auto plan = std::make_unique<FlagFFTPlan>();
    plan->desc = std::move(desc);

    PlanBuilder builder;

    if (is_2d) {
      const int64_t n0 = plan->desc.n[0];
      const int64_t n1 = plan->desc.n[1];
      const int64_t batch = plan->desc.batch;
      const int64_t half_n1 = n1 / 2 + 1;
      const bool real_forward = plan->desc.type == FLAGFFT_R2C || plan->desc.type == FLAGFFT_D2Z;
      const bool real_inverse = plan->desc.type == FLAGFFT_C2R || plan->desc.type == FLAGFFT_Z2D;
      const flagfftType c2c_type =
          (plan->desc.type == FLAGFFT_Z2D || plan->desc.type == FLAGFFT_D2Z || plan->desc.type == FLAGFFT_Z2Z)
              ? FLAGFFT_Z2Z
              : FLAGFFT_C2C;

      // For R2C/D2Z:
      //   Row: R2C along n1 (batch*n0), col: C2C along n0 (batch*half_n1)
      // For C2R/Z2D:
      //   Col: C2C along n0 (batch*half_n1), row: C2C along n1 (batch*n0)
      // Both use "forward" for R2C, "inverse" for C2R

      // Build row descriptor
      FlagFFTPlanDesc row_desc = plan->desc;
      row_desc.rank = 1;
      row_desc.n = {n1};
      row_desc.istride = 1;
      row_desc.ostride = 1;
      if (real_forward) {
        // R2C/D2Z row: real input -> half-packed complex output
        row_desc.type = plan->desc.type;
        row_desc.inembed = {n1};
        row_desc.onembed = {half_n1};
        row_desc.idist = n1;
        row_desc.odist = half_n1;
        row_desc.batch = batch * n0;
      } else if (real_inverse) {
        // C2R/Z2D row: C2C IFFT along n1 on full complex data
        row_desc.type = c2c_type;
        row_desc.inembed = {n1};
        row_desc.onembed = {n1};
        row_desc.idist = n1;
        row_desc.odist = n1;
        row_desc.batch = batch * n0;
      } else {
        row_desc.type = c2c_type;
        row_desc.inembed = {n1};
        row_desc.onembed = {n1};
        row_desc.idist = n1;
        row_desc.odist = n1;
        row_desc.batch = batch * n0;
      }

      FFTRequest row_forward_request = request_from_desc(row_desc, "forward");
      PlanNodePtr row_plan = lookup_or_build_root(builder, row_forward_request);
      if (!raw_supported_node(row_plan)) {
        row_plan = lookup_or_build_root(builder, request_from_desc(row_desc, "inverse"));
      }
      if (!raw_supported_node(row_plan)) {
        row_plan = raw_compatible_rader_plan(n1, builder, row_forward_request);
      }
      if (!raw_supported_node(row_plan)) {
        row_plan = raw_compatible_bluestein_plan(n1, builder, row_forward_request);
      }
      if (!raw_supported_node(row_plan)) {
        return FLAGFFT_NOT_SUPPORTED;
      }

      // Build column descriptor (always C2C/Z2Z)
      FlagFFTPlanDesc col_desc = plan->desc;
      col_desc.rank = 1;
      col_desc.type = c2c_type;
      col_desc.n = {n0};
      col_desc.inembed = {n0};
      col_desc.onembed = {n0};
      col_desc.istride = 1;
      col_desc.ostride = 1;
      col_desc.idist = n0;
      col_desc.odist = n0;
      col_desc.batch = batch * half_n1;

      // For C2R/Z2D, row FFT should be inverse (IFFT)
      // For R2C/D2Z or C2C/Z2Z, row FFT should be forward
      const std::string row_direction = real_inverse ? "inverse" : "forward";
      FFTRequest row_request = request_from_desc(row_desc, row_direction);
      PlanNodePtr row_plan = lookup_or_build_root(builder, row_request);
      if (!raw_supported_node(row_plan)) {
        row_plan = lookup_or_build_root(
            builder,
            request_from_desc(row_desc, row_direction == "forward" ? "inverse" : "forward"));
      }
      if (!raw_supported_node(row_plan)) {
        row_plan = raw_compatible_bluestein_plan(n1, builder, row_request);
      }
      if (!raw_supported_node(row_plan)) {
        return FLAGFFT_NOT_SUPPORTED;
      }

      // For C2R/Z2D, column FFT should be inverse (IFFT)
      // For R2C/D2Z or C2C/Z2Z, column FFT should be forward
      const std::string col_direction = real_inverse ? "inverse" : "forward";
      FFTRequest col_request = request_from_desc(col_desc, col_direction);
      PlanNodePtr col_plan = lookup_or_build_root(builder, col_request);
      if (!raw_supported_node(col_plan)) {
        col_plan = lookup_or_build_root(
            builder,
            request_from_desc(col_desc, col_direction == "forward" ? "inverse" : "forward"));
      }
      if (!raw_supported_node(col_plan)) {
        col_plan = raw_compatible_rader_plan(n0, builder, col_forward_request);
      }
      if (!raw_supported_node(col_plan)) {
        col_plan = raw_compatible_bluestein_plan(n0, builder, col_forward_request);
      }
      if (!raw_supported_node(col_plan)) {
        return FLAGFFT_NOT_SUPPORTED;
      }

      plan->executable.root =
          std::make_shared<TwoDimPlanNode>(n0, n1, TwoDimStrategy::RTRT, row_plan, col_plan);

      plan->executable.forward_request = request_from_desc(plan->desc, "forward");
      plan->executable.inverse_request = request_from_desc(plan->desc, "inverse");
    } else {
      // 1D FFT: original path
      plan->executable.forward_request = request_from_desc(plan->desc, "forward");
      plan->executable.inverse_request = request_from_desc(plan->desc, "inverse");

      plan->executable.root = lookup_or_build_root(builder, plan->executable.forward_request);
      if (!raw_supported_node(plan->executable.root)) {
        plan->executable.root = lookup_or_build_root(builder, plan->executable.inverse_request);
      }
      if (!raw_supported_node(plan->executable.root)) {
        if (PlanNodePtr fallback = raw_compatible_rader_plan(plan->executable.forward_request.requested_n,
                                                             builder,
                                                             plan->executable.forward_request)) {
          plan->executable.root = std::move(fallback);
        }
      }
      if (!raw_supported_node(plan->executable.root)) {
        if (PlanNodePtr fallback = raw_compatible_bluestein_plan(plan->executable.forward_request.requested_n,
                                                                 builder,
                                                                 plan->executable.forward_request)) {
          plan->executable.root = std::move(fallback);
        }
      }
      if (!raw_supported_node(plan->executable.root)) {
        return FLAGFFT_NOT_SUPPORTED;
      }
    }

    plan->executable.forward_problem_key = ProblemKey::from_request(plan->executable.forward_request);
    plan->executable.inverse_problem_key = ProblemKey::from_request(plan->executable.inverse_request);
    plan->executable.plan_key = PlanKey::from_node(plan->executable.root);

    TritonCompiler compiler;
    if (is_2d) {
      auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(plan->executable.root);
      if (plan->desc.type == FLAGFFT_R2C || plan->desc.type == FLAGFFT_D2Z) {
        plan->executable.forward =
            compiler.compile_raw_2d_r2c_node(two_dim, plan->executable.forward_request, plan->desc.batch);
      } else if (plan->desc.type == FLAGFFT_C2R || plan->desc.type == FLAGFFT_Z2D) {
        plan->executable.inverse =
            compiler.compile_raw_2d_c2r_node(two_dim, plan->executable.inverse_request, plan->desc.batch);
      } else {
        plan->executable.forward =
            compiler.compile_raw_2d_node(two_dim, plan->executable.forward_request, plan->desc.batch);
        plan->executable.inverse =
            compiler.compile_raw_2d_node(two_dim, plan->executable.inverse_request, plan->desc.batch);
      }
    } else {
      // 1D FFT: original compilation
      if (plan->desc.type == FLAGFFT_R2C || plan->desc.type == FLAGFFT_D2Z) {
        plan->executable.forward = compiler.compile_raw_r2c_node(plan->executable.root,
                                                                 plan->executable.forward_request,
                                                                 plan->desc.batch);
      } else if (plan->desc.type == FLAGFFT_C2R || plan->desc.type == FLAGFFT_Z2D) {
        plan->executable.inverse = compiler.compile_raw_c2r_node(plan->executable.root,
                                                                 plan->executable.inverse_request,
                                                                 plan->desc.batch);
      } else {
        plan->executable.forward = compiler.compile_raw_node(plan->executable.root,
                                                             plan->executable.forward_request,
                                                             plan->desc.batch);
        plan->executable.inverse = compiler.compile_raw_node(plan->executable.root,
                                                             plan->executable.inverse_request,
                                                             plan->desc.batch);
      }
    }
    plan->state.initialized = true;

    std::unique_ptr<flagfftPlan_t> handle(new flagfftPlan_t());
    handle->impl = plan.release();
    *out = handle.release();
    return FLAGFFT_SUCCESS;
  } catch (const std::bad_alloc &) {
    return FLAGFFT_ALLOC_FAILED;
  } catch (const std::exception &) {
    return FLAGFFT_SETUP_FAILED;
  }
}

}  // namespace flagfft

extern "C" flagfftResult flagfftPlan1d(flagfftHandle *plan, int nx, flagfftType type, int batch) {
  int n[1] = {nx};
  const bool real_forward = type == FLAGFFT_R2C || type == FLAGFFT_D2Z;
  const bool real_inverse = type == FLAGFFT_C2R || type == FLAGFFT_Z2D;
  const int half = nx / 2 + 1;
  const int idist = real_inverse ? half : nx;
  const int odist = real_forward ? half : nx;
  return flagfftPlanMany(plan, 1, n, nullptr, 1, idist, nullptr, 1, odist, type, batch);
}

extern "C" flagfftResult flagfftPlan2d(flagfftHandle *plan, int nx, int ny, flagfftType type) {
  int n[2] = {nx, ny};
  const bool real_forward = type == FLAGFFT_R2C || type == FLAGFFT_D2Z;
  const bool real_inverse = type == FLAGFFT_C2R || type == FLAGFFT_Z2D;
  const int half_ny = ny / 2 + 1;
  const int idist = real_inverse ? nx * half_ny : nx * ny;
  const int odist = real_forward ? nx * half_ny : nx * ny;
  return flagfftPlanMany(plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, type, 1);
}

extern "C" flagfftResult flagfftPlan3d(flagfftHandle *plan, int nx, int ny, int nz, flagfftType type) {
  int n[3] = {nx, ny, nz};
  return flagfftPlanMany(plan, 3, n, nullptr, 1, nx * ny * nz, nullptr, 1, nx * ny * nz, type, 1);
}

extern "C" flagfftResult flagfftPlanMany(flagfftHandle *plan,
                                         int rank,
                                         int *n,
                                         int *inembed,
                                         int istride,
                                         int idist,
                                         int *onembed,
                                         int ostride,
                                         int odist,
                                         flagfftType type,
                                         int batch) {
  if (plan == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  *plan = nullptr;
  if (rank <= 0 || n == nullptr || batch <= 0) {
    return FLAGFFT_INVALID_VALUE;
  }
  for (int i = 0; i < rank; ++i) {
    if (n[i] <= 0) {
      return FLAGFFT_INVALID_SIZE;
    }
  }

  flagfft::FlagFFTPlanDesc desc;
  desc.rank = rank;
  desc.n = flagfft::copy_dims(n, rank);
  desc.istride = istride;
  desc.idist = idist;
  desc.ostride = ostride;
  desc.odist = odist;
  desc.batch = batch;
  desc.type = type;

  flagfftResult type_result =
      flagfft::type_metadata(type, desc.precision, desc.kind, desc.real_input, desc.real_output);
  if (type_result != FLAGFFT_SUCCESS) {
    return type_result;
  }

  if (inembed != nullptr) {
    desc.inembed = flagfft::copy_dims(inembed, rank);
  } else if ((type == FLAGFFT_C2R || type == FLAGFFT_Z2D) && rank == 1) {
    desc.inembed = {desc.n[0] / 2 + 1};
  } else if ((type == FLAGFFT_C2R || type == FLAGFFT_Z2D) && rank == 2) {
    // 2D C2R/Z2D: input embed is (n0, n1/2+1)
    desc.inembed = {desc.n[0], desc.n[1] / 2 + 1};
  } else {
    desc.inembed = desc.n;
  }
  if (onembed != nullptr) {
    desc.onembed = flagfft::copy_dims(onembed, rank);
  } else if ((type == FLAGFFT_R2C || type == FLAGFFT_D2Z) && rank == 1) {
    desc.onembed = {desc.n[0] / 2 + 1};
  } else if ((type == FLAGFFT_R2C || type == FLAGFFT_D2Z) && rank == 2) {
    // 2D R2C/D2Z: output embed is (n0, n1/2+1)
    desc.onembed = {desc.n[0], desc.n[1] / 2 + 1};
  } else {
    desc.onembed = desc.n;
  }

  return flagfft::build_plan(plan, std::move(desc));
}
