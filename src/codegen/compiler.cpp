// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "flagfft/maca_tail_policy.hpp"

#include <cstdlib>
#include <cmath>
#include <optional>
#include <utility>

namespace flagfft {
namespace {

  bool flag_or_default(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    return value == nullptr ? default_value : std::string(value) == "1";
  }

  bool use_npu_2d_cube_dft(const FFTRequest &request, int64_t n) {
    const char *setting = std::getenv("FLAGFFT_NPU_2D_CUBE_DFT");
    return setting != nullptr && std::string(setting) == "1" && request.device_type == "npu" &&
           request.origin_rank == 2 && request.input_dtype == "complex64" &&
           request.output_dtype == "complex64" && n == 64;
  }

  void build_npu_aiv_fft64_tables(const FFTRequest &request,
                                  std::vector<uint32_t> &indices,
                                  std::vector<float> &twiddles,
                                  NpuAivFFT64Mode mode = NpuAivFFT64Mode::Complex) {
    constexpr int64_t n = 64;
    constexpr int64_t stages = 6;
    constexpr int64_t output_index_base = 2 * n + 2 * stages * n;
    constexpr double pi = 3.141592653589793238462643383279502884;
    indices.assign(output_index_base + 2 * n, 0);
    twiddles.assign(2 * stages * n + (mode == NpuAivFFT64Mode::RealInverse ? n : 0), 0.0f);
    const double sign = request.direction == "inverse" ? 1.0 : -1.0;

    for (int64_t i = 0; i < n; ++i) {
      int64_t value = i;
      int64_t reversed = 0;
      for (int64_t bit = 0; bit < stages; ++bit) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
      }
      if (mode == NpuAivFFT64Mode::RealForward) {
        indices[i] = static_cast<uint32_t>(reversed * sizeof(float));
      } else if (mode == NpuAivFFT64Mode::RealInverse) {
        const int64_t compact_bin = reversed <= n / 2 ? reversed : n - reversed;
        const int64_t compact_scalar = 2 * compact_bin;
        indices[i] = static_cast<uint32_t>(compact_scalar * sizeof(float));
        indices[n + i] = static_cast<uint32_t>((compact_scalar + 1) * sizeof(float));
        twiddles[2 * stages * n + i] = reversed > n / 2 ? -1.0f : 1.0f;
      } else {
        // Gather offsets are byte offsets into the interleaved complex input.
        indices[i] = static_cast<uint32_t>(reversed * 2 * sizeof(float));
        // A strided DataCopyPad stores each complex point at the start of a
        // 32-byte VECIN block.
        indices[n + i] = static_cast<uint32_t>(reversed * 8 * sizeof(float));
      }
    }

    for (int64_t i = 0; i < 2 * n; ++i) {
      const int64_t row = i / 2;
      const int64_t component = i % 2;
      if (mode == NpuAivFFT64Mode::RealInverse) {
        if (i < n) indices[output_index_base + i] = static_cast<uint32_t>(i * sizeof(float));
      } else {
        indices[output_index_base + i] =
            static_cast<uint32_t>((row + component * n) * sizeof(float));
      }
    }

    for (int64_t stage = 0; stage < stages; ++stage) {
      const int64_t length = int64_t{1} << (stage + 1);
      const int64_t half = length / 2;
      const int64_t a_base = 2 * n + stage * n;
      const int64_t b_base = 2 * n + stages * n + stage * n;
      const int64_t imag_base = stages * n;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t group = (i / length) * length;
        const int64_t offset = i % length;
        const bool upper = offset >= half;
        const int64_t a = group + (upper ? offset - half : offset);
        const int64_t b = a + half;
        const int64_t twiddle_offset = offset % half;
        const double angle = sign * 2.0 * pi * static_cast<double>(twiddle_offset) /
                             static_cast<double>(length);
        const float negate = upper ? -1.0f : 1.0f;
        indices[a_base + i] = static_cast<uint32_t>(a * sizeof(float));
        indices[b_base + i] = static_cast<uint32_t>(b * sizeof(float));
        twiddles[stage * n + i] = negate * static_cast<float>(std::cos(angle));
        twiddles[imag_base + stage * n + i] = negate * static_cast<float>(std::sin(angle));
      }
    }

  }

  void build_npu_aiv_fft64_group_tables(const FFTRequest &request,
                                        int64_t group_size,
                                        bool strided_columns,
                                        std::vector<uint32_t> &indices,
                                        std::vector<float> &twiddles,
                                        NpuAivFFT64Mode mode = NpuAivFFT64Mode::Complex) {
    constexpr int64_t n = 64;
    constexpr int64_t stages = 6;
    const int64_t compute_group_size = group_size == 8 ? 4 : group_size;
    const int64_t group_n = n * compute_group_size;
    const int64_t output_index_base = group_n;
    const int64_t stage_index_base = 3 * group_n;
    const int64_t stage_b_index_base = stage_index_base + stages * group_n;
    const int64_t group_index_count = 3 * group_n + 2 * stages * group_n;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int64_t second_input_base = group_index_count;
    const int64_t imag_input_base = group_index_count + (group_size == 8 ? group_n : 0);
    indices.assign(imag_input_base +
                       (mode == NpuAivFFT64Mode::RealInverse
                            ? group_n * (group_size == 8 ? 2 : 1)
                            : 0),
                   0);
    twiddles.assign(2 * stages * group_n +
                        (mode == NpuAivFFT64Mode::RealInverse ? group_n : 0),
                    0.0f);
    const double sign = request.direction == "inverse" ? 1.0 : -1.0;

    for (int64_t group = 0; group < compute_group_size; ++group) {
      for (int64_t i = 0; i < n; ++i) {
        int64_t value = i;
        int64_t reversed = 0;
        for (int64_t bit = 0; bit < stages; ++bit) {
          reversed = (reversed << 1) | (value & 1);
          value >>= 1;
        }
        if (mode == NpuAivFFT64Mode::RealForward) {
          indices[group * n + i] =
              static_cast<uint32_t>((group * n + reversed) * sizeof(float));
        } else if (mode == NpuAivFFT64Mode::RealInverse) {
          const int64_t compact_bin = reversed <= n / 2 ? reversed : n - reversed;
          const int64_t compact_complex_index = group * (n / 2 + 1) + compact_bin;
          const int64_t destination = group * n + i;
          indices[destination] =
              static_cast<uint32_t>(2 * compact_complex_index * sizeof(float));
          indices[imag_input_base + destination] =
              static_cast<uint32_t>((2 * compact_complex_index + 1) * sizeof(float));
          twiddles[2 * stages * group_n + destination] =
              reversed > n / 2 ? -1.0f : 1.0f;
        } else {
          const int64_t input_complex_index = strided_columns
              ? reversed * group_size + group
              : group * n + reversed;
          indices[group * n + i] =
              static_cast<uint32_t>(input_complex_index * 2 * sizeof(float));
        }
      }
    }

    if (mode == NpuAivFFT64Mode::RealInverse) {
      for (int64_t i = 0; i < group_n; ++i) {
        indices[output_index_base + i] = static_cast<uint32_t>(i * sizeof(float));
      }
    } else if (mode == NpuAivFFT64Mode::RealForward) {
      int64_t output_index = 0;
      for (int64_t row = 0; row < compute_group_size; ++row) {
        for (int64_t bin = 0; bin <= n / 2; ++bin) {
          indices[output_index_base + output_index++] =
              static_cast<uint32_t>((row * n + bin) * sizeof(float));
          indices[output_index_base + output_index++] =
              static_cast<uint32_t>((row * n + bin + group_n) * sizeof(float));
        }
      }
    } else {
      for (int64_t i = 0; i < 2 * group_n; ++i) {
        const int64_t component = i % 2;
        int64_t source_index;
        if (strided_columns) {
          const int64_t row = i / (2 * compute_group_size);
          const int64_t column = (i / 2) % compute_group_size;
          source_index = column * n + row + component * group_n;
        } else {
          const int64_t row_group = i / (2 * n);
          const int64_t point = (i / 2) % n;
          source_index = row_group * n + point + component * group_n;
        }
        indices[output_index_base + i] = static_cast<uint32_t>(source_index * sizeof(float));
      }
    }

    for (int64_t stage = 0; stage < stages; ++stage) {
      const int64_t length = int64_t{1} << (stage + 1);
      const int64_t half = length / 2;
      for (int64_t transform_group = 0; transform_group < compute_group_size; ++transform_group) {
        for (int64_t i = 0; i < n; ++i) {
          const int64_t butterfly_group = (i / length) * length;
          const int64_t offset = i % length;
          const bool upper = offset >= half;
          const int64_t a = butterfly_group + (upper ? offset - half : offset);
          const int64_t b = a + half;
          const int64_t twiddle_offset = offset % half;
          const double angle = sign * 2.0 * pi * static_cast<double>(twiddle_offset) /
                               static_cast<double>(length);
          const float negate = upper ? -1.0f : 1.0f;
          const int64_t destination = transform_group * n + i;
          const int64_t table_index = stage * group_n + destination;
          indices[stage_index_base + table_index] =
              static_cast<uint32_t>((transform_group * n + a) * sizeof(float));
          indices[stage_b_index_base + table_index] =
              static_cast<uint32_t>((transform_group * n + b) * sizeof(float));
          twiddles[table_index] = negate * static_cast<float>(std::cos(angle));
          twiddles[stages * group_n + table_index] = negate * static_cast<float>(std::sin(angle));
        }
      }
    }

    if (group_size == 8) {
      // The block loads an eight-transform tile, but computes two groups of
      // four sequentially to keep the FFT work/index/twiddle buffers within
      // the group-of-four UB footprint. Append the second half's input map.
      for (int64_t group = 0; group < compute_group_size; ++group) {
        for (int64_t i = 0; i < n; ++i) {
          int64_t value = i;
          int64_t reversed = 0;
          for (int64_t bit = 0; bit < stages; ++bit) {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
          }
          if (mode == NpuAivFFT64Mode::RealForward) {
            indices[second_input_base + group * n + i] =
                static_cast<uint32_t>(((group + compute_group_size) * n + reversed) * sizeof(float));
          } else if (mode == NpuAivFFT64Mode::RealInverse) {
            const int64_t compact_bin = reversed <= n / 2 ? reversed : n - reversed;
            const int64_t compact_complex_index =
                (group + compute_group_size) * (n / 2 + 1) + compact_bin;
            const int64_t destination = group * n + i;
            indices[second_input_base + destination] =
                static_cast<uint32_t>(2 * compact_complex_index * sizeof(float));
            indices[imag_input_base + group_n + destination] =
                static_cast<uint32_t>((2 * compact_complex_index + 1) * sizeof(float));
          } else {
            const int64_t input_complex_index = strided_columns
                ? reversed * group_size + group + compute_group_size
                : (group + compute_group_size) * n + reversed;
            indices[second_input_base + group * n + i] =
                static_cast<uint32_t>(input_complex_index * 2 * sizeof(float));
          }
        }
      }
    }
  }

#if defined(FLAGFFT_BACKEND_NPU)
  std::shared_ptr<CompiledRawNode> make_npu_aiv_fft64_child(const FFTRequest &request,
                                                            int64_t stride,
                                                            int32_t group_size,
                                                            NpuAivFFT64Mode mode = NpuAivFFT64Mode::Complex) {
    std::vector<uint32_t> host_indices;
    std::vector<float> host_twiddles;
    if (group_size == 1) {
      build_npu_aiv_fft64_tables(request, host_indices, host_twiddles, mode);
    } else {
      build_npu_aiv_fft64_group_tables(
          request, group_size, stride != 1, host_indices, host_twiddles, mode);
    }

    auto indices = std::make_shared<DeviceAllocation>(host_indices.size() * sizeof(uint32_t));
    indices->copy_from_host(host_indices.data(), host_indices.size() * sizeof(uint32_t));
    auto twiddles = std::make_shared<DeviceAllocation>(host_twiddles.size() * sizeof(float));
    twiddles->copy_from_host(host_twiddles.data(), host_twiddles.size() * sizeof(float));
    return std::make_shared<CompiledRawNpuAivFFT64Node>(stride, group_size, std::move(indices),
                                                       std::move(twiddles), mode);
  }

  int32_t npu_aiv_fft64_group_size(const char *setting, int64_t batch) {
    if (setting == nullptr) return 1;
    const std::string value(setting);
    if (value == "auto") return batch > 1 ? 8 : 4;
    if (value == "4") return 4;
    if (value == "8") return 8;
    return 1;
  }

  int32_t npu_aiv_fft64_real_row_group_size(const char *setting, int64_t batch) {
    const int32_t group_size = npu_aiv_fft64_group_size(setting, batch);
    // Compact rows contain 33 complex values. Grouping at least four rows
    // keeps AIV transfers aligned while preserving the compact layout.
    return group_size == 1 ? (batch > 1 ? 8 : 4) : group_size;
  }
#endif

  void mark_npu_portable_leaf(KernelKey &key, const FFTRequest &request) {
    const char *npu_3d_leaf = std::getenv("FLAGFFT_NPU_3D_LEAF");
    const bool allow_3d_leaf = request.origin_rank == 3 && npu_3d_leaf != nullptr &&
                               std::string(npu_3d_leaf) == "1";
    key.npu_portable_leaf = request.device_type == "npu" &&
                            (request.origin_rank == 2 || allow_3d_leaf);
  }

  bool use_ix_prime_real_bluestein(const PlanNodePtr &node,
                                   const FFTRequest &request,
                                   int64_t batch) {
    const auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node);
    if (!bluestein || request.device_type != "ix" || request.device_arch != "71" ||
        request.raw_dim != 1 || request.origin_rank > 1 || batch != 64 ||
        request.packed_real_child || request.fft_length != request.requested_n ||
        request.requested_n != bluestein->length ||
        (bluestein->length != 1009 && bluestein->length != 8191 && bluestein->length != 16381) ||
        request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
        (request.real_transform_kind != "r2c" && request.real_transform_kind != "c2r") ||
        request.input_strides.empty() || request.input_strides.back() != 1) {
      return false;
    }
    const char *setting = std::getenv("FLAGFFT_IX_PRIME_REAL_FUSION");
    if (setting && std::string(setting) != "0" && std::string(setting) != "1") {
      throw std::runtime_error("FLAGFFT_IX_PRIME_REAL_FUSION must be 0 or 1");
    }
    if (setting && std::string(setting) == "0") return false;
    if (std::dynamic_pointer_cast<LeafPlanNode>(bluestein->fft_plan)) return true;
    const auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(bluestein->fft_plan);
    if (!four_step) return false;
    const auto row = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    const auto col = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    return row && col && row->length < 512 && col->length < 512;
  }

  struct PackedRealChild {
    FFTRequest request;
    PlanNodePtr plan;
  };

  std::optional<PackedRealChild> select_packed_real_child(const PlanNodePtr &original_plan,
                                                          const FFTRequest &request,
                                                          int64_t batch,
                                                          bool inverse) {
    const char *setting = std::getenv("FLAGFFT_PACKED_REAL");
    const bool force = setting != nullptr && std::string(setting) == "1";
    const bool disable = setting != nullptr && std::string(setting) == "0";
    const int64_t n = request.requested_n;
    if (disable || (request.input_dtype != "complex128" && request.input_dtype != "complex64") || n <= 0 ||
        n % 2 != 0) {
      return std::nullopt;
    }
    const bool is_a100_fp64_target = request.device_type == "cuda" && request.device_arch == "sm_80";
    // Resource bounds remain target-local; batching itself is implemented
    // by the shared packed-real kernels with a distance-aware fallback.
    const bool is_musa_s5000_fp64_target = request.device_type == "musa" && request.device_arch == "31";
    // Ascend's single FP32 Stockham transforms avoid a full-length complex
    // expansion and halve the stage traffic. Keep small transforms and batch
    // layouts on their existing paths until they are qualified independently;
    // 2^20 is the upper end of the single-transform qualification range.
    const bool is_npu_single_fp32_target = request.device_type == "npu" &&
        request.input_dtype == "complex64" && batch == 1 && n >= 1024 && n <= 1048576;
    const bool is_ix_single_fp32_target = batch == 1 && ix_packed_real_policy_enabled(request);
    const bool is_ix_batch_fp32_target = batch == 64 && n == 16384 &&
        ix_ct_batch_policy_enabled(request);
    const char *ix_batch_setting = std::getenv("FLAGFFT_IX_CT_BATCH");
    if (request.device_type == "ix" && request.device_arch == "71" &&
        batch == 64 && (n == 185640 || n == 340200 || n == 524288 || n == 663000) &&
        ix_batch_setting != nullptr && std::string(ix_batch_setting) != "0" &&
        std::string(ix_batch_setting) != "1") {
      throw std::runtime_error("FLAGFFT_IX_CT_BATCH must be 0 or 1");
    }
    const bool is_ix_large_batch_fp32_target =
        request.device_type == "ix" && request.device_arch == "71" &&
        request.raw_dim == 1 && request.origin_rank <= 1 && batch == 64 &&
        request.fft_length == n && request.input_dtype == "complex64" &&
        request.output_dtype == "complex64" &&
        (n == 185640 || n == 340200 || n == 524288 || n == 663000) &&
        !request.input_strides.empty() && request.input_strides.back() == 1 &&
        (ix_batch_setting == nullptr || std::string(ix_batch_setting) != "0");
    const char *maca_batch_setting = std::getenv("FLAGFFT_MACA_1D_BATCH");
    const bool is_maca_batch_real_target =
        request.device_type == "maca" && request.device_arch == "102" &&
        request.raw_dim == 1 && request.origin_rank <= 1 && batch == 64 &&
        (n == 16384 || n == 185640 || n == 340200 || n == 524288 || n == 663000) &&
        (request.input_dtype == "complex64" || request.input_dtype == "complex128") &&
        (maca_batch_setting == nullptr || std::string(maca_batch_setting) != "0");
    // On C550, the compact-input four-step C2R pass still runs a full-length
    // complex FFT. The measured half-length route wins for these single
    // transforms in both precisions; keep short and batched paths unchanged.
    const bool is_maca_single_c2r_target =
        request.device_type == "maca" && request.raw_dim == 1 && batch == 1 && inverse &&
        std::dynamic_pointer_cast<FourStepPlanNode>(original_plan) != nullptr &&
        (n == 185640 || n == 340200 || n == 524288 || n == 663000);
    if (!force && !is_a100_fp64_target && !is_musa_s5000_fp64_target && !is_npu_single_fp32_target &&
        !is_ix_single_fp32_target && !is_ix_batch_fp32_target && !is_ix_large_batch_fp32_target &&
        !is_maca_batch_real_target && !is_maca_single_c2r_target) {
      return std::nullopt;
    }
    if (!force && !is_npu_single_fp32_target && !is_ix_single_fp32_target &&
        !is_ix_batch_fp32_target && !is_ix_large_batch_fp32_target &&
        !is_maca_batch_real_target && !is_maca_single_c2r_target &&
        (request.input_dtype != "complex128" || n < 65536 ||
         (batch == 1 && is_musa_s5000_fp64_target && n < 300000))) {
      return std::nullopt;
    }

    FFTRequest child_request = request;
    child_request.fft_length = n / 2;
    child_request.requested_n = n / 2;
    child_request.n = n / 2;
    child_request.output_dtype = child_request.input_dtype;
    child_request.direction = inverse ? "inverse" : "forward";
    child_request.norm = "backward";
    child_request.batch = batch;
    child_request.input_shape = {batch, n / 2};
    child_request.input_strides = {n / 2, 1};
    child_request.input_layout = "contiguous";
    child_request.packed_real_child = true;

    PlanBuilder child_builder;
    PlanNodePtr child_plan = child_builder.build(n / 2, child_request);
    if (!force && is_npu_single_fp32_target) {
      // Do not implicitly enable unmeasured Bluestein children or GPU tuning
      // cache entries. New radices qualify naturally when the NPU planner
      // supplies a Stockham plan for both the original and half length.
      if (!std::dynamic_pointer_cast<StockhamPlanNode>(original_plan) ||
          !std::dynamic_pointer_cast<StockhamPlanNode>(child_plan)) {
        return std::nullopt;
      }
      return PackedRealChild {std::move(child_request), std::move(child_plan)};
    }
    // A measured complex plan is also valid for this dense half-length
    // child. Keep the same exact-batch, direction and fingerprint lookup.
    if (auto tuned = lookup_tuned_plan_json(child_request)) {
      try {
        auto candidate = plan_node_from_json(child_builder, tuned->at("root"));
        auto pair = std::dynamic_pointer_cast<FourStepPlanNode>(candidate);
        if (pair && pair->n1 * pair->n2 == n / 2 && std::dynamic_pointer_cast<LeafPlanNode>(pair->row_plan) &&
            std::dynamic_pointer_cast<LeafPlanNode>(pair->col_plan)) {
          child_plan = std::move(candidate);
        }
      } catch (const std::exception &) {
        // An obsolete or malformed cache must not prevent plan creation.
      }
    }
    auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(child_plan);
    const bool child_is_leaf_pair = four_step != nullptr &&
                                    std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan) != nullptr &&
                                    std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan) != nullptr;
    if (!force && (is_ix_single_fp32_target || is_ix_batch_fp32_target ||
                   is_ix_large_batch_fp32_target)) {
      if (!child_is_leaf_pair) return std::nullopt;
      return PackedRealChild {std::move(child_request), std::move(child_plan)};
    }
    if (!force && is_maca_single_c2r_target) {
      // A tuned child may choose a different factor pair. Use the route only
      // for the exact half-length decompositions measured on C550.
      const bool measured_child = child_is_leaf_pair &&
          ((n == 185640 && four_step->n1 == 260 && four_step->n2 == 357) ||
           (n == 340200 && four_step->n1 == 243 && four_step->n2 == 700) ||
           (n == 524288 && four_step->n1 == 512 && four_step->n2 == 512) ||
           (n == 663000 && four_step->n1 == 300 && four_step->n2 == 1105));
      if (!measured_child) return std::nullopt;
      return PackedRealChild {std::move(child_request), std::move(child_plan)};
    }
    // A half-length transform wins only while both generated leaf kernels stay
    // below the high-register large-leaf regime.  The MUSA S5000 threshold is
    // wider than A100's based on the validated grid, but remains target-local.
    const int64_t leaf_limit = (batch == 1 && is_musa_s5000_fp64_target) ? 1088 : 768;
    const bool bounded_leaf_pair =
        child_is_leaf_pair && four_step->n1 <= leaf_limit && four_step->n2 <= leaf_limit;
    auto original_four_step = std::dynamic_pointer_cast<FourStepPlanNode>(original_plan);
    const bool original_has_large_leaf =
        inverse && original_four_step != nullptr &&
        std::dynamic_pointer_cast<LeafPlanNode>(original_four_step->row_plan) != nullptr &&
        std::dynamic_pointer_cast<LeafPlanNode>(original_four_step->col_plan) != nullptr &&
        (original_four_step->n1 > 768 || original_four_step->n2 > 768);
    // Compact C2R input handling is especially expensive in a large generated
    // leaf.  Its half-length complex route remains profitable a little beyond
    // the general FP64 leaf bound, provided it replaces such a large leaf.
    const bool c2r_large_leaf_relief =
        original_has_large_leaf && child_is_leaf_pair && four_step->n1 <= 1536 && four_step->n2 <= 1536;
    if (!force && !is_maca_batch_real_target && !bounded_leaf_pair && !c2r_large_leaf_relief) {
      return std::nullopt;
    }
    return PackedRealChild {std::move(child_request), std::move(child_plan)};
  }

  // Whether 3D should fuse its axis permutations into the FFT stores.  The
  // trade depends on the backend.  On IX, the fused path is much slower on
  // large cubes than the standalone tiled transposes, so keep it off by
  // default.  FLAGFFT_3D_FUSED_STORE=0/1 overrides this for screening.
  bool fused_3d_store_enabled() {
    const char *override_value = std::getenv("FLAGFFT_3D_FUSED_STORE");
    if (override_value != nullptr && *override_value != '\0') {
      return std::string(override_value) != "0";
    }
    return adaptor::backend_name() != "cuda" && adaptor::backend_name() != "ix";
  }

  bool maca_flag_or_default(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
      return default_value;
    }
    return std::string(value) == "1";
  }

  bool real_direct_dft_enabled(const PlanNodePtr &node,
                               const FFTRequest &request,
                               int64_t batch) {
    // Keep real DirectDFT at the public rank-1 boundary. Batch 64 at N=23
    // avoids two conversion launches, with a separate opt-out for this path.
    auto direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node);
    const bool single_target = request.batch == 1 && batch == 1 &&
                               maca_flag_or_default("FLAGFFT_MACA_REAL_DIRECT_DFT",
                                                    maca_tail_real_direct_dft(request));
    const char *batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
    const bool batch_target = request.device_arch == "102" && request.origin_rank <= 1 &&
                              request.batch == 64 && batch == 64 && request.requested_n == 23 &&
                              (batch_policy == nullptr || std::string(batch_policy) != "0") &&
                              maca_flag_or_default("FLAGFFT_MACA_BATCH_REAL_DIRECT_DFT", true);
    const char *ix_setting = std::getenv("FLAGFFT_IX_REAL_DIRECT_DFT");
    const bool ix_batch_target = request.device_type == "ix" && request.device_arch == "71" &&
                                 request.origin_rank <= 1 && request.batch == 64 && batch == 64 &&
                                 request.requested_n == 23 && request.input_dtype == "complex64" &&
                                 request.output_dtype == "complex64" &&
                                 request.input_strides.size() == 2 &&
                                 request.input_strides.back() == 1 &&
                                 request.input_strides.front() ==
                                     (request.real_transform_kind == "c2r" ? 12 : 23) &&
                                 (ix_setting == nullptr || std::string(ix_setting) != "0");
    return request.raw_dim == 1 &&
           ((request.device_type == "maca" && (single_target || batch_target)) || ix_batch_target) &&
           direct != nullptr &&
           direct->length == request.requested_n && direct->length > 0 &&
           direct->length <= kDirectDftMaxN &&
           (request.input_dtype == "complex64" || request.input_dtype == "complex128");
  }

  // Row/column requests replace batch with the number of axis transforms.
  // Carry the root 2D policy through that recursion, then restore it so a
  // compiler reused for an unrelated plan cannot leak the defaults.
  struct Maca2dPolicyScope {
    bool &state;
    bool previous;
    Maca2dPolicyScope(bool &state, bool enabled) : state(state), previous(state) {
      state = enabled;
    }
    Maca2dPolicyScope(bool &state, const FFTRequest &request, int64_t batch, int64_t n0, int64_t n1)
        : Maca2dPolicyScope(state,
              request.device_type == "maca" && request.raw_dim == 2 && request.batch == 1 &&
              request.input_dtype == "complex64" && request.output_dtype == "complex64" && batch == 1 &&
              request.input_layout == "contiguous" && !request.requires_contiguous_copy && n0 > 1 && n1 > 1 &&
              maca_flag_or_default("FLAGFFT_MACA_2D_SINGLE", true)) {
    }
    ~Maca2dPolicyScope() { state = previous; }
    Maca2dPolicyScope(const Maca2dPolicyScope &) = delete;
    Maca2dPolicyScope &operator=(const Maca2dPolicyScope &) = delete;
  };

  bool has_real_boundary_row_plan(const PlanNodePtr &node) {
    if (std::dynamic_pointer_cast<LeafPlanNode>(node) != nullptr) {
      return true;
    }
    auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node);
    if (four_step == nullptr) {
      return false;
    }
    return std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan) != nullptr &&
           std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan) != nullptr;
  }

}  // namespace

void TritonCompiler::configure_single_transform_policies(const FFTRequest &request) {
  ix_ct_single_policy_ = ix_ct_single_policy_enabled(request);
  ix_ct_batch_policy_ = ix_ct_batch_policy_enabled(request);
  ix_real_single_pack_ = request.device_type == "ix" && request.device_arch == "71" &&
                         request.raw_dim == 1 && request.origin_rank <= 1 &&
                         request.batch == 1 && request.requested_n == 210 &&
                         request.fft_length == 210 && request.input_dtype == "complex64" &&
                         request.output_dtype == "complex64" &&
                         (request.real_transform_kind == "r2c" ||
                          request.real_transform_kind == "c2r") &&
                         !request.input_strides.empty() && request.input_strides.back() == 1;
  ix_ct_single_tle_policy_ = ix_ct_single_tle_policy(request);
  maca_tail_policy_ = maca_tail_codegen_root(request);
  maca_1d_batch_policy_ = request.device_type == "maca" && request.device_arch == "102" &&
                          request.raw_dim == 1 && request.origin_rank <= 1 && request.batch == 64 &&
                          request.fft_length == request.requested_n &&
                          (request.requested_n == 16 || request.requested_n == 1024 ||
                           request.requested_n == 2048 ||
                           ((request.requested_n == 16384 || request.requested_n == 524288 ||
                             (request.packed_real_child && request.requested_n == 262144)) &&
                            request.input_dtype == "complex64") ||
                           (request.requested_n == 524287 &&
                            (request.input_dtype == "complex64" ||
                             request.input_dtype == "complex128")) ||
                           ((request.requested_n == 524288 ||
                             (request.packed_real_child && request.requested_n == 262144)) &&
                            request.input_dtype == "complex128")) &&
                          !request.input_strides.empty() && request.input_strides.back() == 1;
  if (maca_1d_batch_policy_) {
    const char *batch_override = std::getenv("FLAGFFT_MACA_1D_BATCH");
    if (batch_override != nullptr && *batch_override != '\0') {
      if (std::string(batch_override) == "0") {
        maca_1d_batch_policy_ = false;
      } else if (std::string(batch_override) != "1") {
        throw std::runtime_error("FLAGFFT_MACA_1D_BATCH must be 0 or 1");
      }
    }
  }
  maca_1d_single_policy_ = request.device_type == "maca" && request.raw_dim == 1 && request.batch == 1;
  if (!maca_1d_single_policy_) {
    return;
  }
  const char *override_value = std::getenv("FLAGFFT_MACA_1D_SINGLE");
  if (override_value == nullptr || *override_value == '\0') {
    return;
  }
  if (std::string(override_value) == "0") {
    maca_1d_single_policy_ = false;
  } else if (std::string(override_value) != "1") {
    throw std::runtime_error("FLAGFFT_MACA_1D_SINGLE must be 0 or 1");
  }
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_node(const PlanNodePtr &node,
                                                                  const FFTRequest &request,
                                                                  int64_t batch) {
  configure_single_transform_policies(request);
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
#if defined(FLAGFFT_BACKEND_NPU)
    const char *npu_3d_aiv64 = std::getenv("FLAGFFT_NPU_3D_AIV64");
    const bool use_npu_3d_aiv64 = request.device_type == "npu" && request.origin_rank == 3 &&
                                  request.raw_dim == 1 && request.input_dtype == "complex64" &&
                                  request.output_dtype == "complex64" && leaf->length == 64 &&
                                  npu_3d_aiv64 != nullptr && std::string(npu_3d_aiv64) == "1";
    if (use_npu_3d_aiv64) {
      const char *group_setting = std::getenv("FLAGFFT_NPU_3D_AIV64_GROUP");
      if (group_setting == nullptr) group_setting = "auto";
      int32_t group_size = npu_aiv_fft64_group_size(group_setting, batch);
      if (group_size > 1 && batch % group_size != 0) group_size = 1;
      return make_npu_aiv_fft64_child(request, 1, group_size);
    }
#endif
    return compile_raw_leaf(*leaf, request);
  }
  if (auto direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node)) {
    return compile_raw_direct_dft(*direct, request, batch);
  }
  if (auto stockham = std::dynamic_pointer_cast<StockhamPlanNode>(node)) {
    std::vector<std::shared_ptr<JitKernel>> kernels;
    int64_t stage_span = 1;
    // Small single transforms use smaller tiles to expose more NPU programs;
    // batch-64 1024/2048 transforms use the best measured 32/64 tile sizes.
    int64_t butterfly_block = 128;
    if (stockham->length == 1024 || stockham->length == 2048) {
      if (batch == 1) {
        butterfly_block = 8;
      } else if (batch == 64) {
        butterfly_block = stockham->length == 1024 ? 32 : 64;
      }
    }
    auto block_override = [](const char *name, int64_t fallback) {
      const char *raw = std::getenv(name);
      if (raw == nullptr) return fallback;
      const std::string value(raw);
      if (value != "8" && value != "16" && value != "32" && value != "64" && value != "128") {
        throw std::runtime_error(std::string(name) + " must be 8, 16, 32, 64 or 128");
      }
      return static_cast<int64_t>(std::stoll(value));
    };
    if (stockham->length <= 2048) {
      butterfly_block = block_override("FLAGFFT_NPU_STOCKHAM_BLOCK", butterfly_block);
    }
    const int64_t prime_block = block_override("FLAGFFT_NPU_PRIME_BLOCK", 128);
    for (int64_t radix : stockham->factors) {
      KernelKey key = KernelKey::direct_dft(triton_target_for_request(request),
                                            request.direction,
                                            request.input_dtype,
                                            stockham->length);
      key.kind = KernelKind::StockhamStage;
      // Stockham kernel identity includes the stage span; the plan factors
      // remain the radix sequence. This removes dynamic integer division and
      // lets the first stage omit all unit twiddle loads and multiplies.
      const bool vector_prime = radix == 13 || radix == 17 || radix == 19;
      key.factors = {radix, stage_span, vector_prime ? prime_block : butterfly_block};
      kernels.push_back(compile_kernel(key));
      stage_span *= radix;
    }
    const int64_t n = stockham->length;
    std::vector<double> values(static_cast<std::size_t>(2 * n));
    for (int64_t i = 0; i < n; ++i) {
      double angle = (request.direction == "inverse" ? 2.0 : -2.0) * kPi * i / n;
      values[2 * i] = std::cos(angle);
      values[2 * i + 1] = std::sin(angle);
    }
    DeviceAllocation twiddle =
        request.input_dtype == "complex128"
            ? adaptor::Memory::from_doubles(values)
            : adaptor::Memory::from_floats(std::vector<float>(values.begin(), values.end()));
    const auto bytes = static_cast<std::size_t>(batch * n * complex_element_bytes(request.input_dtype));
    return std::make_shared<CompiledRawStockhamNode>(n,
                                                     stockham->factors,
                                                     std::move(kernels),
                                                     std::move(twiddle),
                                                     adaptor::Memory(bytes),
                                                     adaptor::Memory(bytes));
  }
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    if (row_leaf != nullptr && col_leaf != nullptr) {
      DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
      return std::make_shared<CompiledRawFourStepFusedNode>(
          four_step->length,
          four_step->n1,
          four_step->n2,
          compile_four_step_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*row_leaf, request),
          compile_four_step_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*col_leaf, request),
          std::move(twiddle),
          std::move(stage1));
    }
    return compile_raw_four_step_generic(*four_step, request, batch);
  }
  if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
    const std::string real_kind = use_ix_prime_real_bluestein(node, request, batch)
                                      ? request.real_transform_kind : "";
    auto make_real_layout_fallback = [node, request, batch]() -> std::shared_ptr<CompiledRawNode> {
      FFTRequest fallback_request = request;
      fallback_request.real_transform_kind.clear();
      TritonCompiler compiler;
      return request.real_transform_kind == "r2c"
                 ? compiler.compile_raw_r2c_node(node, fallback_request, batch, false)
                 : compiler.compile_raw_c2r_node(node, fallback_request, batch, false);
    };
    FFTRequest child_request = forward_child_request(request);
    // The generic Bluestein pipeline uses per-batch convolution buffers
    // (a_buf/work_buf plus the child FFT workspace).  For large primes and
    // large batch these can exceed device memory, so the batch is compiled at
    // chunk granularity and executed as a sequence of chunks.  The chunk is
    // sized by a byte budget instead of a fixed count: small convolutions
    // (e.g. 997 -> conv 2048) run in one launch, while 2^20 convolutions keep
    // the previous 32-transform chunks.
    const int64_t bluestein_element_bytes = complex_element_bytes(request.input_dtype);
    const int64_t conv_bytes = bluestein->conv_length * bluestein_element_bytes;
    constexpr int64_t kBluesteinChunkByteBudget = 256 * 1024 * 1024;
    // The prepare/pointwise/finalize kernels launch ceil(conv_length/256)
    // column blocks times the chunk in grid.y, so the chunk also has to respect
    // the backend's per-launch block limit (for example 65535 on Ascend NPU).
    const int64_t bluestein_columns = (bluestein->conv_length + 255) / 256;
    const int64_t bluestein_grid_chunk =
        std::max<int64_t>(1, adaptor::max_launch_blocks() / std::max<int64_t>(1, bluestein_columns));
    const int64_t chunk_batch =
        std::min<int64_t>({batch,
                           std::max<int64_t>(1, kBluesteinChunkByteBudget / std::max<int64_t>(1, conv_bytes)),
                           bluestein_grid_chunk});
    auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(bluestein->fft_plan);
    auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(bluestein->fft_plan);
    auto row_leaf = four_step ? std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan) : nullptr;
    auto col_leaf = four_step ? std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan) : nullptr;
    std::shared_ptr<CompiledRawNode> fft = compile_raw_node(bluestein->fft_plan, child_request, chunk_batch);
    DeviceAllocation chirp =
        build_raw_bluestein_chirp(request, bluestein->length, request.direction == "inverse");
    DeviceAllocation b_time = build_raw_bluestein_b(request, bluestein->length, bluestein->conv_length);
    const bool use_a100_fp64_full_leaf = request.device_type == "cuda" && request.device_arch == "sm_80" &&
                                         request.input_dtype == "complex128" && batch == 1;
    const bool use_musa_s5000_fp64_full_leaf = request.device_type == "musa" && request.device_arch == "31" &&
                                               request.input_dtype == "complex128" && batch == 1;
    const bool use_musa_3d_fp64_full_leaf = request.device_type == "musa" && request.device_arch == "31" &&
                                           request.input_dtype == "complex128" && request.origin_rank == 3 &&
                                           bluestein->length == 997 && batch <= 4096;
    // MACA's portable register exchange is compiled separately for each FFT.
    // Combining both FFTs makes this plugin's optimization prohibitively slow.
    const bool allow_bluestein_fusion = request.device_type != "maca";
    // Keep each MACA FFT in its own kernel while folding the surrounding
    // elementwise work into its loads/stores. The measured boundary path is
    // the default for MACA 1D single transforms; the named environment
    // variable remains an explicit A/B override.
    const bool use_maca_boundary_leaf =
        request.device_type == "maca" && batch == 1 && leaf != nullptr &&
        maca_flag_or_default("FLAGFFT_MACA_BLUESTEIN_LEAF_FUSION", maca_1d_single_policy_);
    const char *ix_3d_boundary = std::getenv("FLAGFFT_IX_3D_BLUESTEIN_BOUNDARY");
    const bool use_ix_3d_boundary_leaf =
        request.device_type == "ix" && request.device_arch == "71" &&
        request.origin_rank == 3 && request.input_dtype == "complex64" &&
        bluestein->length == 997 && leaf != nullptr && ix_3d_boundary != nullptr &&
        std::string(ix_3d_boundary) == "1";
    const bool use_full_leaf =
        allow_bluestein_fusion &&
        (request.input_dtype == "complex64" || use_a100_fp64_full_leaf || use_musa_s5000_fp64_full_leaf ||
         use_musa_3d_fp64_full_leaf) &&
        leaf != nullptr;
    // Batched S5000 FP64 convolutions can fuse the boundary when both
    // leaves fit the bounds below and the complete batch fits the existing
    // workspace budget. This is independent of the original prime length.
    const bool use_musa_fp64_four_step = request.device_type == "musa" && request.device_arch == "31" &&
                                         request.input_dtype == "complex128" && batch >= 16 &&
                                         batch == chunk_batch;
    const bool use_default_four_step = allow_bluestein_fusion &&
                               (request.input_dtype == "complex64" || use_musa_fp64_four_step) &&
                               four_step != nullptr && row_leaf != nullptr && col_leaf != nullptr &&
                               row_leaf->length < 512 && col_leaf->length < 512;
    // Separate from the two-kernel leaf experiment: large convolutions retain
    // one FFT per boundary kernel and execute four kernels in total. Bound
    // each child to the existing 1024-point four-step leaf family.
    const bool use_maca_four_step =
        request.device_type == "maca" && batch == 1 && four_step != nullptr &&
        row_leaf != nullptr && col_leaf != nullptr &&
        row_leaf->length <= 1024 && col_leaf->length <= 1024 &&
        maca_flag_or_default("FLAGFFT_MACA_BLUESTEIN_FOUR_STEP_FUSION", maca_1d_single_policy_);
    const bool use_four_step = use_default_four_step || use_maca_four_step;
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    DeviceAllocation b_fft_buf =
        adaptor::Memory(static_cast<std::size_t>(bluestein->conv_length * element_bytes));
    if (use_maca_boundary_leaf || use_ix_3d_boundary_leaf) {
      DeviceAllocation work_buf =
          adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
      return std::make_shared<CompiledRawBluesteinLeafNode>(
          bluestein->length,
          bluestein->conv_length,
          std::move(fft),
          compile_leaf_bluestein_prepare_kernel(*leaf, child_request, bluestein->length),
          compile_leaf_bluestein_finish_kernel(*leaf, child_request, bluestein->length),
          build_raw_leaf_tables(*leaf, child_request),
          std::move(chirp),
          std::move(b_time),
          std::move(work_buf),
          std::move(b_fft_buf));
    }
    if (use_full_leaf) {
      std::vector<int64_t> fused_factors = leaf->factors;
      if (fused_factors.size() >= 3) {
        std::reverse(fused_factors.begin() + 1, fused_factors.end());
      }
      int64_t fused_warps = leaf->num_warps;
      if (request.device_type == "ix" && request.device_arch == "71" &&
          request.origin_rank == 3 && bluestein->length == 997) {
        const char *value = std::getenv("FLAGFFT_IX_3D_BLUESTEIN_WARPS");
        if (value != nullptr && (std::string(value) == "4" || std::string(value) == "8")) {
          fused_warps = std::strtoll(value, nullptr, 10);
        }
      }
      LeafPlanNode fused_leaf(leaf->length,
                              std::move(fused_factors),
                              leaf->remainder,
                              leaf->lanes,
                              fused_warps,
                              leaf->generic_radices,
                              leaf->smem_size);
      KernelKey fused_key = KernelKey::leaf_bluestein(triton_target_for_request(child_request),
                                                       child_request.direction,
                                                       child_request.input_dtype,
                                                       bluestein->length,
                                                       fused_leaf.length,
                                                       fused_leaf.factors,
                                                       fused_leaf.lanes,
                                                       fused_leaf.num_warps,
                                                       fused_leaf.generic_radices,
                                                       fused_leaf.smem_size);
      if (!real_kind.empty()) fused_key.perm_form = real_kind;
      return std::make_shared<CompiledRawBluesteinFullLeafNode>(
          bluestein->length,
          bluestein->conv_length,
          std::move(fft),
          compile_kernel(fused_key),
          build_raw_leaf_tables(fused_leaf, child_request),
          std::move(chirp),
          std::move(b_time),
          std::move(b_fft_buf),
          real_kind,
          real_kind.empty() ? std::function<std::shared_ptr<CompiledRawNode>()>{}
                            : make_real_layout_fallback);
    }
    if (use_four_step) {
      auto make_boundary_leaf = [use_maca_four_step](const LeafPlanNode &source) {
        std::vector<int64_t> factors = source.factors;
        if (!use_maca_four_step && factors.size() == 2 && factors.front() > factors.back()) {
          std::reverse(factors.begin(), factors.end());
        }
        return LeafPlanNode(source.length,
                            std::move(factors),
                            source.remainder,
                            source.lanes,
                            source.num_warps,
                            source.generic_radices,
                            source.smem_size);
      };
      LeafPlanNode boundary_row = make_boundary_leaf(*row_leaf);
      LeafPlanNode boundary_col = make_boundary_leaf(*col_leaf);
      auto compile_boundary_kernel = [&](const LeafPlanNode &boundary_leaf, KernelKind kind, bool is_row) {
        KernelKey key = is_row ? KernelKey::four_step_row(triton_target_for_request(child_request),
                                                          child_request.direction,
                                                          child_request.input_dtype,
                                                          four_step->n1,
                                                          four_step->n2,
                                                          boundary_leaf.length,
                                                          boundary_leaf.factors,
                                                          boundary_leaf.lanes,
                                                          boundary_leaf.num_warps,
                                                          boundary_leaf.generic_radices,
                                                          boundary_leaf.smem_size)
                               : KernelKey::four_step_col(triton_target_for_request(child_request),
                                                          child_request.direction,
                                                          child_request.input_dtype,
                                                          four_step->n1,
                                                          four_step->n2,
                                                          boundary_leaf.length,
                                                          boundary_leaf.factors,
                                                          boundary_leaf.lanes,
                                                          boundary_leaf.num_warps,
                                                          boundary_leaf.generic_radices,
                                                          boundary_leaf.smem_size);
        key.kind = kind;
        key.bluestein_n = bluestein->length;
        key.bluestein_m = bluestein->conv_length;
        if (!real_kind.empty() &&
            (kind == KernelKind::BluesteinFourStepPrepareRow ||
             kind == KernelKind::BluesteinFourStepFinishCol)) {
          key.perm_form = real_kind;
        }
        return compile_kernel(key);
      };

      DeviceAllocation twiddle = build_raw_four_step_twiddle(child_request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
      DeviceAllocation work_buf =
          adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
      return std::make_shared<CompiledRawBluesteinFourStepNode>(
          bluestein->length,
          bluestein->conv_length,
          four_step->n1,
          four_step->n2,
          std::move(fft),
          compile_boundary_kernel(boundary_row, KernelKind::BluesteinFourStepPrepareRow, true),
          compile_four_step_col_kernel(boundary_col, child_request, four_step->n1, four_step->n2),
          compile_boundary_kernel(boundary_row, KernelKind::BluesteinFourStepPointwiseRow, true),
          compile_boundary_kernel(boundary_col, KernelKind::BluesteinFourStepFinishCol, false),
          build_raw_leaf_tables(boundary_row, child_request),
          build_raw_leaf_tables(boundary_col, child_request),
          std::move(twiddle),
          std::move(chirp),
          std::move(b_time),
          std::move(stage1),
          std::move(work_buf),
          std::move(b_fft_buf),
          real_kind,
          real_kind.empty() ? std::function<std::shared_ptr<CompiledRawNode>()>{}
                            : make_real_layout_fallback);
    }
    DeviceAllocation work_buf =
        adaptor::Memory(static_cast<std::size_t>(chunk_batch * bluestein->conv_length * element_bytes));
    DeviceAllocation a_buf =
        adaptor::Memory(static_cast<std::size_t>(chunk_batch * bluestein->conv_length * element_bytes));
    return std::make_shared<CompiledRawBluesteinNode>(
        bluestein->length,
        bluestein->conv_length,
        std::move(fft),
        compile_bluestein_prepare_kernel(request, bluestein->length, bluestein->conv_length),
        compile_bluestein_pointwise_kernel(request, bluestein->length, bluestein->conv_length),
        compile_bluestein_finalize_kernel(request, bluestein->length, bluestein->conv_length),
        std::move(chirp),
        std::move(b_time),
        std::move(a_buf),
        std::move(work_buf),
        std::move(b_fft_buf),
        chunk_batch);
  }
  if (auto rader = std::dynamic_pointer_cast<RaderPlanNode>(node)) {
    FFTRequest child_request = forward_child_request(request);
    std::shared_ptr<CompiledRawNode> fft = compile_raw_node(rader->conv_plan, child_request, batch);
    std::shared_ptr<JitKernel> fused_leaf_kernel;
    std::vector<DeviceAllocation> fused_leaf_tables;
    std::shared_ptr<JitKernel> boundary_prepare_kernel;
    std::shared_ptr<JitKernel> boundary_finish_kernel;
    std::vector<DeviceAllocation> boundary_tables;
    const auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(rader->conv_plan);
    const bool maca_batch1009_leaf = request.device_type == "maca" &&
        request.device_arch == "102" && request.raw_dim == 1 &&
        request.origin_rank <= 1 && batch == 64 && rader->prime == 1009 &&
        request.input_dtype == "complex64" && leaf != nullptr;
    const char* fusion = std::getenv("FLAGFFT_MACA_RADER_FULL_LEAF");
    if (fusion && std::string(fusion) == "1" && maca_batch1009_leaf) {
      std::vector<int64_t> fused_factors = leaf->factors;
      if (fused_factors.size() >= 3) {
        std::reverse(fused_factors.begin() + 1, fused_factors.end());
      }
      LeafPlanNode fused_leaf(leaf->length, std::move(fused_factors), leaf->remainder,
                              leaf->lanes, leaf->num_warps, leaf->generic_radices,
                              leaf->smem_size);
      fused_leaf_kernel = compile_leaf_rader_full_kernel(fused_leaf, child_request, rader->prime);
      fused_leaf_tables = build_raw_leaf_tables(fused_leaf, child_request);
    }
    const char* boundary = std::getenv("FLAGFFT_MACA_RADER_BOUNDARY_LEAF");
    const bool use_boundary = boundary ? std::string(boundary) == "1"
                                       : request.real_transform_kind == "c2r";
    if (use_boundary && maca_batch1009_leaf && !fused_leaf_kernel) {
      boundary_prepare_kernel = compile_leaf_rader_prepare_kernel(*leaf, child_request, rader->prime);
      boundary_finish_kernel = compile_leaf_rader_finish_kernel(*leaf, child_request, rader->prime);
      boundary_tables = build_raw_leaf_tables(*leaf, child_request);
    }
    DeviceAllocation idx = build_raw_rader_idx_table(rader->idx);
    DeviceAllocation b_time = build_raw_rader_conv_kernel(request, rader->prime, rader->idx);
    const int64_t conv_length = rader->prime - 1;
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    DeviceAllocation a_buf = adaptor::Memory(static_cast<std::size_t>(batch * conv_length * element_bytes));
    DeviceAllocation work_buf =
        adaptor::Memory(static_cast<std::size_t>(batch * conv_length * element_bytes));
    DeviceAllocation b_fft_buf = adaptor::Memory(static_cast<std::size_t>(conv_length * element_bytes));
    DeviceAllocation input_copy =
        adaptor::Memory(static_cast<std::size_t>(batch * rader->prime * element_bytes));
    return std::make_shared<CompiledRawRaderNode>(
        rader->prime,
        conv_length,
        std::move(fft),
        compile_rader_prepare_kernel(request, rader->prime, conv_length),
        compile_rader_pointwise_kernel(request, rader->prime, conv_length),
        compile_rader_finalize_kernel(request, rader->prime, conv_length),
        std::move(idx),
        std::move(b_time),
        std::move(a_buf),
        std::move(work_buf),
        std::move(b_fft_buf),
        std::move(input_copy),
        std::move(fused_leaf_kernel),
        std::move(fused_leaf_tables),
        std::move(boundary_prepare_kernel),
        std::move(boundary_finish_kernel),
        std::move(boundary_tables));
  }
  if (auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(node)) {
    return compile_raw_2d_node(two_dim, request, batch);
  }
  if (auto three_dim = std::dynamic_pointer_cast<ThreeDimPlanNode>(node)) {
    return compile_raw_3d_node(three_dim, request, batch);
  }
  throw std::runtime_error("raw C API does not support plan node kind: " + plan_node_kind_name(node->kind));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_r2c_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch,
                                                                      bool allow_packed) {
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = request.requested_n;
  if (real_direct_dft_enabled(node, request, batch)) {
    return compile_raw_real_direct_dft(request, false);
  }
  if (use_ix_prime_real_bluestein(node, request, batch)) {
    return compile_raw_node(node, request, batch);
  }
  // Pack even/odd input samples, run a half-length complex leaf, and
  // reconstruct the compact spectrum in one launch on the qualified IX case.
  // Keep an opt-out for driver/toolchain regressions.
  const char *fused_setting = std::getenv("FLAGFFT_IX_FUSED_R2C");
  if (fused_setting != nullptr && std::string(fused_setting) != "0" &&
      std::string(fused_setting) != "1") {
    throw std::runtime_error("FLAGFFT_IX_FUSED_R2C must be 0 or 1");
  }
  if (request.device_type == "ix" && request.device_arch == "71" &&
      request.raw_dim == 1 && request.origin_rank <= 1 &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      n == 2048 &&
      (batch == 1 || batch == 64) &&
      (batch == 1 ? ix_ct_single_policy_ : ix_ct_batch_policy_) &&
      !request.input_strides.empty() && request.input_strides.back() == 1 &&
      (std::getenv("FLAGFFT_IX_PORTABLE_LEAF") == nullptr ||
       std::string(std::getenv("FLAGFFT_IX_PORTABLE_LEAF")) == "1") &&
      (fused_setting == nullptr || std::string(fused_setting) == "1")) {
    FFTRequest child_request = request;
    child_request.n = n / 2;
    child_request.requested_n = n / 2;
    child_request.fft_length = n / 2;
    child_request.output_dtype = child_request.input_dtype;
    child_request.real_transform_kind.clear();
    child_request.input_shape = {batch, n / 2};
    child_request.input_strides = {n / 2, 1};
    PlanBuilder child_builder;
    child_builder.build(n / 2, child_request);
    const std::vector<int64_t> factors {16, 8, 8};
    const int64_t lanes = child_builder.choose_lanes(n / 2, factors);
    LeafPlanNode packed_leaf(n / 2, factors, 1, lanes,
                             child_builder.choose_num_warps(lanes), {}, n / 2);
    KernelKey key = KernelKey::leaf_r2c(triton_target_for_request(request),
                                         request.direction, request.input_dtype,
                                         packed_leaf.length, packed_leaf.factors,
                                         packed_leaf.lanes, packed_leaf.num_warps,
                                         packed_leaf.generic_radices, packed_leaf.smem_size);
    key.kind = KernelKind::LeafPackedR2C;
    return std::make_shared<CompiledRawR2CLeafNode>(
        n, compile_kernel(key), build_raw_leaf_tables(packed_leaf, child_request),
        build_raw_packed_real_twiddle(request, n));
  }
  // The 3D real boundary has many contiguous rows.  Screen the existing
  // one-launch even/odd leaf there to halve its FFT length without adding a
  // packing pass; n=64 and n=256 cover the current large FP32 3D matrix.
  const char *ix_3d_packed_override = std::getenv("FLAGFFT_IX_3D_PACKED_R2C");
  if (request.device_type == "ix" && request.device_arch == "71" &&
      request.origin_rank == 3 && request.input_dtype == "complex64" &&
      request.output_dtype == "complex64" && (n == 64 || n == 256) &&
      batch >= 1024 && ix_3d_packed_override != nullptr &&
      std::string(ix_3d_packed_override) == "1") {
    FFTRequest child_request = request;
    child_request.n = n / 2;
    child_request.requested_n = n / 2;
    child_request.fft_length = n / 2;
    child_request.real_transform_kind.clear();
    child_request.input_shape = {batch, n / 2};
    child_request.input_strides = {n / 2, 1};
    PlanBuilder child_builder;
    child_builder.build(n / 2, child_request);
    const std::vector<int64_t> factors = n == 64
        ? std::vector<int64_t>{4, 8} : std::vector<int64_t>{8, 16};
    const int64_t lanes = child_builder.choose_lanes(n / 2, factors);
    LeafPlanNode packed_leaf(n / 2, factors, 1, lanes,
                             child_builder.choose_num_warps(lanes), {}, n / 2);
    KernelKey key = KernelKey::leaf_r2c(triton_target_for_request(request),
                                         request.direction, request.input_dtype,
                                         packed_leaf.length, packed_leaf.factors,
                                         packed_leaf.lanes, packed_leaf.num_warps,
                                         packed_leaf.generic_radices, packed_leaf.smem_size);
    key.kind = KernelKind::LeafPackedR2C;
    return std::make_shared<CompiledRawR2CLeafNode>(
        n, compile_kernel(key), build_raw_leaf_tables(packed_leaf, child_request),
        build_raw_packed_real_twiddle(request, n));
  }
  if (auto packed_child =
          allow_packed ? select_packed_real_child(node, request, batch, false) : std::nullopt) {
    const int64_t packed = n / 2;
    DeviceAllocation packed_output =
        adaptor::Memory(static_cast<std::size_t>(batch * packed * element_bytes));
    return std::make_shared<CompiledRawPackedR2CNode>(
        n,
        compile_raw_node(packed_child->plan, packed_child->request, batch),
        compile_r2c_packed_postprocess_kernel(request, n),
        build_raw_packed_real_twiddle(request, n),
        std::move(packed_output),
        [node, request, batch]() {
          TritonCompiler compiler;
          return compiler.compile_raw_r2c_node(node, request, batch, false);
        });
  }
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    if (row_leaf != nullptr && col_leaf != nullptr) {
      DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
      return std::make_shared<CompiledRawR2CFourStepRealInHalfOutNode>(
          n,
          four_step->n1,
          four_step->n2,
          compile_four_step_real_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*row_leaf, request),
          compile_four_step_r2c_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*col_leaf, request),
          std::move(twiddle),
          std::move(stage1));
    }
  }
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    return std::make_shared<CompiledRawR2CLeafNode>(n,
                                                    compile_leaf_r2c_kernel(*leaf, request),
                                                    build_raw_leaf_tables(*leaf, request));
  }
  DeviceAllocation complex_input = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation full_output = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  return std::make_shared<CompiledRawR2CNode>(n,
                                              compile_real_to_complex_kernel(request, n),
                                              compile_raw_node(node, request, batch),
                                              compile_r2c_half_pack_kernel(request, n),
                                              std::move(complex_input),
                                              std::move(full_output));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_c2r_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch,
                                                                      bool allow_packed) {
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = request.requested_n;
  if (real_direct_dft_enabled(node, request, batch)) {
    return compile_raw_real_direct_dft(request, true);
  }
  if (use_ix_prime_real_bluestein(node, request, batch)) {
    return compile_raw_node(node, request, batch);
  }
  if (auto packed_child =
          allow_packed ? select_packed_real_child(node, request, batch, true) : std::nullopt) {
    const int64_t packed = n / 2;
    DeviceAllocation packed_input = adaptor::Memory(static_cast<std::size_t>(batch * packed * element_bytes));
    return std::make_shared<CompiledRawPackedC2RNode>(
        n,
        compile_c2r_packed_preprocess_kernel(request, n),
        compile_raw_node(packed_child->plan, packed_child->request, batch),
        build_raw_packed_real_twiddle(request, n),
        std::move(packed_input),
        [node, request, batch]() {
          TritonCompiler compiler;
          return compiler.compile_raw_c2r_node(node, request, batch, false);
        });
  }
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    if (row_leaf != nullptr && col_leaf != nullptr) {
      DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
      return std::make_shared<CompiledRawC2RFourStepCompactInRealOutNode>(
          n,
          four_step->n1,
          four_step->n2,
          compile_four_step_hermitian_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*row_leaf, request),
          compile_four_step_c2r_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*col_leaf, request),
          std::move(twiddle),
          std::move(stage1));
    }
  }
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    return std::make_shared<CompiledRawC2RLeafNode>(n,
                                                    compile_leaf_c2r_kernel(*leaf, request),
                                                    build_raw_leaf_tables(*leaf, request));
  }
  DeviceAllocation full_input = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation full_output = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  return std::make_shared<CompiledRawC2RNode>(n,
                                              compile_compact_to_hermitian_full_kernel(request, n),
                                              compile_raw_node(node, request, batch),
                                              compile_complex_to_real_kernel(request, n),
                                              std::move(full_input),
                                              std::move(full_output));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_leaf(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf(target,
                                  request.direction,
                                  request.input_dtype,
                                  leaf.length,
                                  leaf.factors,
                                  leaf.lanes,
                                  leaf.num_warps,
                                  leaf.generic_radices,
                                  leaf.smem_size);
  mark_npu_portable_leaf(key, request);
  std::shared_ptr<JitKernel> kernel = compile_kernel(key);
  return std::make_shared<CompiledRawLeafNode>(leaf.length,
                                               std::move(kernel),
                                               build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_permuted_store_leaf(const LeafPlanNode &leaf,
                                                                                const FFTRequest &request,
                                                                                int64_t perm_span,
                                                                                const std::string &perm_form) {
  std::string target = triton_target_for_request(request);
  // The fused store vectorizes along the batch slots. FP64 length-256 uses a
  // smaller pack than FP32, and one warp measured faster than two on S5000.
  // Keep the planner's four warps for the long length-2048 axis.
  int64_t num_warps = std::max<int64_t>(2, leaf.num_warps);
  if (request.device_type == "musa" && request.origin_rank == 3) {
    if (request.input_dtype == "complex128" && leaf.length == 256) num_warps = 1;
    if (const char *value = std::getenv("FLAGFFT_MUSA_3D_FUSED_WARPS")) {
      const int64_t override = std::strtoll(value, nullptr, 10);
      if (override != 1 && override != 2 && override != 4 && override != 8) {
        throw std::runtime_error("FLAGFFT_MUSA_3D_FUSED_WARPS must be 1, 2, 4 or 8");
      }
      num_warps = override;
    }
  }
  if (request.device_type == "ix" && request.origin_rank == 3) {
    if (const char *value = std::getenv("FLAGFFT_IX_3D_FUSED_WARPS")) {
      const int64_t planner_hint = std::strtoll(value, nullptr, 10);
      if (planner_hint != 2 && planner_hint != 4 && planner_hint != 8) {
        throw std::runtime_error("FLAGFFT_IX_3D_FUSED_WARPS must be 2, 4 or 8");
      }
      // The plan stores warp budgets in 32-thread units.  On IX, 2/4/8
      // therefore become 1/2/4 physical 64-thread warps at codegen.
      num_warps = planner_hint;
    }
  }
  KernelKey key = KernelKey::leaf_permuted_store(target,
                                                 request.direction,
                                                 request.input_dtype,
                                                 leaf.length,
                                                 leaf.factors,
                                                 leaf.lanes,
                                                 num_warps,
                                                 leaf.generic_radices,
                                                 leaf.smem_size,
                                                 perm_form);
  mark_npu_portable_leaf(key, request);
  std::shared_ptr<JitKernel> kernel = compile_kernel(key);
  // Same argument shape as the strided leaf: the permutation span rides in the
  // slot that carries outer_stride there, so the node is reused unchanged.
  return std::make_shared<CompiledRawStridedLeafNode>(leaf.length,
                                                      perm_span,
                                                      std::move(kernel),
                                                      build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_strided_leaf(const LeafPlanNode &leaf,
                                                                          const FFTRequest &request,
                                                                          int64_t outer_stride) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_strided(target,
                                          request.direction,
                                          request.input_dtype,
                                          leaf.length,
                                          leaf.factors,
                                          leaf.lanes,
                                          leaf.num_warps,
                                          leaf.generic_radices,
                                          leaf.smem_size);
  mark_npu_portable_leaf(key, request);
  std::shared_ptr<JitKernel> kernel = compile_kernel(key);
  return std::make_shared<CompiledRawStridedLeafNode>(leaf.length,
                                                      outer_stride,
                                                      std::move(kernel),
                                                      build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_strided_direct_dft(const DirectDFTPlanNode &node,
                                                                                const FFTRequest &request,
                                                                                int64_t outer_stride) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::direct_dft_strided(target, request.direction, request.input_dtype, node.length);
  const bool cube_dft = use_npu_2d_cube_dft(request, node.length);
  if (cube_dft) key.kind = KernelKind::DirectDftCubeStrided;
  std::shared_ptr<JitKernel> kernel = compile_kernel(key);
  return std::make_shared<CompiledRawStridedDirectDftNode>(node.length,
                                                           outer_stride,
                                                           std::move(kernel),
                                                           cube_dft ? build_raw_cube_dft_tables(node.length, request)
                                                                    : build_raw_direct_dft_tables(node.length, request));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_direct_dft(const DirectDFTPlanNode &node,
                                                                        const FFTRequest &request,
                                                                        int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  DeviceAllocation input_copy =
      adaptor::Memory(static_cast<std::size_t>(batch * node.length * element_bytes));
  const bool cube_dft = use_npu_2d_cube_dft(request, node.length);
  return std::make_shared<CompiledRawDirectDftNode>(node.length,
                                                    compile_direct_dft_kernel(request, node.length),
                                                    cube_dft ? build_raw_cube_dft_tables(node.length, request)
                                                             : build_raw_direct_dft_tables(node.length, request),
                                                    std::move(input_copy));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_cube_transposed_direct_dft(
    const DirectDFTPlanNode &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  DeviceAllocation input_copy =
      adaptor::Memory(static_cast<std::size_t>(batch * node.length * element_bytes));
  KernelKey key = KernelKey::direct_dft(triton_target_for_request(request),
                                        request.direction,
                                        request.input_dtype,
                                        node.length);
  key.kind = KernelKind::DirectDftCubeTransposed;
  return std::make_shared<CompiledRawDirectDftNode>(node.length,
                                                    compile_kernel(key),
                                                    build_raw_cube_dft_tables(node.length, request),
                                                    std::move(input_copy));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_real_direct_dft(
    const FFTRequest &request, bool inverse) {
  FFTRequest real_request = request;
  real_request.direction = inverse ? "inverse" : "forward";
  const int64_t n = request.requested_n;
  KernelKey key = KernelKey::direct_dft(triton_target_for_request(real_request),
                                       real_request.direction, request.input_dtype, n);
  key.kind = inverse ? KernelKind::DirectDftC2R : KernelKind::DirectDftR2C;
  // Reuse DirectDFT's table/launch ABI and alias protection, copying only the
  // actual input extent (real N scalars or compact N/2+1 complex values).
  const int64_t complex_bytes = complex_element_bytes(request.input_dtype);
  const int64_t input_bytes = inverse ? (n / 2 + 1) * complex_bytes : n * (complex_bytes / 2);
  return std::make_shared<CompiledRawDirectDftNode>(
      n, compile_kernel(key), build_raw_direct_dft_tables(n, real_request),
      adaptor::Memory(static_cast<std::size_t>(request.batch * input_bytes)));
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_r2c_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_r2c(target,
                                      request.direction,
                                      request.input_dtype,
                                      leaf.length,
                                      leaf.factors,
                                      leaf.lanes,
                                      leaf.num_warps,
                                      leaf.generic_radices,
                                      leaf.smem_size);
  mark_npu_portable_leaf(key, request);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_c2r_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_c2r(target,
                                      request.direction,
                                      request.input_dtype,
                                      leaf.length,
                                      leaf.factors,
                                      leaf.lanes,
                                      leaf.num_warps,
                                      leaf.generic_radices,
                                      leaf.smem_size);
  mark_npu_portable_leaf(key, request);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_direct_dft_kernel(const FFTRequest &request, int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::direct_dft(target, request.direction, request.input_dtype, n);
  if (use_npu_2d_cube_dft(request, n)) key.kind = KernelKind::DirectDftCube;
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_row(target,
                                           request.direction,
                                           request.input_dtype,
                                           n1,
                                           n2,
                                           leaf.length,
                                           leaf.factors,
                                           leaf.lanes,
                                           leaf.num_warps,
                                           leaf.generic_radices,
                                           leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_row_strided_kernel(const LeafPlanNode &leaf,
                                                                                const FFTRequest &request,
                                                                                int64_t n1,
                                                                                int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_row_strided(target,
                                                   request.direction,
                                                   request.input_dtype,
                                                   n1,
                                                   n2,
                                                   leaf.length,
                                                   leaf.factors,
                                                   leaf.lanes,
                                                   leaf.num_warps,
                                                   leaf.generic_radices,
                                                   leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_real_row_kernel(const LeafPlanNode &leaf,
                                                                             const FFTRequest &request,
                                                                             int64_t n1,
                                                                             int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_real_row(target,
                                                request.direction,
                                                request.input_dtype,
                                                n1,
                                                n2,
                                                leaf.length,
                                                leaf.factors,
                                                leaf.lanes,
                                                leaf.num_warps,
                                                leaf.generic_radices,
                                                leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_hermitian_row_kernel(const LeafPlanNode &leaf,
                                                                                  const FFTRequest &request,
                                                                                  int64_t n1,
                                                                                  int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_hermitian_row(target,
                                                     request.direction,
                                                     request.input_dtype,
                                                     n1,
                                                     n2,
                                                     leaf.length,
                                                     leaf.factors,
                                                     leaf.lanes,
                                                     leaf.num_warps,
                                                     leaf.generic_radices,
                                                     leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_col(target,
                                           request.direction,
                                           request.input_dtype,
                                           n1,
                                           n2,
                                           leaf.length,
                                           leaf.factors,
                                           leaf.lanes,
                                           leaf.num_warps,
                                           leaf.generic_radices,
                                           leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_col_strided_kernel(const LeafPlanNode &leaf,
                                                                                const FFTRequest &request,
                                                                                int64_t n1,
                                                                                int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_col_strided(target,
                                                   request.direction,
                                                   request.input_dtype,
                                                   n1,
                                                   n2,
                                                   leaf.length,
                                                   leaf.factors,
                                                   leaf.lanes,
                                                   leaf.num_warps,
                                                   leaf.generic_radices,
                                                   leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_r2c_col_kernel(const LeafPlanNode &leaf,
                                                                            const FFTRequest &request,
                                                                            int64_t n1,
                                                                            int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_r2c_col(target,
                                               request.direction,
                                               request.input_dtype,
                                               n1,
                                               n2,
                                               leaf.length,
                                               leaf.factors,
                                               leaf.lanes,
                                               leaf.num_warps,
                                               leaf.generic_radices,
                                               leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_c2r_col_kernel(const LeafPlanNode &leaf,
                                                                            const FFTRequest &request,
                                                                            int64_t n1,
                                                                            int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_c2r_col(target,
                                               request.direction,
                                               request.input_dtype,
                                               n1,
                                               n2,
                                               leaf.length,
                                               leaf.factors,
                                               leaf.lanes,
                                               leaf.num_warps,
                                               leaf.generic_radices,
                                               leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                            int64_t n,
                                                                            int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::bluestein_prepare(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                              int64_t n,
                                                                              int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::bluestein_pointwise(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                             int64_t n,
                                                                             int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::bluestein_finalize(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_kernel(const LeafPlanNode &leaf,
                                                                         const FFTRequest &request,
                                                                         int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein(target,
                                            request.direction,
                                            request.input_dtype,
                                            n,
                                            leaf.length,
                                            leaf.factors,
                                            leaf.lanes,
                                            leaf.num_warps,
                                            leaf.generic_radices,
                                            leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_rader_full_kernel(const LeafPlanNode &leaf,
                                                                           const FFTRequest &request,
                                                                           int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_rader_full(target, request.direction, request.input_dtype,
                                             n, leaf.length, leaf.factors, leaf.lanes,
                                             leaf.num_warps, leaf.generic_radices, leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_rader_prepare_kernel(const LeafPlanNode &leaf,
                                                                              const FFTRequest &request,
                                                                              int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_rader_prepare(target, request.direction, request.input_dtype,
                                                n, leaf.length, leaf.factors, leaf.lanes,
                                                leaf.num_warps, leaf.generic_radices, leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_rader_finish_kernel(const LeafPlanNode &leaf,
                                                                             const FFTRequest &request,
                                                                             int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_rader_finish(target, request.direction, request.input_dtype,
                                               n, leaf.length, leaf.factors, leaf.lanes,
                                               leaf.num_warps, leaf.generic_radices, leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_prepare_kernel(const LeafPlanNode &leaf,
                                                                                 const FFTRequest &request,
                                                                                 int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein_prepare(target,
                                                    request.direction,
                                                    request.input_dtype,
                                                    n,
                                                    leaf.length,
                                                    leaf.factors,
                                                    leaf.lanes,
                                                    leaf.num_warps,
                                                    leaf.generic_radices,
                                                    leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_finish_kernel(const LeafPlanNode &leaf,
                                                                                const FFTRequest &request,
                                                                                int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein_finish(target,
                                                   request.direction,
                                                   request.input_dtype,
                                                   n,
                                                   leaf.length,
                                                   leaf.factors,
                                                   leaf.lanes,
                                                   leaf.num_warps,
                                                   leaf.generic_radices,
                                                   leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_prepare_kernel(const FFTRequest &request,
                                                                        int64_t n,
                                                                        int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_prepare(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_pointwise_kernel(const FFTRequest &request,
                                                                          int64_t n,
                                                                          int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_pointwise(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_finalize_kernel(const FFTRequest &request,
                                                                         int64_t n,
                                                                         int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_finalize(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_reshape_pack_kernel(const FFTRequest &request,
                                                                       int64_t n1,
                                                                       int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::reshape_pack(target, request.input_dtype, n1, n2);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                               int64_t n1,
                                                                               int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::twiddle_reshape_pack(target, request.input_dtype, n1, n2);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_real_to_complex_kernel(const FFTRequest &request,
                                                                          int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::real_to_complex(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_r2c_half_pack_kernel(const FFTRequest &request,
                                                                        int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::r2c_half_pack(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_r2c_packed_postprocess_kernel(const FFTRequest &request,
                                                                                 int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::r2c_packed_postprocess(target, complex_dtype_for(request.input_dtype), n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_c2r_packed_preprocess_kernel(const FFTRequest &request,
                                                                                int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::c2r_packed_preprocess(target, complex_dtype_for(request.input_dtype), n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_compact_to_hermitian_full_kernel(const FFTRequest &request,
                                                                                    int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::compact_to_hermitian_full(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_complex_to_real_kernel(const FFTRequest &request,
                                                                          int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::complex_to_real(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_tiled_transpose_kernel(const FFTRequest &request,
                                                                          int64_t n0,
                                                                          int64_t n1) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::tiled_transpose(target, request.input_dtype, n0, n1);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_transpose3d_kernel(
    const FFTRequest &request, int64_t n0, int64_t n1, int64_t n2, const std::string &order) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::transpose3d(target, request.input_dtype, n0, n1, n2, order);
  return compile_kernel(key);
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;

  // Build per-axis C2C requests.  Each axis is processed as a batch of
  // contiguous rows after the corresponding axis permutation.
  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * n2, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * n2;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * n2, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * n2;

  // Strided fast path: when both non-contiguous axes are plain leaves, run
  // them directly on the natural layout with their own stride and skip the
  // three full-cube permutations.  The strided passes coalesce far worse
  // than a contiguous one (the leaf spreads its lanes along the FFT axis,
  // which is exactly the strided direction), so this only pays off while
  // the cube still fits in L2 and the miss cost is absorbed: measured
  // faster up to 64^3 and slower from 96^3 on.  Above that the permutations
  // win and the caller falls through to the RTRT path.
  constexpr int64_t kStridedMaxElements = 64 * 64 * 64;
  auto n1_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n1_plan);
  auto n0_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n0_plan);
  auto n2_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n2_plan);

  const bool ix_fused_cube = request.device_type == "ix" && request.device_arch == "71" &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      batch == 1 && n0 == 16 && n1 == 16 && n2 == 16 &&
      n0_leaf && n1_leaf && n2_leaf &&
      flag_or_default("FLAGFFT_IX_3D_FUSED16_CUBE", true);
  if (ix_fused_cube) {
    std::vector<float> tw_r(16);
    std::vector<float> tw_i(16);
    const double sign = request.direction == "inverse" ? 1.0 : -1.0;
    for (int64_t k = 0; k < 16; ++k) {
      const double angle = sign * 2.0 * kPi * static_cast<double>(k) / 16.0;
      tw_r[k] = static_cast<float>(std::cos(angle));
      tw_i[k] = static_cast<float>(std::sin(angle));
    }
    auto kernel = compile_kernel(KernelKey::fused_16_cube(
        triton_target_for_request(request), request.direction, request.input_dtype));
    return std::make_shared<CompiledRaw3DFusedCubeNode>(
        std::move(kernel), adaptor::Memory::from_floats(tw_r),
        adaptor::Memory::from_floats(tw_i));
  }

  // A small plane fits in one block. Transform n2 and n1 together so a
  // cube needs only one plane launch plus the outer strided leaf.
  const char *fused16_override = std::getenv("FLAGFFT_MUSA_3D_FUSED16");
  const bool fused16_enabled =
      (request.device_type == "musa" &&
       (fused16_override == nullptr || std::string(fused16_override) != "0")) ||
      (request.device_type == "ix" && request.device_arch == "71" &&
       request.input_dtype == "complex64" && request.output_dtype == request.input_dtype &&
       flag_or_default("FLAGFFT_IX_3D_FUSED16", true));
  const bool fused32_enabled = request.device_type == "ix" && request.device_arch == "71" &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      flag_or_default("FLAGFFT_IX_3D_FUSED32", true);
  const int64_t fused_size = fused16_enabled && n0 == 16 && n1 == 16 && n2 == 16 ? 16 :
      fused32_enabled && n0 == 32 && n1 == 32 && n2 == 32 ? 32 : 0;
  if (fused_size != 0 &&
      batch <= 4 && n0_leaf && n1_leaf && n2_leaf) {
    std::vector<double> tw_r_d(fused_size / 2);
    std::vector<double> tw_i_d(fused_size / 2);
    const double sign = request.direction == "inverse" ? 1.0 : -1.0;
    for (int64_t k = 0; k < fused_size / 2; ++k) {
      const double angle = sign * 2.0 * kPi * static_cast<double>(k) /
          static_cast<double>(fused_size);
      tw_r_d[k] = std::cos(angle);
      tw_i_d[k] = std::sin(angle);
    }
    DeviceAllocation tw_r;
    DeviceAllocation tw_i;
    if (request.input_dtype == "complex128") {
      tw_r = adaptor::Memory::from_doubles(tw_r_d);
      tw_i = adaptor::Memory::from_doubles(tw_i_d);
    } else {
      tw_r = adaptor::Memory::from_floats(std::vector<float>(tw_r_d.begin(), tw_r_d.end()));
      tw_i = adaptor::Memory::from_floats(std::vector<float>(tw_i_d.begin(), tw_i_d.end()));
    }
    auto plane_key = fused_size == 16
        ? KernelKey::fused_16_plane(triton_target_for_request(request), request.direction,
                                    request.input_dtype)
        : KernelKey::fused_32_plane(triton_target_for_request(request), request.direction,
                                    request.input_dtype);
    auto plane_fft = compile_kernel(plane_key);
    std::shared_ptr<CompiledRawNode> outer_fft;
    const bool ix_column = fused_size == 32 && request.device_type == "ix" &&
        request.device_arch == "71" && flag_or_default("FLAGFFT_IX_3D_32_COLUMN", true);
    if (ix_column) {
      auto column_kernel = compile_kernel(KernelKey::fused_32_column(
          triton_target_for_request(request), request.direction, request.input_dtype));
      outer_fft = std::make_shared<CompiledRaw3DColumnNode>(
          fused_size * fused_size, 16, std::move(column_kernel),
          adaptor::Memory::from_floats(std::vector<float>(tw_r_d.begin(), tw_r_d.end())),
          adaptor::Memory::from_floats(std::vector<float>(tw_i_d.begin(), tw_i_d.end())));
    } else {
      outer_fft = compile_raw_strided_leaf(*n0_leaf, request, fused_size * fused_size);
    }
    DeviceAllocation temp = adaptor::Memory(
        static_cast<std::size_t>(batch * fused_size * fused_size * fused_size * element_bytes));
    return std::make_shared<CompiledRaw3DFusedPlaneNode>(
        fused_size, fused_size * fused_size, std::move(plane_fft),
        std::move(outer_fft), std::move(temp),
        std::move(tw_r), std::move(tw_i));
  }

  // At 128x2048x64 in MUSA FP32, a contiguous n1 leaf plus one tiled
  // transpose beats the packed permuted-store leaf.  FP64 measured slower
  // with this exchange, so it stays on the fully fused path.  The short
  // outer axes keep their fused stores, and the axes commute for inverse.
  const bool musa_hybrid = request.device_type == "musa" && request.input_dtype == "complex64" &&
      fused_3d_store_enabled() && n1 >= 1024 && n1 >= 4 * std::max(n0, n2);
  const bool ix_hybrid = request.device_type == "ix" && request.device_arch == "71" &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      (n2 == 64 || n2 == 256) && flag_or_default("FLAGFFT_IX_3D_HYBRID", true);
  // IX also screens the prime middle axis: its non-leaf FFT still consumes
  // contiguous rows, while the short outer leaves can fold two permutations.
  if ((musa_hybrid || ix_hybrid) && n2_leaf && n0_leaf &&
      (n1_leaf || ix_hybrid) &&
      batch * n0 * n1 * n2 > kStridedMaxElements) {
    const char *ix_last_transpose = std::getenv("FLAGFFT_IX_3D_HYBRID_LAST_TRANSPOSE");
    const bool final_transpose = ix_hybrid && ix_last_transpose != nullptr &&
                                 std::string(ix_last_transpose) == "1";
    auto n2_fft = compile_raw_permuted_store_leaf(*n2_leaf, n2_request, n1, "outer");
    auto n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * n2);
    auto perm_210 = compile_transpose3d_kernel(request, n0, n2, n1, "210");
    auto n0_fft = final_transpose
        ? compile_raw_node(node->n0_plan, n0_request, batch * n1 * n2)
        : compile_raw_permuted_store_leaf(*n0_leaf, n0_request, n1 * n2, "outer");
    auto perm_201 = final_transpose
        ? compile_transpose3d_kernel(request, n1, n2, n0, "201")
        : std::shared_ptr<JitKernel>{};
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
    DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
    return std::make_shared<CompiledRaw3DHybridNode>(n0,
                                                     n1,
                                                     n2,
                                                     std::move(n2_fft),
                                                     std::move(n1_fft),
                                                     std::move(n0_fft),
                                                     std::move(perm_210),
                                                     std::move(perm_201),
                                                     std::move(temp1),
                                                     std::move(temp2));
  }

  // Fused fast path: each axis runs as a leaf whose store also applies the
  // permutation the next axis wants, so three FFT passes plus three full-cube
  // transposes collapse into three passes.  Worth it only where the standalone
  // transpose is expensive next to the FFT pass -- on MUSA it is (490 against
  // 789 GB/s), on A100 it is not (already vectorized), where the fused store
  // measured slower than the FFT pass it replaces.  Forward and inverse share
  // the chain because the per-axis transforms commute and only the final
  // layout has to be the natural one.  Small cubes keep the strided path
  // below, whose win there is already established, so the two are disjoint.
  if (n2_leaf && n1_leaf && n0_leaf && fused_3d_store_enabled() &&
      batch * n0 * n1 * n2 > kStridedMaxElements) {
    std::shared_ptr<CompiledRawNode> n2_fft =
        compile_raw_permuted_store_leaf(*n2_leaf, n2_request, /*perm_span=*/n1, "outer");
    std::shared_ptr<CompiledRawNode> n1_fft =
        compile_raw_permuted_store_leaf(*n1_leaf, n1_request, /*perm_span=*/n2, "inner");
    std::shared_ptr<CompiledRawNode> n0_fft =
        compile_raw_permuted_store_leaf(*n0_leaf, n0_request, /*perm_span=*/n1 * n2, "outer");

    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
    DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

    return std::make_shared<CompiledRaw3DStridedNode>(n0,
                                                      n1,
                                                      n2,
                                                      std::move(n2_fft),
                                                      std::move(n1_fft),
                                                      std::move(n0_fft),
                                                      std::move(temp1),
                                                      std::move(temp2));
  }

  if (n1_leaf && n0_leaf && batch * n0 * n1 * n2 <= kStridedMaxElements) {
    std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
    std::shared_ptr<CompiledRawNode> n1_fft =
        compile_raw_strided_leaf(*n1_leaf, request, /*outer_stride=*/n2);
    std::shared_ptr<CompiledRawNode> n0_fft =
        compile_raw_strided_leaf(*n0_leaf, request, /*outer_stride=*/n1 * n2);

    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
    DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

    return std::make_shared<CompiledRaw3DStridedNode>(n0,
                                                      n1,
                                                      n2,
                                                      std::move(n2_fft),
                                                      std::move(n1_fft),
                                                      std::move(n0_fft),
                                                      std::move(temp1),
                                                      std::move(temp2));
  }

  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * n2);
  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * n2);

  // Forward: (n0,n1,n2) -021-> (n0,n2,n1) -210-> (n1,n2,n0) -201-> (n0,n1,n2).
  // Inverse: (n0,n1,n2) -120-> (n1,n2,n0) -210-> (n0,n2,n1) -021-> (n0,n1,n2).
  auto perm_021_fwd = compile_transpose3d_kernel(request, n0, n1, n2, "021");
  auto perm_210_fwd = compile_transpose3d_kernel(request, n0, n2, n1, "210");
  auto perm_201_fwd = compile_transpose3d_kernel(request, n1, n2, n0, "201");
  auto perm_120_inv = compile_transpose3d_kernel(request, n0, n1, n2, "120");
  auto perm_210_inv = compile_transpose3d_kernel(request, n1, n2, n0, "210");
  auto perm_021_inv = compile_transpose3d_kernel(request, n0, n2, n1, "021");

  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

  return std::make_shared<CompiledRaw3DNode>(n0,
                                             n1,
                                             n2,
                                             std::move(n2_fft),
                                             std::move(n1_fft),
                                             std::move(n0_fft),
                                             std::move(perm_021_fwd),
                                             std::move(perm_210_fwd),
                                             std::move(perm_201_fwd),
                                             std::move(perm_120_inv),
                                             std::move(perm_210_inv),
                                             std::move(perm_021_inv),
                                             std::move(temp1),
                                             std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_real_leaf_node(
    const std::shared_ptr<ThreeDimPlanNode> &node,
    const FFTRequest &request,
    int64_t batch,
    bool inverse) {
  // Screen IX's small strided and large fused-store real paths separately.
  const char *ix_real_fused_override = std::getenv("FLAGFFT_IX_3D_REAL_FUSED");
  const bool ix_real_leaf_screen = request.device_type == "ix" && request.device_arch == "71" &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      flag_or_default("FLAGFFT_IX_3D_REAL_LEAF", true);
  const bool ix_real_fused_screen = request.device_type == "ix" && request.device_arch == "71" &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      ix_real_fused_override != nullptr && std::string(ix_real_fused_override) == "1";
  const char *npu_3d_real_leaf = std::getenv("FLAGFFT_NPU_3D_LEAF");
  const bool npu_real_leaf_screen = request.device_type == "npu" && request.origin_rank == 3 &&
      (request.real_transform_kind == "r2c" || request.real_transform_kind == "c2r") &&
      npu_3d_real_leaf != nullptr && std::string(npu_3d_real_leaf) == "1";
  if (request.device_type != "musa" && !ix_real_leaf_screen && !ix_real_fused_screen &&
      !npu_real_leaf_screen) return nullptr;
  auto n2_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n2_plan);
  auto n1_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n1_plan);
  auto n0_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n0_plan);
  if (!n2_leaf || !n1_leaf || !n0_leaf) return nullptr;

  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;
  const int64_t packed = batch * n0 * n1 * half;
  const bool small = packed <= 64 * 64 * 64;
  if (ix_real_leaf_screen && !ix_real_fused_screen && !small) return nullptr;
  if (!small && !fused_3d_store_enabled() && !ix_real_fused_screen) return nullptr;

  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  const bool fused_real_cube = !inverse && ix_real_leaf_screen && batch == 1 &&
      n0 == 16 && n1 == 16 && n2 == 16 &&
      flag_or_default("FLAGFFT_IX_3D_REAL_FUSED16_CUBE", true);
  if (fused_real_cube) {
    std::vector<float> tw_r(16);
    std::vector<float> tw_i(16);
    for (int64_t k = 0; k < 16; ++k) {
      const double angle = -2.0 * kPi * static_cast<double>(k) / 16.0;
      tw_r[k] = static_cast<float>(std::cos(angle));
      tw_i[k] = static_cast<float>(std::sin(angle));
    }
    auto kernel = compile_kernel(KernelKey::fused_16_real_cube(
        triton_target_for_request(request), request.direction, request.input_dtype));
    return std::make_shared<CompiledRaw3DFusedCubeNode>(
        std::move(kernel), adaptor::Memory::from_floats(tw_r),
        adaptor::Memory::from_floats(tw_i));
  }

  const bool fused_real_plane = !inverse && ix_real_leaf_screen && small && batch <= 4 &&
      n0 == n1 && n1 == n2 && (n0 == 16 || n0 == 32) &&
      flag_or_default("FLAGFFT_IX_3D_REAL_FUSED_PLANE", true);
  if (fused_real_plane) {
    std::vector<float> tw_r(n0 / 2);
    std::vector<float> tw_i(n0 / 2);
    for (int64_t k = 0; k < n0 / 2; ++k) {
      const double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n0);
      tw_r[k] = static_cast<float>(std::cos(angle));
      tw_i[k] = static_cast<float>(std::sin(angle));
    }
    auto key = n0 == 16
        ? KernelKey::fused_16_real_plane(triton_target_for_request(request), request.direction,
                                         request.input_dtype)
        : KernelKey::fused_32_real_plane(triton_target_for_request(request), request.direction,
                                         request.input_dtype);
    auto plane_fft = compile_kernel(key);
    std::shared_ptr<CompiledRawNode> outer_fft;
    if (n0 == 32 && flag_or_default("FLAGFFT_IX_3D_32_COLUMN", true)) {
      const int64_t columns = batch == 1 && flag_or_default("FLAGFFT_IX_3D_REAL_COLUMN8", true) ? 8 : 16;
      auto column_kernel = compile_kernel(KernelKey::fused_32_column(
          triton_target_for_request(request), request.direction, request.input_dtype, columns));
      outer_fft = std::make_shared<CompiledRaw3DColumnNode>(
          n1 * half, columns, std::move(column_kernel),
          adaptor::Memory::from_floats(tw_r), adaptor::Memory::from_floats(tw_i));
    } else {
      outer_fft = compile_raw_strided_leaf(*n0_leaf, n0_request, n1 * half);
    }
    DeviceAllocation temp = adaptor::Memory(
        static_cast<std::size_t>(packed * complex_element_bytes(request.input_dtype)));
    DeviceAllocation tw_r_dev = adaptor::Memory::from_floats(tw_r);
    DeviceAllocation tw_i_dev = adaptor::Memory::from_floats(tw_i);
    return std::make_shared<CompiledRaw3DFusedPlaneNode>(
        n0, n1 * half, std::move(plane_fft), std::move(outer_fft), std::move(temp),
        std::move(tw_r_dev), std::move(tw_i_dev));
  }

  auto n2_real_fft = inverse
      ? compile_raw_c2r_node(node->n2_plan, n2_request, batch * n0 * n1, false)
      : compile_raw_r2c_node(node->n2_plan, n2_request, batch * n0 * n1, false);
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<JitKernel> perm_021;
  if (small) {
    n1_fft = compile_raw_strided_leaf(*n1_leaf, n1_request, half);
    n0_fft = compile_raw_strided_leaf(*n0_leaf, n0_request, n1 * half);
  } else {
    // The first permutation makes n1 rows contiguous.  Each following leaf
    // writes in the layout consumed by the next axis, leaving natural compact
    // (n0,n1,half) order after n0.
    perm_021 = compile_transpose3d_kernel(request, n0, n1, half, "021");
    n1_fft = compile_raw_permuted_store_leaf(*n1_leaf, n1_request, half, "inner");
    n0_fft = compile_raw_permuted_store_leaf(*n0_leaf, n0_request, n1 * half, "outer");
  }

  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(packed * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(packed * element_bytes));
  return std::make_shared<CompiledRaw3DRealLeafNode>(n0,
                                                     n1,
                                                     n2,
                                                     inverse,
                                                     !small,
                                                     std::move(n2_real_fft),
                                                     std::move(n1_fft),
                                                     std::move(n0_fft),
                                                     std::move(perm_021),
                                                     std::move(temp1),
                                                     std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_real_rtrt_node(
    const std::shared_ptr<ThreeDimPlanNode> &node,
    const FFTRequest &request,
    int64_t batch,
    bool inverse) {
  const char *ix_rtrt_override = std::getenv("FLAGFFT_IX_3D_REAL_RTRT");
  const bool screen_rtrt = ix_rtrt_override != nullptr && std::string(ix_rtrt_override) == "1";
  const bool screen_hybrid = (node->n2 == 64 || node->n2 == 256) &&
      flag_or_default("FLAGFFT_IX_3D_REAL_HYBRID", !screen_rtrt);
  if (request.device_type != "ix" || request.device_arch != "71" ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      (!screen_rtrt && !screen_hybrid)) return nullptr;

  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;
  const int64_t packed = batch * n0 * n1 * half;
  auto n2_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n2_plan);
  auto n1_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n1_plan);
  auto n0_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->n0_plan);
  const bool fused_n0 = screen_hybrid && n0_leaf && packed > 64 * 64 * 64;
  // The middle store removes one full-cube transpose for single 256^3 R2C.
  // Batch four and the elongated shape measured slower, so keep this narrow.
  const bool fused_middle = !inverse && screen_hybrid && n1_leaf &&
      packed > 64 * 64 * 64 &&
      flag_or_default("FLAGFFT_IX_3D_R2C_FUSED_MIDDLE",
                      batch == 1 && n0 == 256 && n1 == 256 && n2 == 256);
  const bool fused_first = !inverse && n2_leaf && (n2 == 64 || n2 == 256) &&
      packed > 64 * 64 * 64 && flag_or_default("FLAGFFT_IX_3D_R2C_FUSED_FIRST", true);

  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  std::shared_ptr<CompiledRawNode> n2_real_fft;
  if (fused_first) {
    KernelKey key = KernelKey::leaf_r2c(triton_target_for_request(n2_request),
                                         n2_request.direction, n2_request.input_dtype,
                                         n2_leaf->length, n2_leaf->factors,
                                         n2_leaf->lanes, n2_leaf->num_warps,
                                         n2_leaf->generic_radices, n2_leaf->smem_size);
    key.perm_form = "permuted";
    n2_real_fft = std::make_shared<CompiledRawR2CLeafNode>(
        n2, compile_kernel(key), build_raw_leaf_tables(*n2_leaf, n2_request));
  } else {
    n2_real_fft = inverse
        ? compile_raw_c2r_node(node->n2_plan, n2_request, batch * n0 * n1, false)
        : compile_raw_r2c_node(node->n2_plan, n2_request, batch * n0 * n1, false);
  }
  auto n1_fft = fused_middle
      ? compile_raw_permuted_store_leaf(*n1_leaf, n1_request, half, "inner")
      : compile_raw_node(node->n1_plan, n1_request, batch * n0 * half);
  auto n0_fft = fused_n0
      ? compile_raw_permuted_store_leaf(*n0_leaf, n0_request, n1 * half, "outer")
      : compile_raw_node(node->n0_plan, n0_request, batch * n1 * half);
  auto perm_021 = fused_first
      ? std::shared_ptr<JitKernel>{}
      : compile_transpose3d_kernel(request, n0, n1, half, "021");
  auto perm_210 = fused_middle
      ? std::shared_ptr<JitKernel>{}
      : compile_transpose3d_kernel(request, n0, half, n1, "210");
  auto perm_201 = fused_n0
      ? std::shared_ptr<JitKernel>{}
      : compile_transpose3d_kernel(request, n1, half, n0, "201");

  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(packed * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(packed * element_bytes));
  return std::make_shared<CompiledRaw3DRealRTRTNode>(n0,
                                                     n1,
                                                     n2,
                                                     inverse,
                                                     std::move(n2_real_fft),
                                                     std::move(n1_fft),
                                                     std::move(n0_fft),
                                                     std::move(perm_021),
                                                     std::move(perm_210),
                                                     std::move(perm_201),
                                                     std::move(temp1),
                                                     std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_r2c_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  configure_single_transform_policies(request);
  if (auto leaf_path = compile_raw_3d_real_leaf_node(node, request, batch, false)) return leaf_path;
  if (auto rtrt_path = compile_raw_3d_real_rtrt_node(node, request, batch, false)) return rtrt_path;
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;

  // The innermost axis n2 runs expand + C2C FFT + half-pack; the remaining
  // two axes are plain C2C after the axis permutations.
  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  auto expand_kernel = compile_real_to_complex_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  auto pack_kernel = compile_r2c_half_pack_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * half);
  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * half);

  // (n0,n1,half) -021-> (n0,half,n1) -210-> (n1,half,n0) -201-> (n0,n1,half).
  auto perm_021 = compile_transpose3d_kernel(request, n0, n1, half, "021");
  auto perm_210 = compile_transpose3d_kernel(request, n0, half, n1, "210");
  auto perm_201 = compile_transpose3d_kernel(request, n1, half, n0, "201");

  DeviceAllocation row_fft_buf =
      adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));

  return std::make_shared<CompiledRaw3DR2CNode>(n0,
                                                n1,
                                                n2,
                                                std::move(expand_kernel),
                                                std::move(n2_fft),
                                                std::move(pack_kernel),
                                                std::move(n1_fft),
                                                std::move(n0_fft),
                                                std::move(perm_021),
                                                std::move(perm_210),
                                                std::move(perm_201),
                                                std::move(row_fft_buf),
                                                std::move(temp1),
                                                std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_c2r_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  configure_single_transform_policies(request);
  if (auto leaf_path = compile_raw_3d_real_leaf_node(node, request, batch, true)) return leaf_path;
  if (auto rtrt_path = compile_raw_3d_real_rtrt_node(node, request, batch, true)) return rtrt_path;
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;

  // C2R is the reverse of R2C: permute the half-packed cube, IFFT along n0
  // and n1, expand half -> full Hermitian, IFFT along n2, pack complex -> real.
  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * half);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * half);
  auto expand_kernel = compile_compact_to_hermitian_full_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  auto pack_kernel = compile_complex_to_real_kernel(request, n2);

  // (n0,n1,half) -120-> (n1,half,n0) -210-> (n0,half,n1) -021-> (n0,n1,half).
  auto perm_120 = compile_transpose3d_kernel(request, n0, n1, half, "120");
  auto perm_210 = compile_transpose3d_kernel(request, n1, half, n0, "210");
  auto perm_021 = compile_transpose3d_kernel(request, n0, half, n1, "021");

  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation full_buf = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

  return std::make_shared<CompiledRaw3DC2RNode>(n0,
                                                n1,
                                                n2,
                                                std::move(perm_120),
                                                std::move(perm_210),
                                                std::move(perm_021),
                                                std::move(n0_fft),
                                                std::move(n1_fft),
                                                std::move(expand_kernel),
                                                std::move(n2_fft),
                                                std::move(pack_kernel),
                                                std::move(temp1),
                                                std::move(temp2),
                                                std::move(full_buf));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_rc_row(
    const PlanNodePtr &node, const FFTRequest &request, int64_t batch) {
  const Maca2dPolicyScope child_scope(
      maca_2d_single_policy_,
      maca_2d_single_policy_ && !(batch > 1 && maca_2d_rc_preserve_batched_row(node)));
  return compile_raw_node(node, request, batch);
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const Maca2dPolicyScope policy_scope(maca_2d_single_policy_, request, batch, node->n0, node->n1);
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const bool rc_eligible = n0 <= 256;
  // On MUSA S5000, replaying the short batch-1 complex 2D graph adds
  // about 0.1 ms versus direct launches (both RC and transpose paths).
  // Keep other devices and unmeasured batch sizes on the existing policy.
  // Per-kernel event timing synchronizes launches, which is invalid during
  // stream capture. Keep diagnostic runs on the direct sequence.
  const bool enable_graph =
      !env_flag_enabled(std::getenv("FLAGFFT_PROFILE_KERNELS")) &&
      !(request.device_type == "musa" && request.device_arch == "31" && batch == 1) &&
      (request.device_type != "maca" ||
       maca_flag_or_default("FLAGFFT_MACA_2D_GRAPH", !maca_2d_single_policy_));

  // Build row FFT request (axis-1, length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Degenerate 2D shapes: a 2D FFT with one unit axis is just a batched 1D FFT.
  if (n0 == 1) {
    row_request.batch = batch;
    row_request.input_shape = {batch, n1};
    return std::make_shared<CompiledRaw1DAs2DNode>(compile_raw_node(node->row_plan, row_request, batch),
                                                   batch);
  }

  const char *npu_fused_2d = std::getenv("FLAGFFT_NPU_2D_FUSED_RADIX");
  if (request.device_type == "npu" && request.input_dtype == "complex64" &&
      request.output_dtype == "complex64" && n0 == 64 && n1 == 64 &&
      npu_fused_2d != nullptr && std::string(npu_fused_2d) == "1") {
    std::vector<float> tw_r(n0 / 2);
    std::vector<float> tw_i(n0 / 2);
    const double sign = request.direction == "inverse" ? 1.0 : -1.0;
    for (int64_t k = 0; k < n0 / 2; ++k) {
      const double angle = sign * 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n0);
      tw_r[k] = static_cast<float>(std::cos(angle));
      tw_i[k] = static_cast<float>(std::sin(angle));
    }
    auto kernel = compile_kernel(KernelKey::fused_2d(
        triton_target_for_request(request), request.direction, request.input_dtype, n0, true));
    DeviceAllocation temp = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DFusedNode>(
        n0, n1, std::move(kernel), adaptor::Memory::from_floats(tw_r),
        adaptor::Memory::from_floats(tw_i), std::move(temp));
  }

  // Build col FFT request (axis-0, length=n0, batch=batch*n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * n1;

  if (n1 == 1) {
    col_request.batch = batch;
    col_request.input_shape = {batch, n0};
    return std::make_shared<CompiledRaw1DAs2DNode>(compile_raw_node(node->col_plan, col_request, batch),
                                                   batch);
  }

#if defined(FLAGFFT_BACKEND_NPU)
  const char *npu_aiv_fft64 = std::getenv("FLAGFFT_NPU_2D_ASCENDC_AIV");
  const bool use_npu_aiv_fft64 =
      request.device_type == "npu" && request.origin_rank == 2 &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      n0 == 64 && n1 == 64 && npu_aiv_fft64 != nullptr &&
      std::string(npu_aiv_fft64) == "1" &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->row_plan) != nullptr &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan) != nullptr;
  if (use_npu_aiv_fft64) {
    auto parse_group_size = [](const char *setting, int64_t batch) {
      if (setting == nullptr) return int32_t{1};
      const std::string value(setting);
      // A single matrix benefits from four-way groups, while the row and
      // column profiles show that batch workloads benefit from eight-way
      // tiling. Keep this choice local to the existing FFT64 leaf plans.
      if (value == "auto") return batch > 1 ? int32_t{8} : int32_t{4};
      if (value == "4") return int32_t{4};
      if (value == "8") return int32_t{8};
      return int32_t{1};
    };
    const char *row_group_setting = std::getenv("FLAGFFT_NPU_2D_ASCENDC_ROW_GROUP");
    const char *column_group_setting = std::getenv("FLAGFFT_NPU_2D_ASCENDC_COL_GROUP");
    const int32_t row_group_size = parse_group_size(row_group_setting, batch);
    const int32_t col_group_size = parse_group_size(column_group_setting, batch);
    std::vector<uint32_t> host_indices;
    std::vector<float> host_twiddles;
    build_npu_aiv_fft64_tables(request, host_indices, host_twiddles);
    auto upload_tables = [](const std::vector<uint32_t> &indices,
                            const std::vector<float> &twiddles) {
      auto index_allocation = std::make_shared<DeviceAllocation>(indices.size() * sizeof(uint32_t));
      index_allocation->copy_from_host(indices.data(), indices.size() * sizeof(uint32_t));
      auto twiddle_allocation = std::make_shared<DeviceAllocation>(twiddles.size() * sizeof(float));
      twiddle_allocation->copy_from_host(twiddles.data(), twiddles.size() * sizeof(float));
      return std::make_pair(std::move(index_allocation), std::move(twiddle_allocation));
    };
    auto [base_indices, base_twiddles] = upload_tables(host_indices, host_twiddles);
    std::shared_ptr<DeviceAllocation> row_indices = base_indices;
    std::shared_ptr<DeviceAllocation> row_twiddles = base_twiddles;
    if (row_group_size > 1) {
      build_npu_aiv_fft64_group_tables(request, row_group_size, false, host_indices, host_twiddles);
      auto grouped_tables = upload_tables(host_indices, host_twiddles);
      row_indices = std::move(grouped_tables.first);
      row_twiddles = std::move(grouped_tables.second);
    }
    std::shared_ptr<DeviceAllocation> col_indices = base_indices;
    std::shared_ptr<DeviceAllocation> col_twiddles = base_twiddles;
    if (col_group_size > 1) {
      build_npu_aiv_fft64_group_tables(request, col_group_size, true, host_indices, host_twiddles);
      auto grouped_tables = upload_tables(host_indices, host_twiddles);
      col_indices = std::move(grouped_tables.first);
      col_twiddles = std::move(grouped_tables.second);
    }
    std::shared_ptr<CompiledRawNode> row_fft =
        std::make_shared<CompiledRawNpuAivFFT64Node>(1, row_group_size, row_indices, row_twiddles);
    std::shared_ptr<CompiledRawNode> col_fft =
        std::make_shared<CompiledRawNpuAivFFT64Node>(n1, col_group_size, col_indices, col_twiddles);
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(temp1),
                                                 enable_graph);
  }
#endif

  const auto cube_row_direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node->row_plan);
  const auto cube_col_direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node->col_plan);
  if (n0 == 64 && n1 == 64 && use_npu_2d_cube_dft(request, n0) && cube_row_direct && cube_col_direct) {
    const char *cube_matmul = std::getenv("FLAGFFT_NPU_2D_CUBE_MATMUL");
    if (cube_matmul != nullptr && std::string(cube_matmul) == "1") {
      auto compile_cube_2d_pass = [&](const DirectDFTPlanNode &direct,
                                      const FFTRequest &pass_request,
                                      int64_t pass_batch,
                                      bool column) -> std::shared_ptr<CompiledRawNode> {
        KernelKey key = KernelKey::direct_dft(triton_target_for_request(pass_request),
                                              pass_request.direction,
                                              pass_request.input_dtype,
                                              direct.length);
        key.kind = column ? KernelKind::DirectDftCube2DCol : KernelKind::DirectDftCube2DRow;
        auto kernel = compile_kernel(key);
        auto tables = build_raw_cube_dft_tables(direct.length, pass_request);
        DeviceAllocation input_copy = adaptor::Memory(
            static_cast<std::size_t>(pass_batch * direct.length * element_bytes));
        return std::make_shared<CompiledRawDirectDftNode>(
            direct.length, std::move(kernel), std::move(tables), std::move(input_copy));
      };
      std::shared_ptr<CompiledRawNode> row_fft =
          compile_cube_2d_pass(*cube_row_direct, row_request, batch * n0, false);
      std::shared_ptr<CompiledRawNode> col_fft =
          compile_cube_2d_pass(*cube_col_direct, col_request, batch * n1, true);
      DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
      return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                   n1,
                                                   std::move(row_fft),
                                                   std::move(col_fft),
                                                   std::move(temp1),
                                                   enable_graph);
    }
    const char *cube_transpose_kernels = std::getenv("FLAGFFT_NPU_2D_CUBE_TRANSPOSE_KERNELS");
    if (cube_transpose_kernels != nullptr && std::string(cube_transpose_kernels) == "1") {
      // Keep each Cube DFT tile in its native contiguous layout. Move the
      // matrix between axes with the existing tiled transpose kernels rather
      // than lowering tl.trans into each Cube kernel's epilogue.
      std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
      std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * n1);
      auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, n1);
      auto transpose_inv = compile_tiled_transpose_kernel(request, n1, n0);
      DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
      DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
      return std::make_shared<CompiledRaw2DNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(transpose_fwd),
                                                 std::move(transpose_inv),
                                                 std::move(temp1),
                                                 std::move(temp2),
                                                 enable_graph);
    }
    std::shared_ptr<CompiledRawNode> row_fft =
        compile_raw_cube_transposed_direct_dft(*cube_row_direct, row_request, batch * n0);
    std::shared_ptr<CompiledRawNode> col_fft =
        compile_raw_cube_transposed_direct_dft(*cube_col_direct, col_request, batch * n1);
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(temp1),
                                                 enable_graph);
  }

  // For small Ascend C2C matrices, write each 1D leaf pass in transposed
  // order. The second pass then reads contiguous rows and transposes its
  // output back to natural matrix order, using the existing TwoDim and raw
  // leaf nodes without standalone transpose kernels.
  const char *npu_2d_transpose_store = std::getenv("FLAGFFT_NPU_2D_TRANSPOSE_STORE");
  const bool use_npu_2d_transpose_store =
      request.device_type == "npu" && request.origin_rank == 2 &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      n0 <= 128 && n1 <= 128 && npu_2d_transpose_store != nullptr &&
      std::string(npu_2d_transpose_store) == "1";
  auto npu_row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->row_plan);
  auto npu_col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan);
  if (use_npu_2d_transpose_store && npu_row_leaf && npu_col_leaf) {
    std::shared_ptr<CompiledRawNode> row_fft =
        compile_raw_permuted_store_leaf(*npu_row_leaf, row_request, n0, "outer");
    std::shared_ptr<CompiledRawNode> col_fft =
        compile_raw_permuted_store_leaf(*npu_col_leaf, col_request, n1, "outer");
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(temp1),
                                                 enable_graph);
  }

  // RC fast path: when the column FFT is a plain leaf transform, run it
  // directly on the strided matrix columns and skip both transposes.
  const char *npu_leaf64_transpose_setting =
      std::getenv("FLAGFFT_NPU_2D_LEAF64_TRANSPOSE");
  const bool npu_leaf64_transpose =
      request.device_type == "npu" && request.input_dtype == "complex64" &&
      request.output_dtype == "complex64" && n0 == 64 && n1 == 64 &&
      npu_leaf64_transpose_setting != nullptr &&
      std::string(npu_leaf64_transpose_setting) == "1";
  if (auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan);
      rc_eligible && col_leaf && !npu_leaf64_transpose) {
    std::shared_ptr<CompiledRawNode> row_fft = compile_raw_2d_rc_row(node->row_plan, row_request, batch * n0);
    std::shared_ptr<CompiledRawNode> col_fft = compile_raw_strided_leaf(*col_leaf, request, n1);
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(temp1),
                                                 enable_graph);
  }

  // Small odd column lengths use DirectDFT; keep the same RC structure.
  if (auto col_direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node->col_plan);
      rc_eligible && col_direct && request.device_type != "npu") {
    // The strided NPU DirectDFT kernel exceeds CANN's UB limit for short 2D
    // axes. The generic path below uses contiguous DirectDFT kernels around
    // the already-tiled transpose instead.
    std::shared_ptr<CompiledRawNode> row_fft = compile_raw_2d_rc_row(node->row_plan, row_request, batch * n0);
    std::shared_ptr<CompiledRawNode> col_fft = compile_raw_strided_direct_dft(*col_direct, request, n1);
    DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                 n1,
                                                 std::move(row_fft),
                                                 std::move(col_fft),
                                                 std::move(temp1),
                                                 enable_graph);
  }

  // Large column lengths that decompose into a four-step leaf pair can also
  // run without transposes through the strided four-step kernels.
  if (auto col_four = std::dynamic_pointer_cast<FourStepPlanNode>(node->col_plan); rc_eligible && col_four) {
    std::shared_ptr<CompiledRawNode> col_fft =
        compile_raw_four_step_strided_node(*col_four, request, batch * n1, n1);
    if (col_fft != nullptr) {
      std::shared_ptr<CompiledRawNode> row_fft = compile_raw_2d_rc_row(node->row_plan, row_request, batch * n0);
      DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
      return std::make_shared<CompiledRaw2DRCNode>(n0,
                                                   n1,
                                                   std::move(row_fft),
                                                   std::move(col_fft),
                                                   std::move(temp1),
                                                   enable_graph);
    }
  }

  // Compile row and col FFT nodes
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * n1);

  // Compile transpose kernels
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, n1, n0);

  // Allocate temporary buffers
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));

  return std::make_shared<CompiledRaw2DNode>(n0,
                                             n1,
                                             std::move(row_fft),
                                             std::move(col_fft),
                                             std::move(transpose_fwd),
                                             std::move(transpose_inv),
                                             std::move(temp1),
                                             std::move(temp2),
                                             enable_graph);
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_r2c_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const Maca2dPolicyScope policy_scope(maca_2d_single_policy_, request, batch, node->n0, node->n1);
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t half_n1 = n1 / 2 + 1;
  const bool rc_eligible = n0 <= 256;

  // Build row C2C FFT request (axis-1, length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Degenerate real 2D: a unit axis reduces to a batched 1D R2C.
  if (n0 == 1) {
    row_request.batch = batch;
    row_request.input_shape = {batch, n1};
    return std::make_shared<CompiledRaw1DAs2DNode>(compile_raw_r2c_node(node->row_plan, row_request, batch),
                                                   batch);
  }

#if defined(FLAGFFT_BACKEND_NPU)
  const char *npu_aiv_fft64 = std::getenv("FLAGFFT_NPU_2D_ASCENDC_AIV");
  const bool use_npu_aiv_fft64_real =
      request.device_type == "npu" && request.origin_rank == 2 &&
      request.real_transform_kind == "r2c" && request.input_dtype == "complex64" &&
      request.output_dtype == "complex64" &&
      n0 == 64 && n1 == 64 && npu_aiv_fft64 != nullptr && std::string(npu_aiv_fft64) == "1" &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->row_plan) != nullptr &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan) != nullptr;
  if (use_npu_aiv_fft64_real) {
    const int32_t row_group_size = npu_aiv_fft64_real_row_group_size(
        std::getenv("FLAGFFT_NPU_2D_ASCENDC_ROW_GROUP"), batch);
    auto row_fft = make_npu_aiv_fft64_child(
        request, 1, row_group_size, NpuAivFFT64Mode::RealForward);
    auto col_fft = make_npu_aiv_fft64_child(request, half_n1, 1);
    DeviceAllocation row_fft_buf(
        static_cast<std::size_t>(batch * n0 * half_n1 * element_bytes));
    return std::make_shared<CompiledRaw2DR2CRCNode>(n0,
                                                    n1,
                                                    nullptr,
                                                    std::move(row_fft),
                                                    nullptr,
                                                    std::move(col_fft),
                                                    std::move(row_fft_buf));
  }
#endif

  // MACA FP32 path: compile the innermost real boundary directly so
  // the 2D schedule does not materialize a full complex row matrix merely to
  // discard its Hermitian half.  Restrict this to row plans that the existing
  // 1D real compiler can fuse into a leaf or leaf-pair FourStep node.  Packed
  // real is deliberately disabled here until it is qualified for this layout.
  if (maca_2d_real_rows_enabled(request, batch, n0, n1, maca_2d_single_policy_) &&
      has_real_boundary_row_plan(node->row_plan)) {
    std::shared_ptr<CompiledRawNode> row_r2c =
        compile_raw_r2c_node(node->row_plan, row_request, batch * n0, false);

    FFTRequest col_request = request;
    col_request.fft_length = n0;
    col_request.input_shape = {batch * half_n1, n0};
    col_request.input_strides = {n0, 1};
    col_request.requested_n = n0;
    col_request.batch = batch * half_n1;
    std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);

    auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
    auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);
    const std::size_t compact_bytes = static_cast<std::size_t>(batch * n0 * half_n1 * element_bytes);
    DeviceAllocation temp1 = adaptor::Memory(compact_bytes);
    DeviceAllocation temp2 = adaptor::Memory(compact_bytes);

    return std::make_shared<CompiledRaw2DR2CRowNode>(n0,
                                                     n1,
                                                     std::move(row_r2c),
                                                     std::move(col_fft),
                                                     std::move(transpose_fwd),
                                                     std::move(transpose_inv),
                                                     std::move(temp1),
                                                     std::move(temp2));
  }

  // RC fast path for real transforms: pack the half spectrum, then run the
  // column FFT directly on the strided half-packed matrix.
  auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan);
  auto col_direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node->col_plan);
  auto col_four = std::dynamic_pointer_cast<FourStepPlanNode>(node->col_plan);
  std::shared_ptr<CompiledRawNode> rc_col_fft;
  if (rc_eligible && col_leaf != nullptr) {
    rc_col_fft = compile_raw_strided_leaf(*col_leaf, request, half_n1);
  } else if (rc_eligible && col_direct != nullptr && request.device_type != "npu") {
    rc_col_fft = compile_raw_strided_direct_dft(*col_direct, request, half_n1);
  } else if (rc_eligible && col_four != nullptr) {
    rc_col_fft = compile_raw_four_step_strided_node(*col_four, request, batch * half_n1, half_n1);
  }
  if (rc_col_fft != nullptr) {
    const bool use_npu_real_row = request.device_type == "npu" &&
                                  has_real_boundary_row_plan(node->row_plan) &&
                                  flag_or_default("FLAGFFT_NPU_2D_REAL_ROW", true);
    if (use_npu_real_row) {
      // The strided column path already accepts the compact row-major half
      // spectrum. Let the existing 1D real boundary write that layout
      // directly, avoiding real-to-complex expansion and half-spectrum pack.
      std::shared_ptr<CompiledRawNode> row_r2c =
          compile_raw_r2c_node(node->row_plan, row_request, batch * n0, false);
      const std::size_t compact_bytes =
          static_cast<std::size_t>(batch * n0 * half_n1 * element_bytes);
      DeviceAllocation row_fft_buf = adaptor::Memory(compact_bytes);
      return std::make_shared<CompiledRaw2DR2CRCNode>(n0,
                                                      n1,
                                                      nullptr,
                                                      std::move(row_r2c),
                                                      nullptr,
                                                      std::move(rc_col_fft),
                                                      std::move(row_fft_buf));
    }

    auto expand_kernel = compile_real_to_complex_kernel(request, n1);
    std::shared_ptr<CompiledRawNode> row_fft = compile_raw_2d_rc_row(node->row_plan, row_request, batch * n0);
    auto pack_kernel = compile_r2c_half_pack_kernel(request, n1);
    DeviceAllocation row_fft_buf = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DR2CRCNode>(n0,
                                                    n1,
                                                    std::move(expand_kernel),
                                                    std::move(row_fft),
                                                    std::move(pack_kernel),
                                                    std::move(rc_col_fft),
                                                    std::move(row_fft_buf));
  }

  // Build col C2C FFT request (axis-0, length=n0, batch=batch*half_n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * half_n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * half_n1;

  // Compile kernels
  auto expand_kernel = compile_real_to_complex_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  auto pack_kernel = compile_r2c_half_pack_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);

  // R2C transposes: (n0, half_n1) <-> (half_n1, n0)
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);

  // Allocate buffers
  // row_fft_buf: full complex output from row R2C FFT (batch*n0*n1 complex)
  DeviceAllocation row_fft_buf = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
  // temp1: transposed data (batch * half_n1 * n0 complex)
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp2: col FFT output (batch * half_n1 * n0 complex)
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));

  return std::make_shared<CompiledRaw2DR2CNode>(n0,
                                                n1,
                                                std::move(expand_kernel),
                                                std::move(row_fft),
                                                std::move(pack_kernel),
                                                std::move(col_fft),
                                                std::move(transpose_fwd),
                                                std::move(transpose_inv),
                                                std::move(row_fft_buf),
                                                std::move(temp1),
                                                std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_c2r_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const Maca2dPolicyScope policy_scope(maca_2d_single_policy_, request, batch, node->n0, node->n1);
  configure_single_transform_policies(request);
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t half_n1 = n1 / 2 + 1;
  const bool rc_eligible = n0 <= 256;

  // C2R is the reverse of R2C:
  // 1. Transpose (n0, half_n1) -> (half_n1, n0)
  // 2. Col IFFT along n0 (batch * half_n1)
  // 3. Transpose back (half_n1, n0) -> (n0, half_n1)
  // 4. Expand half-packed -> full Hermitian (n0, half_n1) -> (n0, n1)
  // 5. Row IFFT along n1 (batch * n0)
  // 6. Pack complex -> real

  // Build col C2C IFFT request (length=n0, batch=batch*half_n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * half_n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * half_n1;

  // Build row C2C IFFT request (length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Degenerate inverse real 2D: a unit axis reduces to a batched 1D C2R.
  if (n0 == 1) {
    row_request.batch = batch;
    row_request.input_shape = {batch, n1};
    return std::make_shared<CompiledRaw1DAs2DNode>(compile_raw_c2r_node(node->row_plan, row_request, batch),
                                                   batch);
  }

#if defined(FLAGFFT_BACKEND_NPU)
  const char *npu_aiv_fft64 = std::getenv("FLAGFFT_NPU_2D_ASCENDC_AIV");
  const bool use_npu_aiv_fft64_real =
      request.device_type == "npu" && request.origin_rank == 2 &&
      request.real_transform_kind == "c2r" && request.input_dtype == "complex64" &&
      request.output_dtype == "complex64" &&
      n0 == 64 && n1 == 64 && npu_aiv_fft64 != nullptr && std::string(npu_aiv_fft64) == "1" &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->row_plan) != nullptr &&
      std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan) != nullptr;
  if (use_npu_aiv_fft64_real) {
    const int32_t row_group_size = npu_aiv_fft64_real_row_group_size(
        std::getenv("FLAGFFT_NPU_2D_ASCENDC_ROW_GROUP"), batch);
    auto col_fft = make_npu_aiv_fft64_child(request, half_n1, 1);
    auto row_fft = make_npu_aiv_fft64_child(
        request, 1, row_group_size, NpuAivFFT64Mode::RealInverse);
    DeviceAllocation temp_half(
        static_cast<std::size_t>(batch * n0 * half_n1 * element_bytes));
    return std::make_shared<CompiledRaw2DC2RRCNode>(n0,
                                                    n1,
                                                    std::move(col_fft),
                                                    nullptr,
                                                    std::move(row_fft),
                                                    nullptr,
                                                    std::move(temp_half),
                                                    DeviceAllocation{});
  }
#endif

  // Symmetric MACA FP32 path.  The column inverse and compact-layout
  // transposes run first; the existing 1D C2R boundary node then consumes the
  // compact rows directly and writes real output.  Keep packed-real disabled
  // until its 2D row layout has a separate qualification.
  if (maca_2d_real_rows_enabled(request, batch, n0, n1, maca_2d_single_policy_) &&
      has_real_boundary_row_plan(node->row_plan)) {
    FFTRequest col_request = request;
    col_request.fft_length = n0;
    col_request.input_shape = {batch * half_n1, n0};
    col_request.input_strides = {n0, 1};
    col_request.requested_n = n0;
    col_request.batch = batch * half_n1;
    std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);

    std::shared_ptr<CompiledRawNode> row_c2r =
        compile_raw_c2r_node(node->row_plan, row_request, batch * n0, false);
    auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
    auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);
    const std::size_t compact_bytes = static_cast<std::size_t>(batch * n0 * half_n1 * element_bytes);
    DeviceAllocation temp1 = adaptor::Memory(compact_bytes);
    DeviceAllocation temp2 = adaptor::Memory(compact_bytes);

    return std::make_shared<CompiledRaw2DC2RRowNode>(n0,
                                                     n1,
                                                     std::move(col_fft),
                                                     std::move(row_c2r),
                                                     std::move(transpose_fwd),
                                                     std::move(transpose_inv),
                                                     std::move(temp1),
                                                     std::move(temp2));
  }

  // RC fast path for inverse real transforms: column IFFT first, then expand,
  // row IFFT, and real pack -- no transposes.
  auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node->col_plan);
  auto col_direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node->col_plan);
  auto col_four = std::dynamic_pointer_cast<FourStepPlanNode>(node->col_plan);
  std::shared_ptr<CompiledRawNode> rc_col_fft;
  if (rc_eligible && col_leaf != nullptr) {
    rc_col_fft = compile_raw_strided_leaf(*col_leaf, request, half_n1);
  } else if (rc_eligible && col_direct != nullptr && request.device_type != "npu") {
    rc_col_fft = compile_raw_strided_direct_dft(*col_direct, request, half_n1);
  } else if (rc_eligible && col_four != nullptr) {
    rc_col_fft = compile_raw_four_step_strided_node(*col_four, request, batch * half_n1, half_n1);
  }
  if (rc_col_fft != nullptr) {
    const bool use_npu_real_row = request.device_type == "npu" &&
                                  has_real_boundary_row_plan(node->row_plan) &&
                                  flag_or_default("FLAGFFT_NPU_2D_REAL_ROW", true);
    if (use_npu_real_row) {
      // After the inverse column transform, each compact row is already a
      // valid 1D C2R input. Consume it directly and avoid Hermitian expansion
      // followed by a full C2C inverse and real pack.
      std::shared_ptr<CompiledRawNode> row_c2r =
          compile_raw_c2r_node(node->row_plan, row_request, batch * n0, false);
      DeviceAllocation temp_half =
          adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
      return std::make_shared<CompiledRaw2DC2RRCNode>(n0,
                                                      n1,
                                                      std::move(rc_col_fft),
                                                      nullptr,
                                                      std::move(row_c2r),
                                                      nullptr,
                                                      std::move(temp_half),
                                                      DeviceAllocation {});
    }

    auto expand_kernel = compile_compact_to_hermitian_full_kernel(request, n1);
    std::shared_ptr<CompiledRawNode> row_fft = compile_raw_2d_rc_row(node->row_plan, row_request, batch * n0);
    auto pack_kernel = compile_complex_to_real_kernel(request, n1);
    DeviceAllocation temp_half =
        adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
    DeviceAllocation temp_full = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
    return std::make_shared<CompiledRaw2DC2RRCNode>(n0,
                                                    n1,
                                                    std::move(rc_col_fft),
                                                    std::move(expand_kernel),
                                                    std::move(row_fft),
                                                    std::move(pack_kernel),
                                                    std::move(temp_half),
                                                    std::move(temp_full));
  }

  // Compile kernels
  auto expand_kernel = compile_compact_to_hermitian_full_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  auto pack_kernel = compile_complex_to_real_kernel(request, n1);

  // C2R transposes: (n0, half_n1) <-> (half_n1, n0)
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);

  // Allocate buffers
  // temp1: transposed data (batch * half_n1 * n0 complex)
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp2: col IFFT output (batch * half_n1 * n0 complex)
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp3: expanded full Hermitian (batch * n0 * n1 complex) + row IFFT output
  DeviceAllocation temp3 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));

  return std::make_shared<CompiledRaw2DC2RNode>(n0,
                                                n1,
                                                std::move(expand_kernel),
                                                std::move(col_fft),
                                                std::move(row_fft),
                                                std::move(transpose_fwd),
                                                std::move(transpose_inv),
                                                std::move(pack_kernel),
                                                std::move(temp1),
                                                std::move(temp2),
                                                std::move(temp3));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_four_step_strided_node(
    const FourStepPlanNode &node, const FFTRequest &request, int64_t batch, int64_t outer_stride) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node.row_plan);
  auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(node.col_plan);
  if (row_leaf == nullptr || col_leaf == nullptr) {
    return nullptr;
  }

  DeviceAllocation twiddle = build_raw_four_step_twiddle(request, node.n1, node.n2);
  DeviceAllocation stage1 = adaptor::Memory(static_cast<std::size_t>(batch * node.length * element_bytes));
  return std::make_shared<CompiledRawFourStepStridedNode>(
      node.length,
      node.n1,
      node.n2,
      outer_stride,
      compile_four_step_row_strided_kernel(*row_leaf, request, node.n1, node.n2),
      build_raw_leaf_tables(*row_leaf, request),
      compile_four_step_col_strided_kernel(*col_leaf, request, node.n1, node.n2),
      build_raw_leaf_tables(*col_leaf, request),
      std::move(twiddle),
      std::move(stage1));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                               const FFTRequest &request,
                                                                               int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = node.length;
  const int64_t n1 = node.n1;
  const int64_t n2 = node.n2;

  std::shared_ptr<CompiledRawNode> row_child = compile_raw_node(node.row_plan, request, batch * n2);
  std::shared_ptr<CompiledRawNode> col_child = compile_raw_node(node.col_plan, request, batch * n1);

  DeviceAllocation twiddle = build_raw_four_step_twiddle(request, n1, n2);
  DeviceAllocation stage1 = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation stage2 = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));

  auto reshape_in_kernel = compile_reshape_pack_kernel(request, n1, n2);
  auto twiddle_reshape_kernel = compile_twiddle_reshape_pack_kernel(request, n2, n1);
  auto final_pack_kernel = compile_reshape_pack_kernel(request, n1, n2);

  return std::make_shared<CompiledRawFourStepGenericNode>(n,
                                                          n1,
                                                          n2,
                                                          std::move(row_child),
                                                          std::move(col_child),
                                                          std::move(reshape_in_kernel),
                                                          std::move(twiddle_reshape_kernel),
                                                          std::move(final_pack_kernel),
                                                          std::move(twiddle),
                                                          std::move(stage1),
                                                          std::move(stage2));
}

}  // namespace flagfft
