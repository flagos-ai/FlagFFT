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

#pragma once
#include <cstdlib>
#include <functional>

#include <string>

#include "adaptor/adaptor.h"
#include "flagfft/plan.hpp"

namespace flagfft {

// The 2D real-row path is intentionally a narrow MACA policy.  Keep
// the predicate inline so planner-only CPU tests can exercise the contract
// without constructing a device-backed TritonCompiler.
inline bool maca_2d_real_rows_enabled(const FFTRequest &request,
                                      int64_t batch,
                                      int64_t n0,
                                      int64_t n1,
                                      bool default_enabled = false) {
  const char *setting = std::getenv("FLAGFFT_MACA_2D_REAL_ROWS");
  const bool enabled = setting == nullptr ? default_enabled : std::string(setting) == "1";
  return enabled && request.device_type == "maca" &&
         request.raw_dim == 2 && request.batch == 1 && batch == 1 &&
         request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
         request.input_layout == "contiguous" && !request.requires_contiguous_copy && n0 > 256 &&
         n1 > 1 && (n1 % 2) == 0;
}

// In the RC schedule retain the existing batched policy for small mixed leaves.
// For the measured 209/221 pair the single-transform register policy reduces
// P8/256 threads to P4/128 threads and regresses the row transform. This does
// not disable graph/column choices at the root or alter other schedules.
inline bool maca_2d_rc_preserve_batched_row(const PlanNodePtr &node) {
  const auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node);
  if (four_step == nullptr) return false;
  const auto small_mixed_leaf = [](const PlanNodePtr &child) {
    const auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(child);
    return leaf != nullptr && leaf->length > 1 && leaf->length <= 256 &&
           (leaf->length & (leaf->length - 1)) != 0 && leaf->lanes == 1 &&
           leaf->factors.size() > 1;
  };
  return small_mixed_leaf(four_step->row_plan) && small_mixed_leaf(four_step->col_plan);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage);
int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit);
std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(const std::vector<int64_t> &radices,
                                                                       int64_t stage,
                                                                       int64_t lanes,
                                                                       const std::string &direction);
std::pair<std::vector<double>, std::vector<double>> build_stage_twiddles_d(
    const std::vector<int64_t> &radices, int64_t stage, int64_t lanes, const std::string &direction);
std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix,
                                                                   const std::string &direction);
std::pair<std::vector<double>, std::vector<double>> build_dft_matrix_d(int64_t radix,
                                                                       const std::string &direction);

enum class JitArgKind { DevicePtr, Int32, Int64 };

struct JitKernelArg {
  static JitKernelArg device(adaptor::DevicePtr value);
  static JitKernelArg i32(int32_t value);
  static JitKernelArg i64(int64_t value);

  JitArgKind kind = JitArgKind::DevicePtr;
  adaptor::DevicePtr device_ptr = 0;
  int32_t int32_value = 0;
  int64_t int64_value = 0;
};

struct JitKernel {
  std::string execution_description() const;
  std::string binary_dir;
  ~JitKernel();
  void compile();
  void launch(adaptor::StreamHandle stream,
              const std::vector<JitKernelArg> &kernel_args,
              int64_t grid_x,
              int64_t grid_y,
              int64_t grid_z);

  std::string kernel_name;
  std::string module_path;
  std::string signature;
  int64_t num_warps = 1;
  int64_t warp_size = 0;
  std::string profile_id;
  int64_t num_stages = 1;
  int64_t batch_per_block = 1;
  int64_t inner_pack = 1;
  int64_t rows_per_block = 1;
  int64_t butterflies_per_block = 128;
  int64_t grid_x_override = 0;
  bool tle_fused_twiddle = false;
  void *jit_function = nullptr;
  std::mutex mutex;
};

using DeviceAllocation = adaptor::Memory;

struct RawExecutionContext {
  const FFTRequest &request;
  adaptor::StreamHandle stream = nullptr;
  int64_t batch = 0;
  int64_t input_distance = 0;
  int64_t output_distance = 0;
};

struct CompiledRawNode {
  virtual ~CompiledRawNode() = default;
  virtual flagfftResult execute(adaptor::DevicePtr input,
                                adaptor::DevicePtr output,
                                const RawExecutionContext &context) const = 0;
  virtual std::string describe() const = 0;
};

struct CompiledRawLeafNode final : CompiledRawNode {
  CompiledRawLeafNode(int64_t length,
                      std::shared_ptr<JitKernel> kernel,
                      std::vector<DeviceAllocation> tables);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

struct CompiledRawStridedLeafNode final : CompiledRawNode {
  CompiledRawStridedLeafNode(int64_t length,
                             int64_t outer_stride,
                             std::shared_ptr<JitKernel> kernel,
                             std::vector<DeviceAllocation> tables);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t outer_stride;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

struct CompiledRawStridedDirectDftNode final : CompiledRawNode {
  CompiledRawStridedDirectDftNode(int64_t length,
                                  int64_t outer_stride,
                                  std::shared_ptr<JitKernel> kernel,
                                  std::vector<DeviceAllocation> tables);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t outer_stride;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

struct CompiledRawStockhamNode final : CompiledRawNode {
  CompiledRawStockhamNode(int64_t length,
                          std::vector<int64_t> factors,
                          std::vector<std::shared_ptr<JitKernel>> kernels,
                          DeviceAllocation twiddle,
                          DeviceAllocation first,
                          DeviceAllocation second);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t length;
  std::vector<int64_t> factors;
  std::vector<std::shared_ptr<JitKernel>> kernels;
  DeviceAllocation twiddle;
  DeviceAllocation first;
  DeviceAllocation second;
};

struct CompiledRawDirectDftNode final : CompiledRawNode {
  CompiledRawDirectDftNode(int64_t length,
                           std::shared_ptr<JitKernel> kernel,
                           std::vector<DeviceAllocation> tables,
                           DeviceAllocation input_copy);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  DeviceAllocation input_copy;
};

struct CompiledRawFourStepFusedNode final : CompiledRawNode {
  CompiledRawFourStepFusedNode(int64_t length,
                               int64_t n1,
                               int64_t n2,
                               std::shared_ptr<JitKernel> row_kernel,
                               std::vector<DeviceAllocation> row_tables,
                               std::shared_ptr<JitKernel> col_kernel,
                               std::vector<DeviceAllocation> col_tables,
                               DeviceAllocation twiddle,
                               DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation stage1;
};

struct CompiledRawBluesteinNode final : CompiledRawNode {
  CompiledRawBluesteinNode(int64_t length,
                           int64_t conv_length,
                           std::shared_ptr<CompiledRawNode> fft,
                           std::shared_ptr<JitKernel> prepare_kernel,
                           std::shared_ptr<JitKernel> pointwise_kernel,
                           std::shared_ptr<JitKernel> finalize_kernel,
                           DeviceAllocation chirp,
                           DeviceAllocation b_time,
                           DeviceAllocation a_buf,
                           DeviceAllocation work_buf,
                           DeviceAllocation b_fft_buf,
                           int64_t batch_chunk);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  int64_t batch_chunk;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> prepare_kernel;
  std::shared_ptr<JitKernel> pointwise_kernel;
  std::shared_ptr<JitKernel> finalize_kernel;
  DeviceAllocation chirp;
  DeviceAllocation b_time;
  DeviceAllocation a_buf;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
};

struct CompiledRawBluesteinLeafNode final : CompiledRawNode {
  CompiledRawBluesteinLeafNode(int64_t length,
                               int64_t conv_length,
                               std::shared_ptr<CompiledRawNode> fft,
                               std::shared_ptr<JitKernel> prepare_kernel,
                               std::shared_ptr<JitKernel> finish_kernel,
                               std::vector<DeviceAllocation> tables,
                               DeviceAllocation chirp,
                               DeviceAllocation b_time,
                               DeviceAllocation work_buf,
                               DeviceAllocation b_fft_buf);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> prepare_kernel;
  std::shared_ptr<JitKernel> finish_kernel;
  std::vector<DeviceAllocation> tables;
  DeviceAllocation chirp;
  DeviceAllocation b_time;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
};

struct CompiledRawBluesteinFullLeafNode final : CompiledRawNode {
  CompiledRawBluesteinFullLeafNode(int64_t length,
                                   int64_t conv_length,
                                   std::shared_ptr<CompiledRawNode> fft,
                                   std::shared_ptr<JitKernel> kernel,
                                   std::vector<DeviceAllocation> tables,
                                   DeviceAllocation chirp,
                                   DeviceAllocation b_time,
                                   DeviceAllocation b_fft_buf);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  DeviceAllocation chirp;
  DeviceAllocation b_time;
  mutable DeviceAllocation b_fft_buf;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
};

struct CompiledRawBluesteinFourStepNode final : CompiledRawNode {
  CompiledRawBluesteinFourStepNode(int64_t length,
                                   int64_t conv_length,
                                   int64_t n1,
                                   int64_t n2,
                                   std::shared_ptr<CompiledRawNode> fft,
                                   std::shared_ptr<JitKernel> prepare_row_kernel,
                                   std::shared_ptr<JitKernel> first_col_kernel,
                                   std::shared_ptr<JitKernel> pointwise_row_kernel,
                                   std::shared_ptr<JitKernel> finish_col_kernel,
                                   std::vector<DeviceAllocation> row_tables,
                                   std::vector<DeviceAllocation> col_tables,
                                   DeviceAllocation twiddle,
                                   DeviceAllocation chirp,
                                   DeviceAllocation b_time,
                                   DeviceAllocation stage1,
                                   DeviceAllocation work_buf,
                                   DeviceAllocation b_fft_buf);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> prepare_row_kernel;
  std::shared_ptr<JitKernel> first_col_kernel;
  std::shared_ptr<JitKernel> pointwise_row_kernel;
  std::shared_ptr<JitKernel> finish_col_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation chirp;
  DeviceAllocation b_time;
  DeviceAllocation stage1;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
};

struct CompiledRawRaderNode final : CompiledRawNode {
  CompiledRawRaderNode(int64_t length,
                       int64_t conv_length,
                       std::shared_ptr<CompiledRawNode> fft,
                       std::shared_ptr<JitKernel> prepare_kernel,
                       std::shared_ptr<JitKernel> pointwise_kernel,
                       std::shared_ptr<JitKernel> finalize_kernel,
                       DeviceAllocation idx,
                       DeviceAllocation b_time,
                       DeviceAllocation a_buf,
                       DeviceAllocation work_buf,
                       DeviceAllocation b_fft_buf,
                       DeviceAllocation input_copy);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> prepare_kernel;
  std::shared_ptr<JitKernel> pointwise_kernel;
  std::shared_ptr<JitKernel> finalize_kernel;
  DeviceAllocation idx;
  DeviceAllocation b_time;
  DeviceAllocation a_buf;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
  DeviceAllocation input_copy;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
};

struct CompiledRawFourStepGenericNode final : CompiledRawNode {
  CompiledRawFourStepGenericNode(int64_t length,
                                 int64_t n1,
                                 int64_t n2,
                                 std::shared_ptr<CompiledRawNode> row_child,
                                 std::shared_ptr<CompiledRawNode> col_child,
                                 std::shared_ptr<JitKernel> reshape_in_kernel,
                                 std::shared_ptr<JitKernel> twiddle_reshape_kernel,
                                 std::shared_ptr<JitKernel> final_pack_kernel,
                                 DeviceAllocation twiddle,
                                 DeviceAllocation stage1,
                                 DeviceAllocation stage2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<CompiledRawNode> row_child;
  std::shared_ptr<CompiledRawNode> col_child;
  std::shared_ptr<JitKernel> reshape_in_kernel;
  std::shared_ptr<JitKernel> twiddle_reshape_kernel;
  std::shared_ptr<JitKernel> final_pack_kernel;
  DeviceAllocation twiddle;
  DeviceAllocation stage1;
  DeviceAllocation stage2;
};

struct CompiledRawR2CNode final : CompiledRawNode {
  CompiledRawR2CNode(int64_t length,
                     std::shared_ptr<JitKernel> expand_kernel,
                     std::shared_ptr<CompiledRawNode> fft,
                     std::shared_ptr<JitKernel> pack_kernel,
                     DeviceAllocation complex_input,
                     DeviceAllocation full_output);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> pack_kernel;
  DeviceAllocation complex_input;
  DeviceAllocation full_output;
};

struct CompiledRawPackedR2CNode final : CompiledRawNode {
  CompiledRawPackedR2CNode(int64_t length,
                           std::shared_ptr<CompiledRawNode> fft,
                           std::shared_ptr<JitKernel> postprocess_kernel,
                           DeviceAllocation twiddle,
                           DeviceAllocation packed_output,
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> postprocess_kernel;
  DeviceAllocation twiddle;
  DeviceAllocation packed_output;
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
};

struct CompiledRawR2CLeafNode final : CompiledRawNode {
  CompiledRawR2CLeafNode(int64_t length,
                         std::shared_ptr<JitKernel> kernel,
                         std::vector<DeviceAllocation> tables);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

struct CompiledRawR2CFourStepHalfOutNode final : CompiledRawNode {
  CompiledRawR2CFourStepHalfOutNode(int64_t length,
                                    int64_t n1,
                                    int64_t n2,
                                    std::shared_ptr<JitKernel> expand_kernel,
                                    std::shared_ptr<JitKernel> row_kernel,
                                    std::vector<DeviceAllocation> row_tables,
                                    std::shared_ptr<JitKernel> col_kernel,
                                    std::vector<DeviceAllocation> col_tables,
                                    DeviceAllocation twiddle,
                                    DeviceAllocation complex_input,
                                    DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation complex_input;
  DeviceAllocation stage1;
};

struct CompiledRawR2CFourStepRealInHalfOutNode final : CompiledRawNode {
  CompiledRawR2CFourStepRealInHalfOutNode(int64_t length,
                                          int64_t n1,
                                          int64_t n2,
                                          std::shared_ptr<JitKernel> row_kernel,
                                          std::vector<DeviceAllocation> row_tables,
                                          std::shared_ptr<JitKernel> col_kernel,
                                          std::vector<DeviceAllocation> col_tables,
                                          DeviceAllocation twiddle,
                                          DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation stage1;
};

struct CompiledRawC2RNode final : CompiledRawNode {
  CompiledRawC2RNode(int64_t length,
                     std::shared_ptr<JitKernel> expand_kernel,
                     std::shared_ptr<CompiledRawNode> fft,
                     std::shared_ptr<JitKernel> pack_kernel,
                     DeviceAllocation full_input,
                     DeviceAllocation full_output);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> pack_kernel;
  DeviceAllocation full_input;
  DeviceAllocation full_output;
};

struct CompiledRawPackedC2RNode final : CompiledRawNode {
  CompiledRawPackedC2RNode(int64_t length,
                           std::shared_ptr<JitKernel> preprocess_kernel,
                           std::shared_ptr<CompiledRawNode> fft,
                           DeviceAllocation twiddle,
                           DeviceAllocation packed_input,
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> preprocess_kernel;
  std::shared_ptr<CompiledRawNode> fft;
  DeviceAllocation twiddle;
  DeviceAllocation packed_input;
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
};

struct CompiledRawC2RLeafNode final : CompiledRawNode {
  CompiledRawC2RLeafNode(int64_t length,
                         std::shared_ptr<JitKernel> kernel,
                         std::vector<DeviceAllocation> tables);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

struct CompiledRawC2RFourStepRealOutNode final : CompiledRawNode {
  CompiledRawC2RFourStepRealOutNode(int64_t length,
                                    int64_t n1,
                                    int64_t n2,
                                    std::shared_ptr<JitKernel> expand_kernel,
                                    std::shared_ptr<JitKernel> row_kernel,
                                    std::vector<DeviceAllocation> row_tables,
                                    std::shared_ptr<JitKernel> col_kernel,
                                    std::vector<DeviceAllocation> col_tables,
                                    DeviceAllocation twiddle,
                                    DeviceAllocation full_input,
                                    DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation full_input;
  DeviceAllocation stage1;
};

struct CompiledRawC2RFourStepCompactInRealOutNode final : CompiledRawNode {
  CompiledRawC2RFourStepCompactInRealOutNode(int64_t length,
                                             int64_t n1,
                                             int64_t n2,
                                             std::shared_ptr<JitKernel> row_kernel,
                                             std::vector<DeviceAllocation> row_tables,
                                             std::shared_ptr<JitKernel> col_kernel,
                                             std::vector<DeviceAllocation> col_tables,
                                             DeviceAllocation twiddle,
                                             DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation stage1;
};

struct CompiledRawFourStepStridedNode final : CompiledRawNode {
  CompiledRawFourStepStridedNode(int64_t length,
                                 int64_t n1,
                                 int64_t n2,
                                 int64_t outer_stride,
                                 std::shared_ptr<JitKernel> row_kernel,
                                 std::vector<DeviceAllocation> row_tables,
                                 std::shared_ptr<JitKernel> col_kernel,
                                 std::vector<DeviceAllocation> col_tables,
                                 DeviceAllocation twiddle,
                                 DeviceAllocation stage1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t n1;
  int64_t n2;
  int64_t outer_stride;
  std::shared_ptr<JitKernel> row_kernel;
  std::vector<DeviceAllocation> row_tables;
  std::shared_ptr<JitKernel> col_kernel;
  std::vector<DeviceAllocation> col_tables;
  DeviceAllocation twiddle;
  DeviceAllocation stage1;
};

struct CompiledRaw2DNode final : CompiledRawNode {
  CompiledRaw2DNode(int64_t n0,
                    int64_t n1,
                    std::shared_ptr<CompiledRawNode> row_fft,
                    std::shared_ptr<CompiledRawNode> col_fft,
                    std::shared_ptr<JitKernel> transpose_fwd,
                    std::shared_ptr<JitKernel> transpose_inv,
                    DeviceAllocation temp1,
                    DeviceAllocation temp2,
                    bool enable_graph = true);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<JitKernel> transpose_fwd;
  std::shared_ptr<JitKernel> transpose_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;

  // Optional CUDA-graph replay cache for repeated executions with the same
  // input/output pointers (falls back to direct launches on failure).
  mutable std::unique_ptr<adaptor::CudaGraph> graph_;
  mutable adaptor::DevicePtr graph_in_ = 0;
  mutable adaptor::DevicePtr graph_out_ = 0;
  const bool graph_enabled_;
  mutable bool graph_failed_ = false;
};

struct CompiledRaw2DRCNode final : CompiledRawNode {
  CompiledRaw2DRCNode(int64_t n0,
                      int64_t n1,
                      std::shared_ptr<CompiledRawNode> row_fft,
                      std::shared_ptr<CompiledRawNode> col_fft,
                      DeviceAllocation temp1,
                      bool enable_graph = true);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<CompiledRawNode> col_fft;
  DeviceAllocation temp1;

  mutable std::unique_ptr<adaptor::CudaGraph> graph_;
  mutable adaptor::DevicePtr graph_in_ = 0;
  mutable adaptor::DevicePtr graph_out_ = 0;
  const bool graph_enabled_;
  mutable bool graph_failed_ = false;
};

struct CompiledRaw1DAs2DNode final : CompiledRawNode {
  CompiledRaw1DAs2DNode(std::shared_ptr<CompiledRawNode> fft, int64_t batch);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  std::shared_ptr<CompiledRawNode> fft;
  int64_t batch;
};

// Dense 2D real transforms with a real boundary on the innermost axis.  The
// row child writes compact rows into temp1; the two compact scratch buffers
// are then reused around the column FFT.  Keeping the row child separate lets
// the compiler select the existing leaf/FourStep real boundary implementations
// without introducing a new kernel family.
struct CompiledRaw2DR2CRowNode final : CompiledRawNode {
  CompiledRaw2DR2CRowNode(int64_t n0,
                          int64_t n1,
                          std::shared_ptr<CompiledRawNode> row_r2c,
                          std::shared_ptr<CompiledRawNode> col_fft,
                          std::shared_ptr<JitKernel> transpose_fwd,
                          std::shared_ptr<JitKernel> transpose_inv,
                          DeviceAllocation temp1,
                          DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> row_r2c;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<JitKernel> transpose_fwd;
  std::shared_ptr<JitKernel> transpose_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

struct CompiledRaw2DC2RRowNode final : CompiledRawNode {
  CompiledRaw2DC2RRowNode(int64_t n0,
                          int64_t n1,
                          std::shared_ptr<CompiledRawNode> col_fft,
                          std::shared_ptr<CompiledRawNode> row_c2r,
                          std::shared_ptr<JitKernel> transpose_fwd,
                          std::shared_ptr<JitKernel> transpose_inv,
                          DeviceAllocation temp1,
                          DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<CompiledRawNode> row_c2r;
  std::shared_ptr<JitKernel> transpose_fwd;
  std::shared_ptr<JitKernel> transpose_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

struct CompiledRaw2DR2CNode final : CompiledRawNode {
  CompiledRaw2DR2CNode(int64_t n0,
                       int64_t n1,
                       std::shared_ptr<JitKernel> expand_kernel,
                       std::shared_ptr<CompiledRawNode> row_fft,
                       std::shared_ptr<JitKernel> pack_kernel,
                       std::shared_ptr<CompiledRawNode> col_fft,
                       std::shared_ptr<JitKernel> transpose_fwd,
                       std::shared_ptr<JitKernel> transpose_inv,
                       DeviceAllocation row_fft_buf,
                       DeviceAllocation temp1,
                       DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<JitKernel> pack_kernel;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<JitKernel> transpose_fwd;
  std::shared_ptr<JitKernel> transpose_inv;
  DeviceAllocation row_fft_buf;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

struct CompiledRaw2DR2CRCNode final : CompiledRawNode {
  CompiledRaw2DR2CRCNode(int64_t n0,
                         int64_t n1,
                         std::shared_ptr<JitKernel> expand_kernel,
                         std::shared_ptr<CompiledRawNode> row_fft,
                         std::shared_ptr<JitKernel> pack_kernel,
                         std::shared_ptr<CompiledRawNode> col_fft,
                         DeviceAllocation row_fft_buf);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<JitKernel> pack_kernel;
  std::shared_ptr<CompiledRawNode> col_fft;
  DeviceAllocation row_fft_buf;
};

struct CompiledRaw2DC2RNode final : CompiledRawNode {
  CompiledRaw2DC2RNode(int64_t n0,
                       int64_t n1,
                       std::shared_ptr<JitKernel> expand_kernel,
                       std::shared_ptr<CompiledRawNode> col_fft,
                       std::shared_ptr<CompiledRawNode> row_fft,
                       std::shared_ptr<JitKernel> transpose_fwd,
                       std::shared_ptr<JitKernel> transpose_inv,
                       std::shared_ptr<JitKernel> pack_kernel,
                       DeviceAllocation temp1,
                       DeviceAllocation temp2,
                       DeviceAllocation temp3);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<JitKernel> transpose_fwd;
  std::shared_ptr<JitKernel> transpose_inv;
  std::shared_ptr<JitKernel> pack_kernel;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  DeviceAllocation temp3;
};

struct CompiledRaw2DC2RRCNode final : CompiledRawNode {
  CompiledRaw2DC2RRCNode(int64_t n0,
                         int64_t n1,
                         std::shared_ptr<CompiledRawNode> col_fft,
                         std::shared_ptr<JitKernel> expand_kernel,
                         std::shared_ptr<CompiledRawNode> row_fft,
                         std::shared_ptr<JitKernel> pack_kernel,
                         DeviceAllocation temp_half,
                         DeviceAllocation temp_full);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<JitKernel> pack_kernel;
  DeviceAllocation temp_half;
  DeviceAllocation temp_full;
};

struct CompiledRaw3DNode final : CompiledRawNode {
  CompiledRaw3DNode(int64_t n0,
                    int64_t n1,
                    int64_t n2,
                    std::shared_ptr<CompiledRawNode> n2_fft,
                    std::shared_ptr<CompiledRawNode> n1_fft,
                    std::shared_ptr<CompiledRawNode> n0_fft,
                    std::shared_ptr<JitKernel> perm_021_fwd,
                    std::shared_ptr<JitKernel> perm_210_fwd,
                    std::shared_ptr<JitKernel> perm_201_fwd,
                    std::shared_ptr<JitKernel> perm_120_inv,
                    std::shared_ptr<JitKernel> perm_210_inv,
                    std::shared_ptr<JitKernel> perm_021_inv,
                    DeviceAllocation temp1,
                    DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<CompiledRawNode> n2_fft;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<JitKernel> perm_021_fwd;
  std::shared_ptr<JitKernel> perm_210_fwd;
  std::shared_ptr<JitKernel> perm_201_fwd;
  std::shared_ptr<JitKernel> perm_120_inv;
  std::shared_ptr<JitKernel> perm_210_inv;
  std::shared_ptr<JitKernel> perm_021_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

// 3D C2C/Z2Z that runs the n1 and n0 axes as strided leaves on the natural
// layout instead of permuting the cube between passes: three launches and no
// full-cube transpose traffic.  Used when both non-contiguous axis plans are
// plain leaves.
struct CompiledRaw3DStridedNode final : CompiledRawNode {
  CompiledRaw3DStridedNode(int64_t n0,
                           int64_t n1,
                           int64_t n2,
                           std::shared_ptr<CompiledRawNode> n2_fft,
                           std::shared_ptr<CompiledRawNode> n1_fft,
                           std::shared_ptr<CompiledRawNode> n0_fft,
                           DeviceAllocation temp1,
                           DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<CompiledRawNode> n2_fft;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

struct CompiledRaw3DR2CNode final : CompiledRawNode {
  CompiledRaw3DR2CNode(int64_t n0,
                       int64_t n1,
                       int64_t n2,
                       std::shared_ptr<JitKernel> expand_kernel,
                       std::shared_ptr<CompiledRawNode> n2_fft,
                       std::shared_ptr<JitKernel> pack_kernel,
                       std::shared_ptr<CompiledRawNode> n1_fft,
                       std::shared_ptr<CompiledRawNode> n0_fft,
                       std::shared_ptr<JitKernel> perm_021,
                       std::shared_ptr<JitKernel> perm_210,
                       std::shared_ptr<JitKernel> perm_201,
                       DeviceAllocation row_fft_buf,
                       DeviceAllocation temp1,
                       DeviceAllocation temp2);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> n2_fft;
  std::shared_ptr<JitKernel> pack_kernel;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<JitKernel> perm_021;
  std::shared_ptr<JitKernel> perm_210;
  std::shared_ptr<JitKernel> perm_201;
  DeviceAllocation row_fft_buf;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

struct CompiledRaw3DC2RNode final : CompiledRawNode {
  CompiledRaw3DC2RNode(int64_t n0,
                       int64_t n1,
                       int64_t n2,
                       std::shared_ptr<JitKernel> perm_120,
                       std::shared_ptr<JitKernel> perm_210,
                       std::shared_ptr<JitKernel> perm_021,
                       std::shared_ptr<CompiledRawNode> n0_fft,
                       std::shared_ptr<CompiledRawNode> n1_fft,
                       std::shared_ptr<JitKernel> expand_kernel,
                       std::shared_ptr<CompiledRawNode> n2_fft,
                       std::shared_ptr<JitKernel> pack_kernel,
                       DeviceAllocation temp1,
                       DeviceAllocation temp2,
                       DeviceAllocation full_buf);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> perm_120;
  std::shared_ptr<JitKernel> perm_210;
  std::shared_ptr<JitKernel> perm_021;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<JitKernel> expand_kernel;
  std::shared_ptr<CompiledRawNode> n2_fft;
  std::shared_ptr<JitKernel> pack_kernel;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  DeviceAllocation full_buf;
};

class TritonCompiler {
 public:
  std::shared_ptr<CompiledRawNode> compile_raw_node(const PlanNodePtr &node,
                                                    const FFTRequest &request,
                                                    int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_r2c_node(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch,
                                                        bool allow_packed = true);
  std::shared_ptr<CompiledRawNode> compile_raw_c2r_node(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch,
                                                        bool allow_packed = true);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                       const FFTRequest &request,
                                                       int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_r2c_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_c2r_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_3d_node(const std::shared_ptr<ThreeDimPlanNode> &node,
                                                         const FFTRequest &request,
                                                         int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_3d_r2c_node(const std::shared_ptr<ThreeDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_3d_c2r_node(const std::shared_ptr<ThreeDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  static void clear_kernel_cache();

 private:
  void configure_single_transform_policies(const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_rc_row(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch);

  bool maca_1d_single_policy_ = false;
  bool maca_2d_single_policy_ = false;
  bool ix_ct_single_policy_ = false;
  int ix_ct_single_tle_policy_ = 0;
  std::string maca_tail_policy_ = "off";

  std::shared_ptr<CompiledRawNode> compile_raw_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_raw_permuted_store_leaf(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request,
                                                                   int64_t perm_span,
                                                                   const std::string &perm_form);
  std::shared_ptr<CompiledRawNode> compile_raw_strided_leaf(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t outer_stride);
  std::shared_ptr<CompiledRawNode> compile_raw_strided_direct_dft(const DirectDFTPlanNode &node,
                                                                  const FFTRequest &request,
                                                                  int64_t outer_stride);
  std::shared_ptr<CompiledRawNode> compile_raw_direct_dft(const DirectDFTPlanNode &node,
                                                          const FFTRequest &request,
                                                          int64_t batch);
  std::shared_ptr<JitKernel> compile_direct_dft_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<CompiledRawNode> compile_raw_real_direct_dft(const FFTRequest &request,
                                                            bool inverse);
  std::shared_ptr<CompiledRawNode> compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                 const FFTRequest &request,
                                                                 int64_t batch);
  std::shared_ptr<JitKernel> compile_leaf_r2c_kernel(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_leaf_c2r_kernel(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                          const FFTRequest &request,
                                                          int64_t n1,
                                                          int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_row_strided_kernel(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request,
                                                                  int64_t n1,
                                                                  int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_real_row_kernel(const LeafPlanNode &leaf,
                                                               const FFTRequest &request,
                                                               int64_t n1,
                                                               int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_hermitian_row_kernel(const LeafPlanNode &leaf,
                                                                    const FFTRequest &request,
                                                                    int64_t n1,
                                                                    int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                          const FFTRequest &request,
                                                          int64_t n1,
                                                          int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_col_strided_kernel(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request,
                                                                  int64_t n1,
                                                                  int64_t n2);
  std::shared_ptr<CompiledRawNode> compile_raw_four_step_strided_node(const FourStepPlanNode &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch,
                                                                      int64_t outer_stride);
  std::shared_ptr<JitKernel> compile_four_step_r2c_col_kernel(const LeafPlanNode &leaf,
                                                              const FFTRequest &request,
                                                              int64_t n1,
                                                              int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_c2r_col_kernel(const LeafPlanNode &leaf,
                                                              const FFTRequest &request,
                                                              int64_t n1,
                                                              int64_t n2);
  std::shared_ptr<JitKernel> compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                              int64_t n,
                                                              int64_t m);
  std::shared_ptr<JitKernel> compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                int64_t n,
                                                                int64_t m);
  std::shared_ptr<JitKernel> compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                               int64_t n,
                                                               int64_t m);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_kernel(const LeafPlanNode &leaf,
                                                           const FFTRequest &request,
                                                           int64_t n);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_prepare_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request,
                                                                   int64_t n);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_finish_kernel(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request,
                                                                  int64_t n);
  std::shared_ptr<JitKernel> compile_rader_prepare_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_pointwise_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_finalize_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_reshape_pack_kernel(const FFTRequest &request, int64_t n1, int64_t n2);
  std::shared_ptr<JitKernel> compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                 int64_t n1,
                                                                 int64_t n2);
  std::shared_ptr<JitKernel> compile_real_to_complex_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_half_pack_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_packed_postprocess_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_c2r_packed_preprocess_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_compact_to_hermitian_full_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_complex_to_real_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_tiled_transpose_kernel(const FFTRequest &request,
                                                            int64_t n0,
                                                            int64_t n1);
  std::shared_ptr<JitKernel> compile_transpose3d_kernel(
      const FFTRequest &request, int64_t n0, int64_t n1, int64_t n2, const std::string &order);
  std::shared_ptr<JitKernel> compile_kernel(const KernelKey &key) const;
  std::filesystem::path out_dir() const;
  std::string python_executable() const;
  std::string triton_jit_source_entrypoint() const;
};

std::string triton_target_for_request(const FFTRequest &request);
FFTRequest forward_child_request(const FFTRequest &request);
DeviceAllocation build_raw_four_step_twiddle(const FFTRequest &request, int64_t n1, int64_t n2);
DeviceAllocation build_raw_packed_real_twiddle(const FFTRequest &request, int64_t n);
DeviceAllocation build_raw_bluestein_chirp(const FFTRequest &request, int64_t n, bool inverse_sign);
DeviceAllocation build_raw_bluestein_b(const FFTRequest &request, int64_t n, int64_t m);
DeviceAllocation build_raw_rader_idx_table(const std::vector<int64_t> &idx);
DeviceAllocation build_raw_rader_conv_kernel(const FFTRequest &request,
                                             int64_t n,
                                             const std::vector<int64_t> &idx);
std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request);
bool ix_ct_single_policy_enabled(const FFTRequest &request);
bool ix_packed_real_policy_enabled(const FFTRequest &request);
int ix_ct_single_tle_policy(const FFTRequest &request);
std::vector<DeviceAllocation> build_raw_direct_dft_tables(int64_t n, const FFTRequest &request);

}  // namespace flagfft
