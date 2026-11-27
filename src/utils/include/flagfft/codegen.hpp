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
#include <atomic>
#include <cstdlib>
#include <functional>

#include <string>

#include "adaptor/adaptor.h"
#include "flagfft/plan.hpp"
#include "flagfft/performance_defaults.hpp"

namespace flagfft {

// The host coefficient order must match the portable vector exchange order.
inline bool npu_portable_leaf_enabled(const FFTRequest &request) {
  const BackendPerformanceScope policy(request.device_type, request.device_arch, request.input_dtype);
  const auto enabled = [](const char *name) {
    const char *value = performance_env(name);
    return value != nullptr && std::string(value) == "1";
  };
  const char *leaf3d = performance_env("FLAGFFT_NPU_3D_LEAF");
  return request.device_type == "npu" &&
      (request.origin_rank == 2 ||
       (request.origin_rank == 3 && (leaf3d == nullptr || std::string(leaf3d) == "1")) ||
       enabled("FLAGFFT_NPU_FOURSTEP_LEAF") ||
       performance_env("FLAGFFT_NPU_PORTABLE_LEAF_MAX") != nullptr);
}

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
  int64_t column_block = 256;
  int64_t butterflies_per_block = 128;
  int64_t grid_x_override = 0;
  int64_t grid_y_override = 0;
  // Nonzero only for the private NPU transpose with a byte-offset table ABI.
  int64_t transpose_tile_size = 0;
  bool tle_fused_twiddle = false;
  bool npu_permutation_table = false;
  std::string npu_join_lowering = "auto";
  bool npu_auto_blockify = false;
  bool npu_single_buffer = false;
  void *jit_function = nullptr;
  std::shared_ptr<void> direct_launch;
  std::atomic<bool> compiled{false};
  std::mutex mutex;
};

using DeviceAllocation = adaptor::Memory;

// Rank-2 transposes use the generated Triton body. The NPU table variant keeps
// its byte-offset table in npu_indices and supplies its own launch grid.
struct CompiledRawTranspose2D {
  int64_t rows = 0;
  int64_t cols = 0;
  std::shared_ptr<JitKernel> triton_kernel;
  DeviceAllocation npu_indices;

  void launch(adaptor::StreamHandle stream,
              const std::vector<JitKernelArg> &kernel_args,
              int64_t grid_x,
              int64_t grid_y,
              int64_t grid_z) const;
  std::string execution_description() const;
};

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
  virtual bool is_single_kernel() const { return false; }
};

// A fused axis permutation is defined within one ND transform. Reuse its
// scratch and kernels serially rather than folding independent transforms
// into the permutation's row index.
struct CompiledRawPerBatchNode final : CompiledRawNode {
  CompiledRawPerBatchNode(std::shared_ptr<CompiledRawNode> inner,
                         int64_t input_elements, int64_t output_elements,
                         int64_t input_element_bytes, int64_t output_element_bytes);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  std::shared_ptr<CompiledRawNode> inner;
  int64_t input_elements, output_elements, input_element_bytes, output_element_bytes;
};

struct CompiledRawGraphNode final : CompiledRawNode {
  explicit CompiledRawGraphNode(std::shared_ptr<CompiledRawNode> inner);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  std::shared_ptr<CompiledRawNode> inner;
  mutable std::unique_ptr<adaptor::CudaGraph> graph;
  mutable adaptor::DevicePtr graph_input = 0;
  mutable adaptor::DevicePtr graph_output = 0;
  mutable adaptor::StreamHandle graph_stream = nullptr;
  mutable int64_t graph_batch = 0;
  mutable int64_t graph_input_distance = 0;
  mutable int64_t graph_output_distance = 0;
  mutable bool graph_failed = false;
};

struct Raw1DGraphState {
  std::unique_ptr<adaptor::CudaGraph> graph;
  adaptor::DevicePtr input = 0;
  adaptor::DevicePtr output = 0;
  int64_t batch = 0;
  int64_t input_distance = 0;
  int64_t output_distance = 0;
  bool failed = false;
};

struct CompiledRawLeafNode final : CompiledRawNode {
  CompiledRawLeafNode(int64_t length,
                      std::shared_ptr<JitKernel> kernel,
                      std::vector<DeviceAllocation> tables,
                      std::shared_ptr<CompiledRawNode> layout_fallback = nullptr,
                      int64_t fixed_batch = 1);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  std::shared_ptr<CompiledRawNode> layout_fallback;
  int64_t fixed_batch;
  mutable Raw1DGraphState graph_state;
};

// Frequency tiles share a read-only input; aliased calls retain the leaf FFT.
struct CompiledRawFrequencyDFTNode final : CompiledRawNode {
  CompiledRawFrequencyDFTNode(int64_t length, std::shared_ptr<JitKernel> kernel,
      DeviceAllocation weights, std::shared_ptr<CompiledRawNode> fallback);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }
  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  DeviceAllocation weights;
  std::shared_ptr<CompiledRawNode> fallback;
};

struct CompiledRawStridedLeafNode final : CompiledRawNode {
  CompiledRawStridedLeafNode(int64_t length,
                             int64_t outer_stride,
                             std::shared_ptr<JitKernel> kernel,
                             std::vector<DeviceAllocation> tables,
                             int64_t rows_per_transform = 0);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  int64_t outer_stride;
  int64_t rows_per_transform;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
};

enum class NpuAivFFT64Mode : int32_t {
  Complex = 0,
  RealForward = 1,
  RealInverse = 2,
};

using NpuAivFFT256Mode = NpuAivFFT64Mode;

using NpuAivFFTSmallMode = NpuAivFFT64Mode;

#if defined(FLAGFFT_BACKEND_NPU)
struct CompiledRawNpuAivFFT64Node final : CompiledRawNode {
  CompiledRawNpuAivFFT64Node(int64_t stride,
                             int64_t group_size,
                             std::shared_ptr<DeviceAllocation> indices,
                             std::shared_ptr<DeviceAllocation> twiddles,
                             NpuAivFFT64Mode mode = NpuAivFFT64Mode::Complex,
                             bool radix4_pair = false,
                             std::shared_ptr<JitKernel> replacement = nullptr);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t stride;
  int64_t group_size;
  NpuAivFFT64Mode mode;
  bool radix4_pair;
  std::shared_ptr<JitKernel> replacement;
  std::shared_ptr<DeviceAllocation> indices;
  std::shared_ptr<DeviceAllocation> twiddles;
};

struct CompiledRawNpuAivFFTSmallNode final : CompiledRawNode {
  CompiledRawNpuAivFFTSmallNode(int64_t length,
                                int64_t stride,
                                int64_t output_stride,
                                int64_t group_size,
                                std::shared_ptr<DeviceAllocation> indices,
                                std::shared_ptr<DeviceAllocation> twiddles,
                                NpuAivFFTSmallMode mode = NpuAivFFTSmallMode::Complex,
                                std::shared_ptr<JitKernel> replacement = nullptr);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t length;
  int64_t stride;
  int64_t output_stride;
  int64_t group_size;
  NpuAivFFTSmallMode mode;
  std::shared_ptr<JitKernel> replacement;
  std::shared_ptr<DeviceAllocation> indices;
  std::shared_ptr<DeviceAllocation> twiddles;
};

struct CompiledRawNpuAivFFTNode final : CompiledRawNode {
  CompiledRawNpuAivFFTNode(int64_t length,
                           std::shared_ptr<DeviceAllocation> indices,
                           std::shared_ptr<DeviceAllocation> twiddles,
                           int64_t group_size = 1,
                           bool radix4_pair = false,
                           std::shared_ptr<JitKernel> replacement = nullptr);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t length;
  int64_t group_size;
  bool radix4_pair;
  std::shared_ptr<JitKernel> replacement;
  std::shared_ptr<DeviceAllocation> indices;
  std::shared_ptr<DeviceAllocation> twiddles;
};

struct CompiledRawNpuAivFFT256Node final : CompiledRawNode {
  CompiledRawNpuAivFFT256Node(std::shared_ptr<DeviceAllocation> indices,
                              std::shared_ptr<DeviceAllocation> twiddles,
                              int64_t group_size = 1,
                              bool pair_mode = false,
                              bool transposed_store = false,
                              bool radix4_mode = false,
                              int64_t transposed_output_row_stride = 0,
                              NpuAivFFT256Mode mode = NpuAivFFT256Mode::Complex,
                              std::shared_ptr<JitKernel> replacement = nullptr);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  std::shared_ptr<DeviceAllocation> indices;
  std::shared_ptr<DeviceAllocation> twiddles;
  int64_t group_size;
  bool pair_mode;
  bool transposed_store;
  bool radix4_mode;
  int64_t transposed_output_row_stride;
  NpuAivFFT256Mode mode;
  std::shared_ptr<JitKernel> replacement;
};
#endif

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
  struct AivStageTables {
    bool enabled = false;
    DeviceAllocation twiddles;
    DeviceAllocation roots;
    DeviceAllocation store_indices;
    std::shared_ptr<JitKernel> tle_kernel;
    DeviceAllocation tle_twiddles;
    DeviceAllocation tle_store;
    int64_t tle_tiles = 0;
  };

  CompiledRawStockhamNode(int64_t length,
                          std::vector<int64_t> factors,
                          std::vector<std::shared_ptr<JitKernel>> kernels,
                          std::vector<AivStageTables> aiv_stages,
                          DeviceAllocation aiv_indices,
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
  std::vector<AivStageTables> aiv_stages;
  DeviceAllocation aiv_indices;
  DeviceAllocation twiddle;
  DeviceAllocation first;
  DeviceAllocation second;
};

#if defined(FLAGFFT_BACKEND_NPU)

#endif

struct CompiledRawDirectDftNode final : CompiledRawNode {
  CompiledRawDirectDftNode(int64_t length,
                           std::shared_ptr<JitKernel> kernel,
                           std::vector<DeviceAllocation> tables,
                           DeviceAllocation input_copy,
                           int real_mode = 0, int64_t scalar_bytes = 0,
                           std::string real_kind = "",
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  DeviceAllocation input_copy;
  int real_mode;
  int64_t scalar_bytes;
  std::string real_kind;
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
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
                           int64_t batch_chunk,
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback);
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
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
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
                                   DeviceAllocation b_fft_buf,
                                   std::string real_kind = "",
                                   std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
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
  std::string real_kind;
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
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
                                   DeviceAllocation b_fft_buf,
                                   std::string real_kind = "",
                                   std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {},
                                   int64_t batch_chunk = 0);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  void ensure_b_fft(const RawExecutionContext &context) const;

  int64_t length;
  int64_t conv_length;
  int64_t batch_chunk;
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
  std::string real_kind;
  std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
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
                       DeviceAllocation input_copy,
                       std::shared_ptr<JitKernel> fused_leaf_kernel = nullptr,
                       std::vector<DeviceAllocation> fused_leaf_tables = {},
                       std::shared_ptr<JitKernel> boundary_prepare_kernel = nullptr,
                       std::shared_ptr<JitKernel> boundary_finish_kernel = nullptr,
                       std::vector<DeviceAllocation> boundary_tables = {},
                       std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
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
  std::shared_ptr<JitKernel> fused_leaf_kernel;
  std::vector<DeviceAllocation> fused_leaf_tables;
  std::shared_ptr<JitKernel> boundary_prepare_kernel;
  std::shared_ptr<JitKernel> boundary_finish_kernel;
  std::vector<DeviceAllocation> boundary_tables;
  DeviceAllocation idx;
  DeviceAllocation b_time;
  DeviceAllocation a_buf;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
  DeviceAllocation input_copy;
  mutable bool b_fft_ready = false;
  mutable std::mutex b_fft_mutex;
  const std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
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
                                 DeviceAllocation stage2,
                                 DeviceAllocation pack_indices = DeviceAllocation {});
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
  DeviceAllocation pack_indices;
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

struct CompiledRawRealPairNode final : CompiledRawNode {
  CompiledRawRealPairNode(int64_t n, int64_t batch, bool inverse,
      FFTRequest complex_request, std::shared_ptr<CompiledRawNode> fft,
      std::shared_ptr<JitKernel> pack, std::shared_ptr<JitKernel> unpack,
      DeviceAllocation work_input, DeviceAllocation work_output);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t n;
  int64_t batch;
  bool inverse;
  FFTRequest complex_request;
  std::shared_ptr<CompiledRawNode> fft;
  std::shared_ptr<JitKernel> pack;
  std::shared_ptr<JitKernel> unpack;
  DeviceAllocation work_input;
  DeviceAllocation work_output;
};

struct CompiledRawPackedR2CNode final : CompiledRawNode {
  CompiledRawPackedR2CNode(int64_t length,
                           std::shared_ptr<CompiledRawNode> fft,
                           std::shared_ptr<JitKernel> postprocess_kernel,
                           DeviceAllocation twiddle,
                           DeviceAllocation packed_output,
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {},
                           std::shared_ptr<CompiledRawNode> aiv_postprocess = nullptr);
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
  std::shared_ptr<CompiledRawNode> aiv_postprocess;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
};

struct CompiledRawR2CLeafNode final : CompiledRawNode {
  CompiledRawR2CLeafNode(int64_t length,
                         std::shared_ptr<JitKernel> kernel,
                         std::vector<DeviceAllocation> tables,
                         DeviceAllocation twiddle = {},
                         int64_t perm_span = 0);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  DeviceAllocation twiddle;
  int64_t perm_span;
  mutable Raw1DGraphState graph_state;
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
                           std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {},
                           std::shared_ptr<CompiledRawNode> aiv_preprocess = nullptr);
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
  std::shared_ptr<CompiledRawNode> aiv_preprocess;
  mutable std::shared_ptr<CompiledRawNode> layout_fallback;
  mutable std::mutex layout_mutex;
};

// Experimental half-spectrum C2R path that folds packed preprocessing into
// the input loads of a dense half-length complex leaf.
struct CompiledRawPackedC2RFusedLeafNode final : CompiledRawNode {
  CompiledRawPackedC2RFusedLeafNode(
      int64_t length,
      std::shared_ptr<JitKernel> kernel,
      DeviceAllocation twiddle,
      std::vector<DeviceAllocation> tables,
      std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback = {});
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  DeviceAllocation twiddle;
  std::vector<DeviceAllocation> tables;
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
  bool is_single_kernel() const override { return true; }

  int64_t length;
  std::shared_ptr<JitKernel> kernel;
  std::vector<DeviceAllocation> tables;
  mutable Raw1DGraphState graph_state;
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
                    std::shared_ptr<CompiledRawTranspose2D> transpose_fwd,
                    std::shared_ptr<CompiledRawTranspose2D> transpose_inv,
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
  std::shared_ptr<CompiledRawTranspose2D> transpose_fwd;
  std::shared_ptr<CompiledRawTranspose2D> transpose_inv;
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

struct CompiledRaw2DFusedNode final : CompiledRawNode {
  CompiledRaw2DFusedNode(int64_t n0,
                         int64_t n1,
                         std::shared_ptr<JitKernel> kernel,
                         DeviceAllocation tw_r,
                         DeviceAllocation tw_i,
                         DeviceAllocation temp);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  int64_t n0;
  int64_t n1;
  std::shared_ptr<JitKernel> kernel;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
  DeviceAllocation temp;
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
                          std::shared_ptr<CompiledRawTranspose2D> transpose_fwd,
                          std::shared_ptr<CompiledRawTranspose2D> transpose_inv,
                          DeviceAllocation temp1,
                          DeviceAllocation temp2,
                          bool col_permuted_store = false);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> row_r2c;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<CompiledRawTranspose2D> transpose_fwd;
  std::shared_ptr<CompiledRawTranspose2D> transpose_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  bool col_permuted_store;
};

struct CompiledRaw2DC2RRowNode final : CompiledRawNode {
  CompiledRaw2DC2RRowNode(int64_t n0,
                          int64_t n1,
                          std::shared_ptr<CompiledRawNode> col_fft,
                          std::shared_ptr<CompiledRawNode> row_c2r,
                          std::shared_ptr<CompiledRawTranspose2D> transpose_fwd,
                          std::shared_ptr<CompiledRawTranspose2D> transpose_inv,
                          DeviceAllocation temp1,
                          DeviceAllocation temp2,
                          bool col_permuted_store = false);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  std::shared_ptr<CompiledRawNode> col_fft;
  std::shared_ptr<CompiledRawNode> row_c2r;
  std::shared_ptr<CompiledRawTranspose2D> transpose_fwd;
  std::shared_ptr<CompiledRawTranspose2D> transpose_inv;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  bool col_permuted_store;
};

struct CompiledRaw2DR2CNode final : CompiledRawNode {
  CompiledRaw2DR2CNode(int64_t n0,
                       int64_t n1,
                       std::shared_ptr<JitKernel> expand_kernel,
                       std::shared_ptr<CompiledRawNode> row_fft,
                       std::shared_ptr<JitKernel> pack_kernel,
                       std::shared_ptr<CompiledRawNode> col_fft,
                       std::shared_ptr<CompiledRawTranspose2D> transpose_fwd,
                       std::shared_ptr<CompiledRawTranspose2D> transpose_inv,
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
  std::shared_ptr<CompiledRawTranspose2D> transpose_fwd;
  std::shared_ptr<CompiledRawTranspose2D> transpose_inv;
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
                       std::shared_ptr<CompiledRawTranspose2D> transpose_fwd,
                       std::shared_ptr<CompiledRawTranspose2D> transpose_inv,
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
  std::shared_ptr<CompiledRawTranspose2D> transpose_fwd;
  std::shared_ptr<CompiledRawTranspose2D> transpose_inv;
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
                    DeviceAllocation temp2,
                    std::vector<DeviceAllocation> npu_transpose_indices = {},
                    bool npu_pair_fused_store = false);
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
                    DeviceAllocation temp2,
                    bool final_permuted_store);
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
  std::vector<DeviceAllocation> npu_transpose_indices;
  bool npu_pair_fused_store;
  bool final_permuted_store = false;
};

// Transform a contiguous inner plane, then the outer axis in natural layout.
// Real inverse reverses the passes so the plane sees compact Hermitian data.
struct CompiledRawVectorDIFNode final : CompiledRawNode {
  struct Pass {
    std::shared_ptr<JitKernel> kernel;
    DeviceAllocation weights;
    int64_t grid_x, grid_y;
    bool transpose;
    int64_t batch_pack=1;
  };
  CompiledRawVectorDIFNode(int64_t length, int64_t max_batch,
      std::vector<Pass> passes, DeviceAllocation first, DeviceAllocation second,
      int64_t workspace_length = 0, std::vector<int64_t> factors = {});
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t length, max_batch, workspace_length;
  std::vector<Pass> passes;
  std::vector<int64_t> factors;
  DeviceAllocation first, second;
};

struct CompiledRawMatrixFourStepNode final : CompiledRawNode {
  CompiledRawMatrixFourStepNode(int64_t n1, int64_t n2, int64_t max_batch,
      std::vector<std::shared_ptr<JitKernel>> kernels,
      std::vector<DeviceAllocation> tables, std::vector<DeviceAllocation> scratch, bool wide=false,
      std::vector<std::shared_ptr<JitKernel>> real_kernels={},
      std::shared_ptr<JitKernel> real_inverse_kernel=nullptr,
      std::shared_ptr<JitKernel> fused_pack_kernel=nullptr,
      std::shared_ptr<JitKernel> real_finish_kernel=nullptr,
      std::shared_ptr<JitKernel> real_fused_pack_kernel=nullptr,
      std::shared_ptr<JitKernel> complex_finish_kernel=nullptr,
      std::vector<std::shared_ptr<JitKernel>> complex_half_kernels={},
      std::vector<DeviceAllocation> complex_half_tables={}, bool complex_direct_input=false, bool complex_direct_combined=false);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  flagfftResult execute_real_forward(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const;
  flagfftResult execute_real_inverse(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const;
  flagfftResult execute_impl(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context, bool real_forward, bool real_inverse) const;
  bool supports_real_forward() const { return real_kernels.size() == 4; }
  bool supports_real_inverse() const { return real_inverse_kernel != nullptr; }
  std::string describe() const override;
  int64_t n1, n2, max_batch;
  bool wide, complex_direct_input, complex_direct_combined;
  std::vector<std::shared_ptr<JitKernel>> kernels, real_kernels;
  std::shared_ptr<JitKernel> real_inverse_kernel, fused_pack_kernel, real_finish_kernel, real_fused_pack_kernel, complex_finish_kernel;
  std::vector<std::shared_ptr<JitKernel>> complex_half_kernels;
  std::vector<DeviceAllocation> complex_half_tables;
  std::vector<DeviceAllocation> tables, scratch;
};

struct CompiledRawPairedMatrixC2RNode final : CompiledRawNode {
  CompiledRawPairedMatrixC2RNode(int64_t n1,int64_t n2,int64_t max_batch,
      std::vector<std::shared_ptr<JitKernel>> kernels,
      std::vector<DeviceAllocation> tables,std::vector<DeviceAllocation> scratch);
  flagfftResult execute(adaptor::DevicePtr input,adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t n1,n2,max_batch;
  std::vector<std::shared_ptr<JitKernel>> kernels;
  std::vector<DeviceAllocation> tables,scratch;
};

struct CompiledRawMatrixFFTNode final : CompiledRawNode {
  CompiledRawMatrixFFTNode(int64_t length, int64_t mode, int64_t max_batch,
      std::shared_ptr<JitKernel> pack, std::shared_ptr<JitKernel> cube,
      DeviceAllocation weights, DeviceAllocation scratch);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t length, mode, max_batch;
  std::shared_ptr<JitKernel> pack, cube;
  DeviceAllocation weights, scratch;
};

struct CompiledRawTensorPlaneNode final : CompiledRawNode {
  CompiledRawTensorPlaneNode(int64_t outer_length, int64_t plane_points,
      int64_t compact_points, bool inverse_real,
      std::shared_ptr<CompiledRawNode> plane, std::shared_ptr<CompiledRawNode> outer,
      DeviceAllocation temp);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
      const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t outer_length, plane_points, compact_points;
  bool inverse_real;
  std::shared_ptr<CompiledRawNode> plane, outer;
  DeviceAllocation temp;
};

// Small 16^3 complex transform: one kernel handles both axes in each 16x16
// plane, then the outer axis uses the existing strided leaf.
struct CompiledRaw3DFusedPlaneNode final : CompiledRawNode {
  CompiledRaw3DFusedPlaneNode(int64_t plane_size,
                                std::shared_ptr<JitKernel> plane_fft,
                                std::shared_ptr<CompiledRawNode> outer_fft,
                                DeviceAllocation temp,
                                DeviceAllocation tw_r,
                                DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t plane_size;
  std::shared_ptr<JitKernel> plane_fft;
  std::shared_ptr<CompiledRawNode> outer_fft;
  DeviceAllocation temp;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

// Experimental HCU path for a single 32^3 real transform. The fused plane
// kernel transforms the two inner axes; the existing strided leaf handles
// the outer axis before or after it, depending on transform direction.
struct CompiledRaw3DRealFusedPlaneNode final : CompiledRawNode {
  CompiledRaw3DRealFusedPlaneNode(int64_t n0,
                                  int64_t n1,
                                  int64_t n2,
                                  bool inverse,
                                  std::shared_ptr<JitKernel> plane_fft,
                                  std::shared_ptr<CompiledRawNode> outer_fft,
                                  DeviceAllocation temp,
                                  DeviceAllocation tw_r,
                                  DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  bool inverse;
  std::shared_ptr<JitKernel> plane_fft;
  std::shared_ptr<CompiledRawNode> outer_fft;
  DeviceAllocation temp;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

// Experimental HCU path for 16x997x64 C2C: fuse the two short axes in one
// 16x64 plane kernel, run the 997-point transform on contiguous rows, then
// restore natural output order with one tiled transpose.
struct CompiledRaw3DPrimePlaneNode final : CompiledRawNode {
  CompiledRaw3DPrimePlaneNode(int64_t n0,
                              int64_t n1,
                              int64_t n2,
                              std::shared_ptr<JitKernel> plane_fft,
                              std::shared_ptr<CompiledRawNode> middle_fft,
                              std::shared_ptr<JitKernel> output_transpose,
                              DeviceAllocation temp1,
                              DeviceAllocation temp2,
                              DeviceAllocation tw_r,
                              DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  std::shared_ptr<JitKernel> plane_fft;
  std::shared_ptr<CompiledRawNode> middle_fft;
  std::shared_ptr<JitKernel> output_transpose;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

struct CompiledRaw3DFusedCubeNode final : CompiledRawNode {
  CompiledRaw3DFusedCubeNode(std::shared_ptr<JitKernel> fft,
                            DeviceAllocation tw_r,
                            DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  bool is_single_kernel() const override { return true; }

  std::shared_ptr<JitKernel> fft;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

// Coalesced outer-axis FFT for the small 32-point IX plane route.
struct CompiledRaw3DColumnNode final : CompiledRawNode {
  CompiledRaw3DColumnNode(int64_t outer_stride,
                          int64_t columns,
                          std::shared_ptr<JitKernel> kernel,
                          DeviceAllocation tw_r,
                          DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t outer_stride;
  int64_t columns;
  std::shared_ptr<JitKernel> kernel;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

// Experimental 32^3 complex path: fuse the two inner axes in each 32x32
// plane, then apply the outer strided axis.
struct CompiledRaw3DFused32PlaneNode final : CompiledRawNode {
  CompiledRaw3DFused32PlaneNode(std::shared_ptr<JitKernel> plane_fft,
                                std::shared_ptr<CompiledRawNode> outer_fft,
                                DeviceAllocation temp,
                                DeviceAllocation tw_r,
                                DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  std::shared_ptr<JitKernel> plane_fft;
  std::shared_ptr<CompiledRawNode> outer_fft;
  DeviceAllocation temp;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};

// Experimental real 32-cube path: fuse the contiguous real axis and middle
// complex axis in each plane, then apply the outer strided axis.
struct CompiledRaw3DFusedRealPlaneNode final : CompiledRawNode {
  CompiledRaw3DFusedRealPlaneNode(int64_t plane_size,
                                  bool inverse,
                                  std::shared_ptr<JitKernel> plane_fft,
                                  std::shared_ptr<CompiledRawNode> outer_fft,
                                  DeviceAllocation temp,
                                  DeviceAllocation tw_r,
                                  DeviceAllocation tw_i);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t plane_size;
  bool inverse;
  std::shared_ptr<JitKernel> plane_fft;
  std::shared_ptr<CompiledRawNode> outer_fft;
  DeviceAllocation temp;
  DeviceAllocation tw_r;
  DeviceAllocation tw_i;
};
// 3D C2C/Z2Z that runs the n1 and n0 axes as strided leaves on the natural
// layout instead of permuting the cube between passes.
// Experimental two-pass Cooley-Tukey with resource-bounded local tiles.
struct CompiledRaw3DCTNode final : CompiledRawNode {
  CompiledRaw3DCTNode(int64_t elements,
                      std::shared_ptr<JitKernel> first,
                      std::shared_ptr<JitKernel> last,
                      DeviceAllocation temp);
  flagfftResult execute(adaptor::DevicePtr input, adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;
  int64_t elements;
  std::shared_ptr<JitKernel> first;
  std::shared_ptr<JitKernel> last;
  DeviceAllocation temp;
};

// Three launches and no full-cube transpose traffic, when both
// non-contiguous axis plans are plain leaves.
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

// Large 3D C2C with a long middle axis: contiguous n1 FFT plus one transpose
// can cost less than the fused n1 store.  The first n2 pass may also use a
// separate transpose when its fused store is less efficient.
struct CompiledRaw3DHybridNode final : CompiledRawNode {
  CompiledRaw3DHybridNode(int64_t n0,
                          int64_t n1,
                          int64_t n2,
                          std::shared_ptr<CompiledRawNode> n2_fft,
                          std::shared_ptr<CompiledRawNode> n1_fft,
                          std::shared_ptr<CompiledRawNode> n0_fft,
                          std::shared_ptr<JitKernel> perm_210,
                          DeviceAllocation temp1,
                          DeviceAllocation temp2,
                          std::shared_ptr<JitKernel> perm_021 = nullptr,
                          std::shared_ptr<JitKernel> perm_201 = nullptr);
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
  std::shared_ptr<JitKernel> perm_210;
  std::shared_ptr<JitKernel> perm_021;
  std::shared_ptr<JitKernel> perm_201;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

// Compact real 3D path.  The innermost axis reads/writes the real boundary
// directly.  A non-leaf middle axis can run between two transposes while the
// outer leaf still fuses its final permutation.
struct CompiledRaw3DRealLeafNode final : CompiledRawNode {
  enum class Layout { Legacy, Strided, FusedStore, Transposed, R2CFirstStore };
  CompiledRaw3DRealLeafNode(int64_t n0,
                            int64_t n1,
                            int64_t n2,
                            bool inverse,
                            bool fused_store,
                            bool n2_permuted,
                            bool n1_strided_input,
                            std::shared_ptr<CompiledRawNode> n2_real_fft,
                            std::shared_ptr<CompiledRawNode> n1_fft,
                            std::shared_ptr<CompiledRawNode> n0_fft,
                            std::shared_ptr<JitKernel> perm_021,
                            DeviceAllocation temp1,
                            DeviceAllocation temp2,
                            std::shared_ptr<JitKernel> perm_210 = nullptr);
  CompiledRaw3DRealLeafNode(int64_t n0,
                            int64_t n1,
                            int64_t n2,
                            bool inverse,
                            Layout layout,
                            std::shared_ptr<CompiledRawNode> n2_real_fft,
                            std::shared_ptr<CompiledRawNode> n1_fft,
                            std::shared_ptr<CompiledRawNode> n0_fft,
                            std::shared_ptr<JitKernel> perm_first,
                            std::shared_ptr<JitKernel> perm_second,
                            std::shared_ptr<JitKernel> perm_third,
                            DeviceAllocation temp1,
                            DeviceAllocation temp2,
                            std::shared_ptr<JitKernel> perm_210 = nullptr);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  bool inverse;
  bool fused_store;
  bool n2_permuted;
  bool n1_strided_input;
  Layout layout;
  std::shared_ptr<CompiledRawNode> n2_real_fft;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<JitKernel> perm_021;
  std::shared_ptr<JitKernel> perm_210;
  std::shared_ptr<JitKernel> perm_first;
  std::shared_ptr<JitKernel> perm_second;
  std::shared_ptr<JitKernel> perm_third;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
};

// Compact real boundary with three transposes. Native NPU paths can replace
// those transposes with indexed device operations.
struct CompiledRaw3DRealRTRTNode final : CompiledRawNode {
  CompiledRaw3DRealRTRTNode(int64_t n0,
                            int64_t n1,
                            int64_t n2,
                            bool inverse,
                            std::shared_ptr<CompiledRawNode> n2_real_fft,
                            std::shared_ptr<CompiledRawNode> n1_fft,
                            std::shared_ptr<CompiledRawNode> n0_fft,
                            std::shared_ptr<JitKernel> perm_021,
                            std::shared_ptr<JitKernel> perm_210,
                            std::shared_ptr<JitKernel> perm_201,
                            DeviceAllocation temp1,
                            DeviceAllocation temp2,
                            std::vector<DeviceAllocation> npu_transpose_indices = {},
                            bool npu_fused_outer_stores = false,
                            bool npu_fused_small_final_store = false);
  flagfftResult execute(adaptor::DevicePtr input,
                        adaptor::DevicePtr output,
                        const RawExecutionContext &context) const override;
  std::string describe() const override;

  int64_t n0;
  int64_t n1;
  int64_t n2;
  bool inverse;
  std::shared_ptr<CompiledRawNode> n2_real_fft;
  std::shared_ptr<CompiledRawNode> n1_fft;
  std::shared_ptr<CompiledRawNode> n0_fft;
  std::shared_ptr<JitKernel> perm_021;
  std::shared_ptr<JitKernel> perm_210;
  std::shared_ptr<JitKernel> perm_201;
  DeviceAllocation temp1;
  DeviceAllocation temp2;
  std::vector<DeviceAllocation> npu_transpose_indices;
  bool npu_fused_outer_stores;
  bool npu_fused_small_final_store;
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
                       DeviceAllocation temp2,
                       std::vector<DeviceAllocation> npu_transpose_indices = {});
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
  std::vector<DeviceAllocation> npu_transpose_indices;
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
                       DeviceAllocation full_buf,
                       std::vector<DeviceAllocation> npu_transpose_indices = {});
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
  std::vector<DeviceAllocation> npu_transpose_indices;
};

class TritonCompiler {
 public:
  std::shared_ptr<CompiledRawNode> compile_raw_node(const PlanNodePtr &node,
                                                    const FFTRequest &request,
                                                    int64_t batch,
                                                    bool allow_npu_aiv256_transposed_store = true);
  std::shared_ptr<CompiledRawNode> compile_raw_hcu_packed_r2c_store(const FFTRequest &request,
                                                                  int64_t batch, int64_t n,
                                                                  int64_t span);
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
  std::shared_ptr<CompiledRawNode> compile_raw_real_batch_pair(
      const PlanNodePtr &node, const FFTRequest &request, int64_t batch, bool inverse);

#if defined(FLAGFFT_BACKEND_NPU)
  std::shared_ptr<CompiledRawNode> make_npu_aiv_fft64_child(
      const FFTRequest &request, int64_t stride, int32_t group_size,
      NpuAivFFT64Mode mode = NpuAivFFT64Mode::Complex, bool radix4_pair = false);
  std::shared_ptr<CompiledRawNode> make_npu_aiv_fft512_child(const FFTRequest &request);
#endif
  void configure_single_transform_policies(const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_raw_3d_real_leaf_node(
      const std::shared_ptr<ThreeDimPlanNode> &node,
      const FFTRequest &request,
      int64_t batch,
      bool inverse);
  std::shared_ptr<CompiledRawNode> compile_raw_3d_real_rtrt_node(
      const std::shared_ptr<ThreeDimPlanNode> &node,
      const FFTRequest &request,
      int64_t batch,
      bool inverse);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_rc_row(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch);

  bool maca_1d_single_policy_ = false;
  bool maca_1d_batch_policy_ = false;
  bool musa_1d_batch_policy_ = false;
  bool maca_2d_single_policy_ = false;
  bool maca_3d_policy_ = false;
  bool maca_3d_c2c32_single_cube_policy_ = false;
  bool ix_ct_single_policy_ = false;
  bool ix_ct_batch_policy_ = false;
  bool ix_real_single_pack_ = false;
  bool ix_prime_3d_policy_ = false;
  int ix_fourstep_preset_ = 0;
  int ix_ct_single_tle_policy_ = 0;
  std::string maca_tail_policy_ = "off";
  // The Bluestein convolution FFT has a row/column layout the TLE AIV stage
  // does not support yet; keep it on the native/JIT path.
  bool npu_stockham_tle_allowed_ = true;

  std::shared_ptr<CompiledRawNode> compile_raw_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_ix_split_leaf(
      int64_t length, const FFTRequest &request, const std::string &form,
      std::shared_ptr<CompiledRawNode> fallback);
  std::shared_ptr<CompiledRawNode> compile_ix_wave_bands(
      int64_t length, int64_t bands, const FFTRequest &request,
      std::shared_ptr<CompiledRawNode> fallback);
  std::shared_ptr<CompiledRawNode> compile_raw_permuted_store_leaf(const LeafPlanNode &leaf,
                                                                    const FFTRequest &request,
                                                                    int64_t perm_span,
                                                                    const std::string &perm_form,
                                                                    bool strided_input = false,
                                                                    int64_t rows_per_transform = 0);
  std::shared_ptr<CompiledRawNode> compile_raw_strided_leaf(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t outer_stride);
  std::shared_ptr<CompiledRawNode> compile_raw_strided_direct_dft(const DirectDFTPlanNode &node,
                                                                  const FFTRequest &request,
                                                                  int64_t outer_stride);
  std::shared_ptr<CompiledRawNode> compile_raw_direct_dft(const DirectDFTPlanNode &node,
                                                          const FFTRequest &request,
                                                          int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_cube_transposed_direct_dft(const DirectDFTPlanNode &node,
                                                                          const FFTRequest &request,
                                                                          int64_t batch);
  std::shared_ptr<JitKernel> compile_direct_dft_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<CompiledRawNode> compile_raw_real_direct_dft(const FFTRequest &request,
                                                            bool inverse);
  std::shared_ptr<CompiledRawNode> compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                 const FFTRequest &request,
                                                                 int64_t batch);
  std::shared_ptr<JitKernel> compile_leaf_r2c_kernel(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_leaf_r2c_permuted_store_kernel(
      const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_leaf_c2r_kernel(const LeafPlanNode &leaf, const FFTRequest &request,
                                                      const std::string &perm_form = "outer");
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
                                                              int64_t m,
                                                              bool real_boundaries = false);
  std::shared_ptr<JitKernel> compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                int64_t n,
                                                                int64_t m);
  std::shared_ptr<JitKernel> compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                               int64_t n,
                                                               int64_t m,
                                                               bool real_boundaries = false);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_kernel(const LeafPlanNode &leaf,
                                                           const FFTRequest &request,
                                                           int64_t n);
  std::shared_ptr<JitKernel> compile_leaf_rader_full_kernel(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t n,
                                                            bool compact_inverse);
  std::shared_ptr<JitKernel> compile_leaf_rader_prepare_kernel(const LeafPlanNode &leaf,
                                                                const FFTRequest &request,
                                                                int64_t n);
  std::shared_ptr<JitKernel> compile_leaf_rader_finish_kernel(const LeafPlanNode &leaf,
                                                               const FFTRequest &request,
                                                               int64_t n);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_prepare_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request,
                                                                   int64_t n,
                                                                   const std::string &perm_form);
  std::shared_ptr<JitKernel> compile_leaf_bluestein_finish_kernel(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request,
                                                                  int64_t n,
                                                                  const std::string &perm_form);
  std::shared_ptr<JitKernel> compile_rader_prepare_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_pointwise_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_finalize_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_reshape_pack_kernel(const FFTRequest &request, int64_t n1, int64_t n2);
  std::shared_ptr<JitKernel> compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                 int64_t n1,
                                                                 int64_t n2);
  std::shared_ptr<JitKernel> compile_real_to_complex_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_half_pack_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_packed_postprocess_kernel(const FFTRequest &request, int64_t n, int64_t batch = 1);
  std::shared_ptr<JitKernel> compile_c2r_packed_preprocess_kernel(const FFTRequest &request, int64_t n, int64_t batch = 1);
  std::shared_ptr<JitKernel> compile_compact_to_hermitian_full_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_complex_to_real_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_tiled_transpose_kernel(const FFTRequest &request,
                                                            int64_t n0,
                                                            int64_t n1);
  std::shared_ptr<CompiledRawTranspose2D> compile_2d_transpose_kernel(const FFTRequest &request,
                                                                      int64_t rows,
                                                                      int64_t cols,
                                                                      int64_t batch);
  std::shared_ptr<JitKernel> compile_transpose3d_kernel(
      const FFTRequest &request, int64_t n0, int64_t n1, int64_t n2, const std::string &order,
      bool npu_table = false);
 public:
  std::shared_ptr<JitKernel> compile_kernel(const KernelKey &key) const;
 private:
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
std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request,
                                                   bool portable_default = false,
                                                   bool ix_fixed_lanes = false);
int64_t permuted_store_outer_blocks(int64_t transforms, int64_t span, int64_t pack);
bool hcu_wave_3d_c2c_policy_enabled(const FFTRequest &request,
                                    int64_t n0, int64_t n1, int64_t n2);
std::vector<int64_t> hcu_packed_cta_factors(const FFTRequest &request, const LeafPlanNode &leaf);
std::vector<int64_t> hcu_real_inverse_factors(const FFTRequest &request, const LeafPlanNode &leaf);
int64_t hcu_packed_grid_pack_limit(int64_t batch, int64_t compute_units, int64_t active_lanes);
bool ix_ct_single_policy_enabled(const FFTRequest &request);
bool ix_real_single_pack_policy_enabled(const FFTRequest &request);
bool ix_ct_batch_policy_enabled(const FFTRequest &request);
bool ix_rader1009_policy_enabled(const FFTRequest &request);
int ix_fourstep_default_preset(const FFTRequest &request);
bool ix_packed_real_policy_enabled(const FFTRequest &request);
int ix_ct_single_tle_policy(const FFTRequest &request);
int64_t ix_nd_warp_default_pack(const FFTRequest &request, int64_t batch,
                                int64_t n0, int64_t n1, int64_t n2 = 0);
bool ix_prime_3d_default_enabled(const FFTRequest &request, int64_t batch,
                                 int64_t n0, int64_t n1, int64_t n2);
bool ix_small_real_warp_default_enabled(const FFTRequest &request);
bool ix_short_real_warp_default_enabled(const FFTRequest &request);
std::vector<DeviceAllocation> build_raw_direct_dft_tables(int64_t n, const FFTRequest &request);
std::vector<DeviceAllocation> build_raw_cube_dft_tables(int64_t n, const FFTRequest &request);

}  // namespace flagfft
