#pragma once

#include <string>

#include "adaptor/adaptor.h"
#include "flagfft/plan.hpp"

namespace flagfft {

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
  int64_t num_stages = 1;
  int64_t batch_per_block = 1;
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
  std::shared_ptr<JitKernel> pointwise_kernel;
  std::shared_ptr<JitKernel> finalize_kernel;
  DeviceAllocation idx;
  DeviceAllocation b_time;
  DeviceAllocation a_buf;
  DeviceAllocation work_buf;
  mutable DeviceAllocation b_fft_buf;
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

struct CompiledRaw2DNode final : CompiledRawNode {
  CompiledRaw2DNode(int64_t n0,
                    int64_t n1,
                    std::shared_ptr<CompiledRawNode> row_fft,
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
  std::shared_ptr<CompiledRawNode> row_fft;
  std::shared_ptr<CompiledRawNode> col_fft;
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

class TritonCompiler {
 public:
  std::shared_ptr<CompiledRawNode> compile_raw_node(const PlanNodePtr &node,
                                                    const FFTRequest &request,
                                                    int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_r2c_node(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_c2r_node(const PlanNodePtr &node,
                                                        const FFTRequest &request,
                                                        int64_t batch);
  std::shared_ptr<CompiledRaw2DNode> compile_raw_2d_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                         const FFTRequest &request,
                                                         int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_r2c_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  std::shared_ptr<CompiledRawNode> compile_raw_2d_c2r_node(const std::shared_ptr<TwoDimPlanNode> &node,
                                                           const FFTRequest &request,
                                                           int64_t batch);
  static void clear_kernel_cache();

 private:
  std::shared_ptr<CompiledRawNode> compile_raw_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                 const FFTRequest &request,
                                                                 int64_t batch);
  std::shared_ptr<JitKernel> compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                          const FFTRequest &request,
                                                          int64_t n1,
                                                          int64_t n2);
  std::shared_ptr<JitKernel> compile_four_step_col_kernel(const LeafPlanNode &leaf,
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
  std::shared_ptr<JitKernel> compile_rader_prepare_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_pointwise_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_rader_finalize_kernel(const FFTRequest &request, int64_t n, int64_t m);
  std::shared_ptr<JitKernel> compile_reshape_pack_kernel(const FFTRequest &request, int64_t n1, int64_t n2);
  std::shared_ptr<JitKernel> compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                 int64_t n1,
                                                                 int64_t n2);
  std::shared_ptr<JitKernel> compile_real_to_complex_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_half_pack_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_compact_to_hermitian_full_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_complex_to_real_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_tiled_transpose_kernel(const FFTRequest &request,
                                                            int64_t n0,
                                                            int64_t n1);
  std::shared_ptr<JitKernel> compile_kernel(const KernelKey &key) const;
  std::filesystem::path out_dir() const;
  std::string python_executable() const;
  std::string triton_jit_source_entrypoint() const;
};

std::string triton_target_for_request(const FFTRequest &request);
FFTRequest forward_child_request(const FFTRequest &request);
DeviceAllocation build_raw_four_step_twiddle(const FFTRequest &request, int64_t n1, int64_t n2);
DeviceAllocation build_raw_bluestein_chirp(const FFTRequest &request, int64_t n, bool inverse_sign);
DeviceAllocation build_raw_bluestein_b(const FFTRequest &request, int64_t n, int64_t m);
DeviceAllocation build_raw_rader_idx_table(const std::vector<int64_t> &idx);
DeviceAllocation build_raw_rader_conv_kernel(const FFTRequest &request,
                                             int64_t n,
                                             const std::vector<int64_t> &idx);
std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request);

}  // namespace flagfft
