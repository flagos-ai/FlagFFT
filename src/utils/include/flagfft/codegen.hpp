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

struct CompiledRawDirectDftNode final : CompiledRawNode {
  CompiledRawDirectDftNode(int64_t length,
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
  static void clear_kernel_cache();

 private:
  std::shared_ptr<CompiledRawNode> compile_raw_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<CompiledRawNode> compile_raw_direct_dft(const DirectDFTPlanNode &node,
                                                          const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_direct_dft_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<CompiledRawNode> compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                 const FFTRequest &request,
                                                                 int64_t batch);
  std::shared_ptr<JitKernel> compile_leaf_r2c_kernel(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_leaf_c2r_kernel(const LeafPlanNode &leaf, const FFTRequest &request);
  std::shared_ptr<JitKernel> compile_four_step_row_kernel(const LeafPlanNode &leaf,
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
  std::shared_ptr<JitKernel> compile_reshape_pack_kernel(const FFTRequest &request, int64_t n1, int64_t n2);
  std::shared_ptr<JitKernel> compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                 int64_t n1,
                                                                 int64_t n2);
  std::shared_ptr<JitKernel> compile_real_to_complex_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_r2c_half_pack_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_compact_to_hermitian_full_kernel(const FFTRequest &request, int64_t n);
  std::shared_ptr<JitKernel> compile_complex_to_real_kernel(const FFTRequest &request, int64_t n);
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
std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request);
std::vector<DeviceAllocation> build_raw_direct_dft_tables(int64_t n, const FFTRequest &request);

}  // namespace flagfft
