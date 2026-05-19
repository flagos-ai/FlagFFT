#pragma once

#include "flagfft/runtime.hpp"

namespace flagfft {

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage);
int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit);
std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(
    const std::vector<int64_t> &radices, int64_t stage, int64_t lanes, const std::string &direction);
std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix,
                                                                   const std::string &direction);
nb::object build_four_step_twiddle_tensor(const FFTRequest &request, int64_t n1, int64_t n2);
nb::object build_bluestein_chirp_tensor(const FFTRequest &request, int64_t n, bool inverse_sign);
nb::object build_bluestein_b_tensor(const FFTRequest &request, int64_t n, int64_t m);

enum class RuntimeArgKind { DevicePtr, Int32, Int64 };

struct RuntimeKernelArg {
    static RuntimeKernelArg device(CUdeviceptr value);
    static RuntimeKernelArg i32(int32_t value);
    static RuntimeKernelArg i64(int64_t value);

    RuntimeArgKind kind = RuntimeArgKind::DevicePtr;
    CUdeviceptr device_ptr = 0;
    int32_t int32_value = 0;
    int64_t int64_value = 0;
};

struct RuntimeKernel {
    ~RuntimeKernel();
    void compile();
    void launch(CUstream stream,
                const std::vector<RuntimeKernelArg> &kernel_args,
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

struct ExecutionContext {
    const FFTRequest &request;
    CUstream stream = nullptr;
};

struct DeviceAllocation {
    DeviceAllocation() = default;
    DeviceAllocation(CUdeviceptr ptr, std::size_t bytes);
    ~DeviceAllocation();
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept;
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept;
    void reset();

    CUdeviceptr ptr = 0;
    std::size_t bytes = 0;
};

struct RawExecutionContext {
    const FFTRequest &request;
    CUstream stream = nullptr;
    int64_t batch = 0;
};

struct CompiledNode {
    virtual ~CompiledNode() = default;
    virtual nb::object execute(const nb::object &input, const ExecutionContext &context) const = 0;
};

struct CompiledRawNode {
    virtual ~CompiledRawNode() = default;
    virtual flagfftResult execute(CUdeviceptr input,
                                  CUdeviceptr output,
                                  const RawExecutionContext &context) const = 0;
};

struct CompiledRawLeafNode final : CompiledRawNode {
    CompiledRawLeafNode(int64_t length,
                        std::shared_ptr<RuntimeKernel> kernel,
                        std::vector<DeviceAllocation> tables);
    flagfftResult execute(CUdeviceptr input,
                          CUdeviceptr output,
                          const RawExecutionContext &context) const override;

    int64_t length;
    std::shared_ptr<RuntimeKernel> kernel;
    std::vector<DeviceAllocation> tables;
};

struct CompiledRawFourStepFusedNode final : CompiledRawNode {
    CompiledRawFourStepFusedNode(int64_t length,
                                 int64_t n1,
                                 int64_t n2,
                                 std::shared_ptr<RuntimeKernel> row_kernel,
                                 std::vector<DeviceAllocation> row_tables,
                                 std::shared_ptr<RuntimeKernel> col_kernel,
                                 std::vector<DeviceAllocation> col_tables,
                                 DeviceAllocation twiddle,
                                 DeviceAllocation stage1);
    flagfftResult execute(CUdeviceptr input,
                          CUdeviceptr output,
                          const RawExecutionContext &context) const override;

    int64_t length;
    int64_t n1;
    int64_t n2;
    std::shared_ptr<RuntimeKernel> row_kernel;
    std::vector<DeviceAllocation> row_tables;
    std::shared_ptr<RuntimeKernel> col_kernel;
    std::vector<DeviceAllocation> col_tables;
    DeviceAllocation twiddle;
    DeviceAllocation stage1;
};

struct CompiledRawBluesteinNode final : CompiledRawNode {
    CompiledRawBluesteinNode(int64_t length,
                             int64_t conv_length,
                             std::shared_ptr<CompiledRawNode> fft,
                             std::shared_ptr<RuntimeKernel> prepare_kernel,
                             std::shared_ptr<RuntimeKernel> pointwise_kernel,
                             std::shared_ptr<RuntimeKernel> finalize_kernel,
                             DeviceAllocation chirp,
                             DeviceAllocation b_time,
                             DeviceAllocation a_buf,
                             DeviceAllocation work_buf,
                             DeviceAllocation b_fft_buf);
    flagfftResult execute(CUdeviceptr input,
                          CUdeviceptr output,
                          const RawExecutionContext &context) const override;
    void ensure_b_fft(const RawExecutionContext &context) const;

    int64_t length;
    int64_t conv_length;
    std::shared_ptr<CompiledRawNode> fft;
    std::shared_ptr<RuntimeKernel> prepare_kernel;
    std::shared_ptr<RuntimeKernel> pointwise_kernel;
    std::shared_ptr<RuntimeKernel> finalize_kernel;
    DeviceAllocation chirp;
    DeviceAllocation b_time;
    DeviceAllocation a_buf;
    DeviceAllocation work_buf;
    mutable DeviceAllocation b_fft_buf;
    mutable bool b_fft_ready = false;
    mutable std::mutex b_fft_mutex;
};

struct CompiledLeafNode final : CompiledNode {
    CompiledLeafNode(int64_t length, std::shared_ptr<RuntimeKernel> kernel, std::vector<nb::object> tables);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;

    int64_t length;
    std::shared_ptr<RuntimeKernel> kernel;
    std::vector<nb::object> tables;
};

struct CompiledFourStepNode final : CompiledNode {
    CompiledFourStepNode(int64_t length,
                         int64_t n1,
                         int64_t n2,
                         std::shared_ptr<CompiledNode> row,
                         std::shared_ptr<CompiledNode> col,
                         std::shared_ptr<RuntimeKernel> transpose_kernel,
                         std::shared_ptr<RuntimeKernel> twiddle_transpose_kernel,
                         nb::object twiddle,
                         nb::object stage0,
                         nb::object stage2);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void launch_transpose(CUstream stream, const nb::object &src, const nb::object &dst) const;
    void launch_twiddle_transpose(CUstream stream,
                                  const nb::object &src,
                                  const nb::object &twiddle,
                                  const nb::object &dst) const;

    int64_t length;
    int64_t n1;
    int64_t n2;
    std::shared_ptr<CompiledNode> row;
    std::shared_ptr<CompiledNode> col;
    std::shared_ptr<RuntimeKernel> transpose_kernel;
    std::shared_ptr<RuntimeKernel> twiddle_transpose_kernel;
    nb::object twiddle;
    nb::object stage0;
    nb::object stage2;
};

struct CompiledFourStepFusedNode final : CompiledNode {
    CompiledFourStepFusedNode(int64_t length,
                              int64_t n1,
                              int64_t n2,
                              std::shared_ptr<RuntimeKernel> row_kernel,
                              std::vector<nb::object> row_tables,
                              std::shared_ptr<RuntimeKernel> col_kernel,
                              std::vector<nb::object> col_tables,
                              nb::object twiddle,
                              nb::object stage1);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void launch_row(CUstream stream, const nb::object &src, const nb::object &dst, int64_t batch) const;
    void launch_col(CUstream stream, const nb::object &src, const nb::object &dst, int64_t batch) const;

    int64_t length;
    int64_t n1;
    int64_t n2;
    std::shared_ptr<RuntimeKernel> row_kernel;
    std::vector<nb::object> row_tables;
    std::shared_ptr<RuntimeKernel> col_kernel;
    std::vector<nb::object> col_tables;
    nb::object twiddle;
    nb::object stage1;
};

struct CompiledBluesteinNode final : CompiledNode {
    CompiledBluesteinNode(int64_t length,
                          int64_t conv_length,
                          std::shared_ptr<CompiledNode> fft,
                          std::shared_ptr<RuntimeKernel> prepare_kernel,
                          std::shared_ptr<RuntimeKernel> pointwise_kernel,
                          std::shared_ptr<RuntimeKernel> finalize_kernel,
                          nb::object chirp,
                          nb::object b_time,
                          nb::object a_buf,
                          nb::object work_buf,
                          nb::object b_fft_buf);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void ensure_b_fft(const ExecutionContext &context) const;

    int64_t length;
    int64_t conv_length;
    std::shared_ptr<CompiledNode> fft;
    std::shared_ptr<RuntimeKernel> prepare_kernel;
    std::shared_ptr<RuntimeKernel> pointwise_kernel;
    std::shared_ptr<RuntimeKernel> finalize_kernel;
    nb::object chirp;
    nb::object b_time;
    nb::object a_buf;
    nb::object work_buf;
    mutable nb::object b_fft_buf;
    mutable bool b_fft_ready = false;
    mutable std::mutex b_fft_mutex;
};

class TritonCompiler {
public:
    std::shared_ptr<CompiledNode> compile_node(const PlanNodePtr &node,
                                               const FFTRequest &request,
                                               int64_t batch);
    std::shared_ptr<CompiledRawNode> compile_raw_node(const PlanNodePtr &node,
                                                      const FFTRequest &request,
                                                      int64_t batch);
    static void clear_kernel_cache();
    static nb::dict kernel_cache_info();
    static nb::list kernel_cache_keys();

private:
    std::shared_ptr<CompiledNode> compile_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
    std::shared_ptr<CompiledRawNode> compile_raw_leaf(const LeafPlanNode &leaf,
                                                      const FFTRequest &request);
    std::shared_ptr<RuntimeKernel> compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t n1,
                                                            int64_t n2);
    std::shared_ptr<RuntimeKernel> compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t n1,
                                                            int64_t n2);
    std::shared_ptr<RuntimeKernel> compile_transpose_kernel(const FFTRequest &request);
    std::shared_ptr<RuntimeKernel> compile_twiddle_transpose_kernel(const FFTRequest &request);
    std::shared_ptr<RuntimeKernel> compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                int64_t n,
                                                                int64_t m);
    std::shared_ptr<RuntimeKernel> compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                  int64_t n,
                                                                  int64_t m);
    std::shared_ptr<RuntimeKernel> compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                 int64_t n,
                                                                 int64_t m);
    std::shared_ptr<RuntimeKernel> compile_kernel(const KernelKey &key) const;
    std::filesystem::path out_dir() const;
    std::string python_executable() const;
    std::string triton_jit_source_entrypoint() const;

    std::shared_ptr<RuntimeKernel> transpose_kernel;
    std::shared_ptr<RuntimeKernel> twiddle_transpose_kernel;
};

std::string triton_target_for_request(const FFTRequest &request);
nb::list kernel_keys_for_plan(const PlanNodePtr &node, const FFTRequest &request);

FFTRequest forward_child_request(const FFTRequest &request);
DeviceAllocation allocate_device_bytes(std::size_t bytes);
DeviceAllocation build_raw_four_step_twiddle(const FFTRequest &request, int64_t n1, int64_t n2);
DeviceAllocation build_raw_bluestein_chirp(const FFTRequest &request, int64_t n, bool inverse_sign);
DeviceAllocation build_raw_bluestein_b(const FFTRequest &request, int64_t n, int64_t m);
std::vector<DeviceAllocation> build_raw_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request);
std::vector<nb::object> build_leaf_tables(const LeafPlanNode &leaf, const FFTRequest &request);

}  // namespace flagfft
