#include "flagfft/core.hpp"

namespace flagfft {
namespace {

std::vector<AotKernelArg> raw_kernel_args(std::initializer_list<CUdeviceptr> ptrs,
                                          const std::vector<DeviceAllocation> &tables,
                                          int64_t batch) {
    std::vector<AotKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 1);
    for (CUdeviceptr ptr : ptrs) {
        args.push_back(AotKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
        args.push_back(AotKernelArg::device(table.ptr));
    }
    args.push_back(AotKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
}

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<AotKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

flagfftResult CompiledRawLeafNode::execute(CUdeviceptr input,
                                           CUdeviceptr output,
                                           const RawExecutionContext &context) const {
    try {
        std::vector<AotKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);

        kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_EXEC_FAILED;
    }
}

CompiledRawFourStepFusedNode::CompiledRawFourStepFusedNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<AotKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<AotKernel> col_kernel,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {}

flagfftResult CompiledRawFourStepFusedNode::execute(CUdeviceptr input,
                                                    CUdeviceptr output,
                                                    const RawExecutionContext &context) const {
    try {
        std::vector<AotKernelArg> row_args =
            raw_kernel_args({input, stage1.ptr}, row_tables, context.batch);
        row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

        std::vector<AotKernelArg> col_args =
            raw_kernel_args({stage1.ptr, twiddle.ptr, output}, col_tables, context.batch);
        col_kernel->launch(context.stream, col_args,
                           ceil_div(n1, four_step_col_inner_pack_for(n1, n2)),
                           context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_EXEC_FAILED;
    }
}

}  // namespace flagfft
