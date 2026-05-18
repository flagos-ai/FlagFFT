#include "flagfft/core.hpp"

namespace flagfft {

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<AotKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

flagfftResult CompiledRawLeafNode::execute(CUdeviceptr input,
                                           CUdeviceptr output,
                                           const RawExecutionContext &context) const {
    try {
        std::vector<AotKernelArg> args;
        args.reserve(3 + tables.size());
        args.push_back(AotKernelArg::device(input));
        args.push_back(AotKernelArg::device(output));
        for (const DeviceAllocation &table : tables) {
            args.push_back(AotKernelArg::device(table.ptr));
        }
        args.push_back(AotKernelArg::i32(static_cast<int32_t>(context.batch)));

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
        std::vector<AotKernelArg> row_args;
        row_args.reserve(3 + row_tables.size());
        row_args.push_back(AotKernelArg::device(input));
        row_args.push_back(AotKernelArg::device(stage1.ptr));
        for (const DeviceAllocation &table : row_tables) {
            row_args.push_back(AotKernelArg::device(table.ptr));
        }
        row_args.push_back(AotKernelArg::i32(static_cast<int32_t>(context.batch)));
        row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

        std::vector<AotKernelArg> col_args;
        col_args.reserve(4 + col_tables.size());
        col_args.push_back(AotKernelArg::device(stage1.ptr));
        col_args.push_back(AotKernelArg::device(twiddle.ptr));
        col_args.push_back(AotKernelArg::device(output));
        for (const DeviceAllocation &table : col_tables) {
            col_args.push_back(AotKernelArg::device(table.ptr));
        }
        col_args.push_back(AotKernelArg::i32(static_cast<int32_t>(context.batch)));
        col_kernel->launch(context.stream, col_args,
                           ceil_div(n1, four_step_col_inner_pack_for(n1, n2)),
                           context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_EXEC_FAILED;
    }
}

}  // namespace flagfft
