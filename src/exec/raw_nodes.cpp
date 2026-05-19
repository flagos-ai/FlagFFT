#include "flagfft/core.hpp"

namespace flagfft {
namespace {

std::vector<RuntimeKernelArg> raw_kernel_args(std::initializer_list<CUdeviceptr> ptrs,
                                          const std::vector<DeviceAllocation> &tables,
                                          int64_t batch) {
    std::vector<RuntimeKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 1);
    for (CUdeviceptr ptr : ptrs) {
        args.push_back(RuntimeKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
        args.push_back(RuntimeKernelArg::device(table.ptr));
    }
    args.push_back(RuntimeKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
}

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<RuntimeKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

flagfftResult CompiledRawLeafNode::execute(CUdeviceptr input,
                                           CUdeviceptr output,
                                           const RawExecutionContext &context) const {
    try {
        std::vector<RuntimeKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);

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
    std::shared_ptr<RuntimeKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<RuntimeKernel> col_kernel,
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
        std::vector<RuntimeKernelArg> row_args =
            raw_kernel_args({input, stage1.ptr}, row_tables, context.batch);
        row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

        std::vector<RuntimeKernelArg> col_args =
            raw_kernel_args({stage1.ptr, twiddle.ptr, output}, col_tables, context.batch);
        col_kernel->launch(context.stream, col_args,
                           ceil_div(n1, four_step_col_inner_pack_for(n1, n2)),
                           context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_EXEC_FAILED;
    }
}

CompiledRawBluesteinNode::CompiledRawBluesteinNode(int64_t length,
                                                   int64_t conv_length,
                                                   std::shared_ptr<CompiledRawNode> fft,
                                                   std::shared_ptr<RuntimeKernel> prepare_kernel,
                                                   std::shared_ptr<RuntimeKernel> pointwise_kernel,
                                                   std::shared_ptr<RuntimeKernel> finalize_kernel,
                                                   DeviceAllocation chirp,
                                                   DeviceAllocation b_time,
                                                   DeviceAllocation a_buf,
                                                   DeviceAllocation work_buf,
                                                   DeviceAllocation b_fft_buf)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      pointwise_kernel(std::move(pointwise_kernel)),
      finalize_kernel(std::move(finalize_kernel)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      a_buf(std::move(a_buf)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)) {}

void CompiledRawBluesteinNode::ensure_b_fft(const RawExecutionContext &context) const {
    std::lock_guard<std::mutex> lock(b_fft_mutex);
    if (b_fft_ready) {
        return;
    }
    RawExecutionContext child_context{context.request, context.stream, 1};
    flagfftResult result = fft->execute(b_time.ptr, b_fft_buf.ptr, child_context);
    if (result != FLAGFFT_SUCCESS) {
        throw std::runtime_error("failed to precompute Bluestein convolution FFT");
    }
    b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinNode::execute(CUdeviceptr input,
                                                CUdeviceptr output,
                                                const RawExecutionContext &context) const {
    try {
        ensure_b_fft(context);

        std::vector<RuntimeKernelArg> prepare_args = {
            RuntimeKernelArg::device(input),
            RuntimeKernelArg::device(chirp.ptr),
            RuntimeKernelArg::device(a_buf.ptr),
            RuntimeKernelArg::i64(length),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), context.batch, 1);

        RawExecutionContext child_context{context.request, context.stream, context.batch};
        flagfftResult result = fft->execute(a_buf.ptr, work_buf.ptr, child_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> pointwise_args = {
            RuntimeKernelArg::device(work_buf.ptr),
            RuntimeKernelArg::device(b_fft_buf.ptr),
            RuntimeKernelArg::device(a_buf.ptr),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        pointwise_kernel->launch(context.stream,
                                 pointwise_args,
                                 ceil_div(conv_length, 256),
                                 context.batch,
                                 1);

        result = fft->execute(a_buf.ptr, work_buf.ptr, child_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> finalize_args = {
            RuntimeKernelArg::device(work_buf.ptr),
            RuntimeKernelArg::device(chirp.ptr),
            RuntimeKernelArg::device(output),
            RuntimeKernelArg::i64(length),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_EXEC_FAILED;
    }
}

}  // namespace flagfft
