#include "flagfft/core.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace flagfft {
namespace {

std::vector<RuntimeKernelArg> raw_kernel_args(std::initializer_list<runtime::DevicePtr> ptrs,
                                          const std::vector<DeviceAllocation> &tables,
                                          int64_t batch) {
    std::vector<RuntimeKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 1);
    for (runtime::DevicePtr ptr : ptrs) {
        args.push_back(RuntimeKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
        args.push_back(RuntimeKernelArg::device(table.get()));
    }
    args.push_back(RuntimeKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
}

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<RuntimeKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

std::string CompiledRawLeafNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawLeaf(n=" << length
        << ", kernel=" << (kernel ? kernel->kernel_name : "null")
        << ", num_warps=" << (kernel ? kernel->num_warps : 0)
        << ", module=" << (kernel ? kernel->module_path : "null")
        << ", tables=" << tables.size() << ")";
    return oss.str();
}

flagfftResult CompiledRawLeafNode::execute(runtime::DevicePtr input,
                                           runtime::DevicePtr output,
                                           const RawExecutionContext &context) const {
    try {
        std::vector<RuntimeKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);

        kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] Leaf execute failed: %s\n", e.what());
        std::fflush(stderr);
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

std::string CompiledRawFourStepFusedNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawFourStepFused(n=" << length
        << ", n1=" << n1 << ", n2=" << n2
        << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
        << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
    return oss.str();
}

flagfftResult CompiledRawFourStepFusedNode::execute(runtime::DevicePtr input,
                                                    runtime::DevicePtr output,
                                                    const RawExecutionContext &context) const {
    try {
        std::vector<RuntimeKernelArg> row_args =
            raw_kernel_args({input, stage1.get()}, row_tables, context.batch);
        row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

        std::vector<RuntimeKernelArg> col_args =
            raw_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, context.batch);
        col_kernel->launch(context.stream, col_args,
                           ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                           context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] FourStepFused execute failed: %s\n", e.what());
        std::fflush(stderr);
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

std::string CompiledRawBluesteinNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawBluestein(n=" << length
        << ", conv_length=" << conv_length
        << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->kernel_name : "null")
        << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->kernel_name : "null")
        << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->kernel_name : "null")
        << ", fft=" << (fft ? fft->describe() : "null") << ")";
    return oss.str();
}

void CompiledRawBluesteinNode::ensure_b_fft(const RawExecutionContext &context) const {
    std::lock_guard<std::mutex> lock(b_fft_mutex);
    if (b_fft_ready) {
        return;
    }
    RawExecutionContext child_context{context.request, context.stream, 1};
    flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
        throw std::runtime_error("failed to precompute Bluestein convolution FFT");
    }
    b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinNode::execute(runtime::DevicePtr input,
                                                runtime::DevicePtr output,
                                                const RawExecutionContext &context) const {
    try {
        ensure_b_fft(context);

        std::vector<RuntimeKernelArg> prepare_args = {
            RuntimeKernelArg::device(input),
            RuntimeKernelArg::device(chirp.get()),
            RuntimeKernelArg::device(a_buf.get()),
            RuntimeKernelArg::i64(length),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), context.batch, 1);

        RawExecutionContext child_context{context.request, context.stream, context.batch};
        flagfftResult result = fft->execute(a_buf.get(), work_buf.get(), child_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> pointwise_args = {
            RuntimeKernelArg::device(work_buf.get()),
            RuntimeKernelArg::device(b_fft_buf.get()),
            RuntimeKernelArg::device(a_buf.get()),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        pointwise_kernel->launch(context.stream,
                                 pointwise_args,
                                 ceil_div(conv_length, 256),
                                 context.batch,
                                 1);

        result = fft->execute(a_buf.get(), work_buf.get(), child_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> finalize_args = {
            RuntimeKernelArg::device(work_buf.get()),
            RuntimeKernelArg::device(chirp.get()),
            RuntimeKernelArg::device(output),
            RuntimeKernelArg::i64(length),
            RuntimeKernelArg::i64(conv_length),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] Bluestein execute failed: %s\n", e.what());
        std::fflush(stderr);
        return FLAGFFT_EXEC_FAILED;
    }
}

CompiledRawFourStepGenericNode::CompiledRawFourStepGenericNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<CompiledRawNode> row_child,
    std::shared_ptr<CompiledRawNode> col_child,
    std::shared_ptr<RuntimeKernel> reshape_in_kernel,
    std::shared_ptr<RuntimeKernel> twiddle_reshape_kernel,
    std::shared_ptr<RuntimeKernel> final_pack_kernel,
    DeviceAllocation twiddle,
    DeviceAllocation stage1,
    DeviceAllocation stage2)
    : length(length),
      n1(n1),
      n2(n2),
      row_child(std::move(row_child)),
      col_child(std::move(col_child)),
      reshape_in_kernel(std::move(reshape_in_kernel)),
      twiddle_reshape_kernel(std::move(twiddle_reshape_kernel)),
      final_pack_kernel(std::move(final_pack_kernel)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)),
      stage2(std::move(stage2)) {}

std::string CompiledRawFourStepGenericNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawFourStepGeneric(n=" << length
        << ", n1=" << n1 << ", n2=" << n2
        << ", row_child=" << (row_child ? row_child->describe() : "null")
        << ", col_child=" << (col_child ? col_child->describe() : "null") << ")";
    return oss.str();
}

flagfftResult CompiledRawFourStepGenericNode::execute(runtime::DevicePtr input,
                                                     runtime::DevicePtr output,
                                                     const RawExecutionContext &context) const {
    try {
        const int64_t total = n1 * n2;
        const int64_t reshape_block = 256;

        std::vector<RuntimeKernelArg> reshape_in_args = {
            RuntimeKernelArg::device(input),
            RuntimeKernelArg::device(stage1.get()),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        reshape_in_kernel->launch(context.stream,
                                  reshape_in_args,
                                  ceil_div(total, reshape_block),
                                  context.batch,
                                  1);

        RawExecutionContext row_context{context.request, context.stream, context.batch * n2};
        flagfftResult result = row_child->execute(stage1.get(), stage2.get(), row_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> twiddle_args = {
            RuntimeKernelArg::device(stage2.get()),
            RuntimeKernelArg::device(twiddle.get()),
            RuntimeKernelArg::device(stage1.get()),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        twiddle_reshape_kernel->launch(context.stream,
                                       twiddle_args,
                                       ceil_div(total, reshape_block),
                                       context.batch,
                                       1);

        RawExecutionContext col_context{context.request, context.stream, context.batch * n1};
        result = col_child->execute(stage1.get(), stage2.get(), col_context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> final_args = {
            RuntimeKernelArg::device(stage2.get()),
            RuntimeKernelArg::device(output),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        final_pack_kernel->launch(context.stream,
                                  final_args,
                                  ceil_div(total, reshape_block),
                                  context.batch,
                                  1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] FourStepGeneric execute failed: %s\n", e.what());
        std::fflush(stderr);
        return FLAGFFT_EXEC_FAILED;
    }
}

CompiledRawR2CNode::CompiledRawR2CNode(int64_t length,
                                       std::shared_ptr<RuntimeKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<RuntimeKernel> pack_kernel,
                                       DeviceAllocation complex_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      complex_input(std::move(complex_input)),
      full_output(std::move(full_output)) {}

std::string CompiledRawR2CNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawR2C(n=" << length
        << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
        << ", fft=" << (fft ? fft->describe() : "null")
        << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
    return oss.str();
}

flagfftResult CompiledRawR2CNode::execute(runtime::DevicePtr input,
                                          runtime::DevicePtr output,
                                          const RawExecutionContext &context) const {
    try {
        constexpr int64_t block = 256;
        const int64_t half = length / 2 + 1;
        const bool in_place = input == output;
        const int64_t padded_real_distance = 2 * half;
        const int64_t input_distance =
            in_place ? std::max(context.input_distance, padded_real_distance)
                     : (context.input_distance > 0 ? context.input_distance : length);
        const int64_t output_distance =
            context.output_distance > 0 ? context.output_distance : half;
        std::vector<RuntimeKernelArg> expand_args = {
            RuntimeKernelArg::device(input),
            RuntimeKernelArg::device(complex_input.get()),
            RuntimeKernelArg::i64(input_distance),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

        flagfftResult result = fft->execute(complex_input.get(), full_output.get(), context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> pack_args = {
            RuntimeKernelArg::device(full_output.get()),
            RuntimeKernelArg::device(output),
            RuntimeKernelArg::i64(output_distance),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        pack_kernel->launch(context.stream,
                            pack_args,
                            ceil_div(length / 2 + 1, block),
                            context.batch,
                            1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] R2C execute failed: %s\n", e.what());
        std::fflush(stderr);
        return FLAGFFT_EXEC_FAILED;
    }
}

CompiledRawC2RNode::CompiledRawC2RNode(int64_t length,
                                       std::shared_ptr<RuntimeKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<RuntimeKernel> pack_kernel,
                                       DeviceAllocation full_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      full_input(std::move(full_input)),
      full_output(std::move(full_output)) {}

std::string CompiledRawC2RNode::describe() const {
    std::ostringstream oss;
    oss << "CompiledRawC2R(n=" << length
        << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
        << ", fft=" << (fft ? fft->describe() : "null")
        << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
    return oss.str();
}

flagfftResult CompiledRawC2RNode::execute(runtime::DevicePtr input,
                                          runtime::DevicePtr output,
                                          const RawExecutionContext &context) const {
    try {
        constexpr int64_t block = 256;
        const int64_t half = length / 2 + 1;
        const bool in_place = input == output;
        const int64_t padded_real_distance = 2 * half;
        const int64_t input_distance =
            context.input_distance > 0 ? context.input_distance : half;
        const int64_t output_distance =
            in_place ? std::max(context.output_distance, padded_real_distance)
                     : (context.output_distance > 0 ? context.output_distance : length);
        std::vector<RuntimeKernelArg> expand_args = {
            RuntimeKernelArg::device(input),
            RuntimeKernelArg::device(full_input.get()),
            RuntimeKernelArg::i64(input_distance),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

        flagfftResult result = fft->execute(full_input.get(), full_output.get(), context);
        if (result != FLAGFFT_SUCCESS) {
            return result;
        }

        std::vector<RuntimeKernelArg> pack_args = {
            RuntimeKernelArg::device(full_output.get()),
            RuntimeKernelArg::device(output),
            RuntimeKernelArg::i64(output_distance),
            RuntimeKernelArg::i32(static_cast<int32_t>(context.batch)),
        };
        pack_kernel->launch(context.stream, pack_args, ceil_div(length, block), context.batch, 1);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[flagfft] C2R execute failed: %s\n", e.what());
        std::fflush(stderr);
        return FLAGFFT_EXEC_FAILED;
    }
}

}  // namespace flagfft
