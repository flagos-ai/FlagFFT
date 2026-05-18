#include "flagfft/core.hpp"

namespace flagfft {

CompiledBluesteinNode::CompiledBluesteinNode(int64_t length,
                                             int64_t conv_length,
                                             std::shared_ptr<CompiledNode> fft,
                                             std::shared_ptr<AotKernel> prepare_kernel,
                                             std::shared_ptr<AotKernel> pointwise_kernel,
                                             std::shared_ptr<AotKernel> finalize_kernel,
                                             nb::object chirp,
                                             nb::object b_time,
                                             nb::object a_buf,
                                             nb::object work_buf,
                                             nb::object b_fft_buf)
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

void CompiledBluesteinNode::ensure_b_fft(const ExecutionContext &context) const {
    std::lock_guard<std::mutex> lock(b_fft_mutex);
    if (b_fft_ready) {
        return;
    }
    b_fft_buf = fft->execute(b_time, context);
    b_fft_ready = true;
}

nb::object CompiledBluesteinNode::execute(const nb::object &input, const ExecutionContext &context) const {
    ensure_b_fft(context);

    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    std::vector<AotKernelArg> prepare_args = {
        AotKernelArg::device(tensor_data_ptr(x_contig)),
        AotKernelArg::device(tensor_data_ptr(chirp)),
        AotKernelArg::device(tensor_data_ptr(a_buf)),
        AotKernelArg::i64(length),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), batch, 1);

    nb::object a_fft = fft->execute(a_buf, context);

    std::vector<AotKernelArg> pointwise_args = {
        AotKernelArg::device(tensor_data_ptr(a_fft)),
        AotKernelArg::device(tensor_data_ptr(b_fft_buf)),
        AotKernelArg::device(tensor_data_ptr(work_buf)),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), batch, 1);

    nb::object conv = fft->execute(work_buf, context);
    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, length));
    std::vector<AotKernelArg> finalize_args = {
        AotKernelArg::device(tensor_data_ptr(conv)),
        AotKernelArg::device(tensor_data_ptr(chirp)),
        AotKernelArg::device(tensor_data_ptr(out)),
        AotKernelArg::i64(length),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), batch, 1);
    return out;
}

}  // namespace flagfft
