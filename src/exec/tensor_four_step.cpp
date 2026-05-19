#include "flagfft/core.hpp"

namespace flagfft {

CompiledFourStepNode::CompiledFourStepNode(int64_t length,
                                           int64_t n1,
                                           int64_t n2,
                                           std::shared_ptr<CompiledNode> row,
                                           std::shared_ptr<CompiledNode> col,
                                           std::shared_ptr<RuntimeKernel> transpose_kernel,
                                           std::shared_ptr<RuntimeKernel> twiddle_transpose_kernel,
                                           nb::object twiddle,
                                           nb::object stage0,
                                           nb::object stage2)
    : length(length),
      n1(n1),
      n2(n2),
      row(std::move(row)),
      col(std::move(col)),
      transpose_kernel(std::move(transpose_kernel)),
      twiddle_transpose_kernel(std::move(twiddle_transpose_kernel)),
      twiddle(std::move(twiddle)),
      stage0(std::move(stage0)),
      stage2(std::move(stage2)) {}

nb::object CompiledFourStepNode::execute(const nb::object &input, const ExecutionContext &context) const {
    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    launch_transpose(context.stream, x_contig.attr("reshape")(nb::make_tuple(batch, n1, n2)), stage0);
    nb::object stage1 = row->execute(
        stage0.attr("reshape")(nb::make_tuple(batch * n2, n1)), context);
    stage1 = stage1.attr("reshape")(nb::make_tuple(batch, n2, n1));

    launch_twiddle_transpose(context.stream, stage1, twiddle, stage2);

    nb::object stage3 = col->execute(
        stage2.attr("reshape")(nb::make_tuple(batch * n1, n2)), context);
    stage3 = stage3.attr("reshape")(nb::make_tuple(batch, n1, n2));

    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, n2, n1));
    launch_transpose(context.stream, stage3, out);
    return out.attr("reshape")(nb::make_tuple(batch, length));
}

void CompiledFourStepNode::launch_transpose(CUstream stream, const nb::object &src, const nb::object &dst) const {
    int64_t batch = tensor_size(src, 0);
    int64_t rows = tensor_size(src, 1);
    int64_t cols = tensor_size(src, 2);
    std::vector<RuntimeKernelArg> args = {
        RuntimeKernelArg::device(tensor_data_ptr(src)),
        RuntimeKernelArg::device(tensor_data_ptr(dst)),
        RuntimeKernelArg::i64(tensor_stride(src, 0) * 2),
        RuntimeKernelArg::i64(tensor_stride(src, 1) * 2),
        RuntimeKernelArg::i64(tensor_stride(src, 2) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 0) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 1) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 2) * 2),
        RuntimeKernelArg::i64(rows),
        RuntimeKernelArg::i64(cols),
    };
    transpose_kernel->launch(stream, args, ceil_div(cols, kFourStepTileCols),
                             ceil_div(rows, kFourStepTileRows), batch);
}

void CompiledFourStepNode::launch_twiddle_transpose(CUstream stream,
                                                    const nb::object &src,
                                                    const nb::object &twiddle,
                                                    const nb::object &dst) const {
    int64_t batch = tensor_size(src, 0);
    int64_t rows = tensor_size(src, 1);
    int64_t cols = tensor_size(src, 2);
    std::vector<RuntimeKernelArg> args = {
        RuntimeKernelArg::device(tensor_data_ptr(src)),
        RuntimeKernelArg::device(tensor_data_ptr(twiddle)),
        RuntimeKernelArg::device(tensor_data_ptr(dst)),
        RuntimeKernelArg::i64(tensor_stride(src, 0) * 2),
        RuntimeKernelArg::i64(tensor_stride(src, 1) * 2),
        RuntimeKernelArg::i64(tensor_stride(src, 2) * 2),
        RuntimeKernelArg::i64(tensor_stride(twiddle, 0) * 2),
        RuntimeKernelArg::i64(tensor_stride(twiddle, 1) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 0) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 1) * 2),
        RuntimeKernelArg::i64(tensor_stride(dst, 2) * 2),
        RuntimeKernelArg::i64(rows),
        RuntimeKernelArg::i64(cols),
    };
    twiddle_transpose_kernel->launch(stream, args, ceil_div(cols, kFourStepTileCols),
                                     ceil_div(rows, kFourStepTileRows), batch);
}

CompiledFourStepFusedNode::CompiledFourStepFusedNode(int64_t length,
                                                     int64_t n1,
                                                     int64_t n2,
                                                     std::shared_ptr<RuntimeKernel> row_kernel,
                                                     std::vector<nb::object> row_tables,
                                                     std::shared_ptr<RuntimeKernel> col_kernel,
                                                     std::vector<nb::object> col_tables,
                                                     nb::object twiddle,
                                                     nb::object stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {}

nb::object CompiledFourStepFusedNode::execute(const nb::object &input,
                                              const ExecutionContext &context) const {
    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    launch_row(context.stream, x_contig, stage1, batch);
    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, length));
    launch_col(context.stream, stage1, out, batch);
    return out;
}

void CompiledFourStepFusedNode::launch_row(CUstream stream,
                                           const nb::object &src,
                                           const nb::object &dst,
                                           int64_t batch) const {
    std::vector<RuntimeKernelArg> args;
    args.reserve(3 + row_tables.size());
    args.push_back(RuntimeKernelArg::device(tensor_data_ptr(src)));
    args.push_back(RuntimeKernelArg::device(tensor_data_ptr(dst)));
    for (const nb::object &table : row_tables) {
        args.push_back(RuntimeKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(RuntimeKernelArg::i32(static_cast<int32_t>(batch)));

    row_kernel->launch(stream, args, n2, batch, 1);
}

void CompiledFourStepFusedNode::launch_col(CUstream stream,
                                           const nb::object &src,
                                           const nb::object &dst,
                                           int64_t batch) const {
    std::vector<RuntimeKernelArg> args;
    args.reserve(4 + col_tables.size());
    args.push_back(RuntimeKernelArg::device(tensor_data_ptr(src)));
    args.push_back(RuntimeKernelArg::device(tensor_data_ptr(twiddle)));
    args.push_back(RuntimeKernelArg::device(tensor_data_ptr(dst)));
    for (const nb::object &table : col_tables) {
        args.push_back(RuntimeKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(RuntimeKernelArg::i32(static_cast<int32_t>(batch)));

    col_kernel->launch(stream, args, ceil_div(n1, four_step_col_inner_pack_for(n1, n2)), batch, 1);
}

}  // namespace flagfft
