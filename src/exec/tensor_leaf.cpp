#include "flagfft/core.hpp"

namespace flagfft {

CompiledLeafNode::CompiledLeafNode(int64_t length,
                                   std::shared_ptr<AotKernel> kernel,
                                   std::vector<nb::object> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

nb::object CompiledLeafNode::execute(const nb::object &input, const ExecutionContext &context) const {
    bool root_leaf = length == context.request.requested_n;
    int64_t batch = root_leaf ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_leaf && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));
    nb::module_ torch = nb::module_::import_("torch");
    nb::object result = torch.attr("empty_like")(x_contig);

    std::vector<AotKernelArg> args;
    args.reserve(3 + tables.size());
    args.push_back(AotKernelArg::device(tensor_data_ptr(x_contig)));
    args.push_back(AotKernelArg::device(tensor_data_ptr(result)));
    for (const nb::object &table : tables) {
        args.push_back(AotKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(AotKernelArg::i32(static_cast<int32_t>(batch)));

    kernel->launch(context.stream, args, ceil_div(batch, kernel->batch_per_block), 1, 1);
    return result;
}

}  // namespace flagfft
