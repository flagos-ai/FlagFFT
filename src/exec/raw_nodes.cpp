#include "flagfft/core.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace flagfft {
namespace {

  std::vector<JitKernelArg> raw_kernel_args(std::initializer_list<adaptor::DevicePtr> ptrs,
                                            const std::vector<DeviceAllocation> &tables,
                                            int64_t batch) {
    std::vector<JitKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 1);
    for (adaptor::DevicePtr ptr : ptrs) {
      args.push_back(JitKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
  }

  std::vector<JitKernelArg> raw_distance_col_kernel_args(std::initializer_list<adaptor::DevicePtr> ptrs,
                                                         const std::vector<DeviceAllocation> &tables,
                                                         int64_t output_distance,
                                                         int64_t batch) {
    std::vector<JitKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 2);
    for (adaptor::DevicePtr ptr : ptrs) {
      args.push_back(JitKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
  }

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<JitKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawLeafNode::execute(adaptor::DevicePtr input,
                                           adaptor::DevicePtr output,
                                           const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);

    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Leaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawFourStepFusedNode::CompiledRawFourStepFusedNode(int64_t length,
                                                           int64_t n1,
                                                           int64_t n2,
                                                           std::shared_ptr<JitKernel> row_kernel,
                                                           std::vector<DeviceAllocation> row_tables,
                                                           std::shared_ptr<JitKernel> col_kernel,
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
      stage1(std::move(stage1)) {
}

std::string CompiledRawFourStepFusedNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawFourStepFused(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepFusedNode::execute(adaptor::DevicePtr input,
                                                    adaptor::DevicePtr output,
                                                    const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> row_args = raw_kernel_args({input, stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

    std::vector<JitKernelArg> col_args =
        raw_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, context.batch);
    col_kernel->launch(context.stream,
                       col_args,
                       ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                       context.batch,
                       1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepFused execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawDirectDftNode::CompiledRawDirectDftNode(int64_t length,
                                                   std::shared_ptr<JitKernel> kernel,
                                                   std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawDirectDftNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawDirectDft(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawDirectDftNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);
    kernel->launch(context.stream, args, context.batch, 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] DirectDFT execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinNode::CompiledRawBluesteinNode(int64_t length,
                                                   int64_t conv_length,
                                                   std::shared_ptr<CompiledRawNode> fft,
                                                   std::shared_ptr<JitKernel> prepare_kernel,
                                                   std::shared_ptr<JitKernel> pointwise_kernel,
                                                   std::shared_ptr<JitKernel> finalize_kernel,
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
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluestein(n=" << length << ", conv_length=" << conv_length
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
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute Bluestein convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    ensure_b_fft(context);

    std::vector<JitKernelArg> prepare_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(chirp.get()),
        JitKernelArg::device(a_buf.get()),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), context.batch, 1);

    RawExecutionContext child_context {context.request, context.stream, context.batch};
    flagfftResult result = fft->execute(a_buf.get(), work_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pointwise_args = {
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(b_fft_buf.get()),
        JitKernelArg::device(a_buf.get()),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), context.batch, 1);

    result = fft->execute(a_buf.get(), work_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> finalize_args = {
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(chirp.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
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
    std::shared_ptr<JitKernel> reshape_in_kernel,
    std::shared_ptr<JitKernel> twiddle_reshape_kernel,
    std::shared_ptr<JitKernel> final_pack_kernel,
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
      stage2(std::move(stage2)) {
}

std::string CompiledRawFourStepGenericNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawFourStepGeneric(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_child=" << (row_child ? row_child->describe() : "null")
      << ", col_child=" << (col_child ? col_child->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepGenericNode::execute(adaptor::DevicePtr input,
                                                      adaptor::DevicePtr output,
                                                      const RawExecutionContext &context) const {
  try {
    const int64_t total = n1 * n2;
    const int64_t reshape_block = 256;

    std::vector<JitKernelArg> reshape_in_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(stage1.get()),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    reshape_in_kernel->launch(context.stream,
                              reshape_in_args,
                              ceil_div(total, reshape_block),
                              context.batch,
                              1);

    RawExecutionContext row_context {context.request, context.stream, context.batch * n2};
    flagfftResult result = row_child->execute(stage1.get(), stage2.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> twiddle_args = {
        JitKernelArg::device(stage2.get()),
        JitKernelArg::device(twiddle.get()),
        JitKernelArg::device(stage1.get()),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    twiddle_reshape_kernel->launch(context.stream,
                                   twiddle_args,
                                   ceil_div(total, reshape_block),
                                   context.batch,
                                   1);

    RawExecutionContext col_context {context.request, context.stream, context.batch * n1};
    result = col_child->execute(stage1.get(), stage2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> final_args = {
        JitKernelArg::device(stage2.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    final_pack_kernel->launch(context.stream, final_args, ceil_div(total, reshape_block), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepGeneric execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CNode::CompiledRawR2CNode(int64_t length,
                                       std::shared_ptr<JitKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<JitKernel> pack_kernel,
                                       DeviceAllocation complex_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      complex_input(std::move(complex_input)),
      full_output(std::move(full_output)) {
}

std::string CompiledRawR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2C(n=" << length
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CNode::execute(adaptor::DevicePtr input,
                                          adaptor::DevicePtr output,
                                          const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(complex_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    flagfftResult result = fft->execute(complex_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(full_output.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(length / 2 + 1, block), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CLeafNode::CompiledRawR2CLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawR2CLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CLeafNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;

    std::vector<JitKernelArg> args;
    args.reserve(2 + tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(input_distance));
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CFourStepHalfOutNode::CompiledRawR2CFourStepHalfOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> expand_kernel,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation complex_input,
    DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      expand_kernel(std::move(expand_kernel)),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      complex_input(std::move(complex_input)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawR2CFourStepHalfOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CFourStepHalfOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CFourStepHalfOutNode::execute(adaptor::DevicePtr input,
                                                         adaptor::DevicePtr output,
                                                         const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;

    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(complex_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    std::vector<JitKernelArg> row_args = raw_kernel_args({complex_input.get(), stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

    std::vector<JitKernelArg> col_args =
        raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, output_distance, context.batch);
    col_kernel->launch(context.stream,
                       col_args,
                       ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                       context.batch,
                       1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CFourStepHalfOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CFourStepRealInHalfOutNode::CompiledRawR2CFourStepRealInHalfOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
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
      stage1(std::move(stage1)) {
}

std::string CompiledRawR2CFourStepRealInHalfOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CFourStepRealInHalfOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CFourStepRealInHalfOutNode::execute(adaptor::DevicePtr input,
                                                               adaptor::DevicePtr output,
                                                               const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;

    std::vector<JitKernelArg> row_args =
        raw_distance_col_kernel_args({input, stage1.get()}, row_tables, input_distance, context.batch);
    row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

    std::vector<JitKernelArg> col_args =
        raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, output_distance, context.batch);
    col_kernel->launch(context.stream,
                       col_args,
                       ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                       context.batch,
                       1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CFourStepRealInHalfOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RNode::CompiledRawC2RNode(int64_t length,
                                       std::shared_ptr<JitKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<JitKernel> pack_kernel,
                                       DeviceAllocation full_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      full_input(std::move(full_input)),
      full_output(std::move(full_output)) {
}

std::string CompiledRawC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2R(n=" << length
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RNode::execute(adaptor::DevicePtr input,
                                          adaptor::DevicePtr output,
                                          const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(full_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    flagfftResult result = fft->execute(full_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(full_output.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(length, block), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RLeafNode::CompiledRawC2RLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawC2RLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RLeafNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);

    std::vector<JitKernelArg> args;
    args.reserve(2 + tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(input_distance));
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RFourStepRealOutNode::CompiledRawC2RFourStepRealOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> expand_kernel,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation full_input,
    DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      expand_kernel(std::move(expand_kernel)),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      full_input(std::move(full_input)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawC2RFourStepRealOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RFourStepRealOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RFourStepRealOutNode::execute(adaptor::DevicePtr input,
                                                         adaptor::DevicePtr output,
                                                         const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);

    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(full_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    std::vector<JitKernelArg> row_args = raw_kernel_args({full_input.get(), stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

    std::vector<JitKernelArg> col_args =
        raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, output_distance, context.batch);
    col_kernel->launch(context.stream,
                       col_args,
                       ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                       context.batch,
                       1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RFourStepRealOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RFourStepCompactInRealOutNode::CompiledRawC2RFourStepCompactInRealOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
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
      stage1(std::move(stage1)) {
}

std::string CompiledRawC2RFourStepCompactInRealOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RFourStepCompactInRealOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RFourStepCompactInRealOutNode::execute(adaptor::DevicePtr input,
                                                                  adaptor::DevicePtr output,
                                                                  const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);

    std::vector<JitKernelArg> row_args =
        raw_distance_col_kernel_args({input, stage1.get()}, row_tables, input_distance, context.batch);
    row_kernel->launch(context.stream, row_args, n2, context.batch, 1);

    std::vector<JitKernelArg> col_args =
        raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, output_distance, context.batch);
    col_kernel->launch(context.stream,
                       col_args,
                       ceil_div(n1, four_step_col_inner_pack_for(n1, n2, context.request.input_dtype)),
                       context.batch,
                       1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RFourStepCompactInRealOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

}  // namespace flagfft
