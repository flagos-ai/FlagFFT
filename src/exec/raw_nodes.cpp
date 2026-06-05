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

CompiledRawRaderNode::CompiledRawRaderNode(int64_t length,
                                           int64_t conv_length,
                                           std::shared_ptr<CompiledRawNode> fft,
                                           std::shared_ptr<JitKernel> prepare_kernel,
                                           std::shared_ptr<JitKernel> pointwise_kernel,
                                           std::shared_ptr<JitKernel> finalize_kernel,
                                           DeviceAllocation idx,
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
      idx(std::move(idx)),
      b_time(std::move(b_time)),
      a_buf(std::move(a_buf)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawRaderNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawRader(n=" << length << ", conv_length=" << conv_length
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->kernel_name : "null")
      << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->kernel_name : "null")
      << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->kernel_name : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

void CompiledRawRaderNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute Rader convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawRaderNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    ensure_b_fft(context);

    std::vector<JitKernelArg> prepare_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(idx.get()),
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
        JitKernelArg::device(input),
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(idx.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    finalize_kernel->launch(context.stream, finalize_args, ceil_div(conv_length, 256), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Rader execute failed: %s\n", e.what());
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

CompiledRaw2DNode::CompiledRaw2DNode(int64_t n0,
                                     int64_t n1,
                                     std::shared_ptr<CompiledRawNode> row_fft,
                                     std::shared_ptr<CompiledRawNode> col_fft,
                                     std::shared_ptr<JitKernel> transpose_fwd,
                                     std::shared_ptr<JitKernel> transpose_inv,
                                     DeviceAllocation temp1,
                                     DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      row_fft(std::move(row_fft)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2D(n0=" << n0 << ", n1=" << n1
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DNode::execute(adaptor::DevicePtr input,
                                         adaptor::DevicePtr output,
                                         const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t total = n0 * n1;
    // Grid tile size for 2D transpose.  Must match the tile_size baked into
    // the compiled transpose kernel — see kernels.py:_build_tiled_transpose_kernel_source
    // (default tile_size=32) and jit_source.py:_emit_tiled_transpose_jit_kernel (ditto).
    constexpr int64_t tile_size = 32;

    // Step 1: Row FFT (input -> temp1)
    // Shape: (batch, n0, n1) -> row FFT along last dimension
    // batch for row FFT = batch * n0
    RawExecutionContext row_context {context.request, context.stream, batch * n0};
    flagfftResult result = row_fft->execute(input, temp1.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 2: Transpose (temp1 -> temp2)
    // Shape: (batch, n0, n1) -> (batch, n1, n0)
    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(temp2.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    // Step 3: Col FFT (temp2 -> temp1)
    // After transpose, shape is (batch, n1, n0)
    // Col FFT along last dimension (n0), batch = batch * n1
    RawExecutionContext col_context {context.request, context.stream, batch * n1};
    result = col_fft->execute(temp2.get(), temp1.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 4: Transpose back (temp1 -> output)
    // Shape: (batch, n1, n0) -> (batch, n0, n1)
    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(n1, tile_size),
                          batch);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DR2CNode::CompiledRaw2DR2CNode(int64_t n0,
                                           int64_t n1,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> row_fft,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           std::shared_ptr<CompiledRawNode> col_fft,
                                           std::shared_ptr<JitKernel> transpose_fwd,
                                           std::shared_ptr<JitKernel> transpose_inv,
                                           DeviceAllocation row_fft_buf,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      expand_kernel(std::move(expand_kernel)),
      row_fft(std::move(row_fft)),
      pack_kernel(std::move(pack_kernel)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      row_fft_buf(std::move(row_fft_buf)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DR2C(n0=" << n0 << ", n1=" << n1
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DR2CNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    constexpr int64_t tile_size = 32;
    const int64_t half_n1 = n1 / 2 + 1;

    // Step 1: Expand real input to complex
    // Input: (batch*n0, n1) real -> row_fft_buf: (batch*n0, n1) complex
    // Each row is processed independently, so total rows = batch * n0
    // input_distance is per-row distance in the input buffer
    const int64_t input_distance = n1;  // Each row in input has n1 real elements
    const int64_t total_rows = batch * n0;
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

    // Step 2: Row C2C FFT
    // row_fft_buf: (batch*n0, n1) complex -> row_fft_buf: (batch*n0, n1) complex (in-place)
    RawExecutionContext row_context {context.request, context.stream, batch * n0};
    flagfftResult result = row_fft->execute(row_fft_buf.get(), row_fft_buf.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Pack half spectrum
    // row_fft_buf: (batch*n0, n1) complex -> output: (batch*n0, n1/2+1) complex
    // Each row is processed independently, so total rows = batch * n0
    // output_distance is per-row distance in the output buffer
    const int64_t output_distance = half_n1;  // Each row in output has half_n1 complex elements
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(half_n1, block), total_rows, 1);

    // Step 4: Transpose (n0, n1/2+1) -> (n1/2+1, n0)
    // transpose_fwd kernel is compiled for (n0, half_n1) -> (half_n1, n0)
    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(output),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    // Step 5: Col C2C FFT
    // temp1: (n1/2+1, n0) complex -> temp2: (n1/2+1, n0) complex
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    result = col_fft->execute(temp1.get(), temp2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: Transpose back (n1/2+1, n0) -> (n0, n1/2+1)
    // transpose_inv kernel is compiled for (half_n1, n0) -> (n0, half_n1)
    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp2.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DC2RNode::CompiledRaw2DC2RNode(int64_t n0,
                                           int64_t n1,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> col_fft,
                                           std::shared_ptr<CompiledRawNode> row_fft,
                                           std::shared_ptr<JitKernel> transpose_fwd,
                                           std::shared_ptr<JitKernel> transpose_inv,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2,
                                           DeviceAllocation temp3)
    : n0(n0),
      n1(n1),
      expand_kernel(std::move(expand_kernel)),
      col_fft(std::move(col_fft)),
      row_fft(std::move(row_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      pack_kernel(std::move(pack_kernel)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      temp3(std::move(temp3)) {
}

std::string CompiledRaw2DC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DC2R(n0=" << n0 << ", n1=" << n1
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DC2RNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    constexpr int64_t tile_size = 32;
    const int64_t half_n1 = n1 / 2 + 1;
    const int64_t total_rows = batch * n0;

    // C2R is the reverse of R2C:
    // 1. Transpose (n0, half_n1) -> (half_n1, n0)
    // 2. Col IFFT along n0 (batch = batch * half_n1)
    // 3. Transpose back (half_n1, n0) -> (n0, half_n1)
    // 4. Expand half-packed -> full Hermitian (n0, half_n1) -> (n0, n1)
    // 5. Row IFFT along n1 (batch = batch * n0)
    // 6. Pack complex -> real

    // Step 1: Transpose (n0, half_n1) -> (half_n1, n0)
    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    // Step 2: Col C2C IFFT along n0 (batch = batch * half_n1)
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    flagfftResult result = col_fft->execute(temp1.get(), temp2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Transpose back (half_n1, n0) -> (n0, half_n1)
    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp2.get()),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);

    // Step 4: Expand half-packed -> full Hermitian
    // temp1: (batch*n0, half_n1) complex -> temp3: (batch*n0, n1) complex
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(temp3.get()),
        JitKernelArg::i64(half_n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

    // Step 5: Row C2C IFFT along n1 (batch = batch * n0, in-place)
    RawExecutionContext row_context {context.request, context.stream, total_rows};
    result = row_fft->execute(temp3.get(), temp3.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: Pack complex -> real
    // temp3: (batch*n0, n1) complex -> output: (batch*n0, n1) real
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(temp3.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(n1, block), total_rows, 1);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

}  // namespace flagfft
