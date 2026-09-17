// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Native output capture for the unified NumPy acceptance runner. It is built
// with the test suite or, optionally, against an existing FlagFFT build.

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "flagfft.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

using flagfft::adaptor::Memory;
using flagfft::adaptor::Stream;
using flagfft::test_adaptor::RefPlanHandle;

enum class Implementation {
  kBoth,
  kFlagFFT,
  kPlatform,
};

struct Spec {
  flagfftType type = FLAGFFT_C2C;
  std::string api;
  std::vector<int> shape;
  int batch = 1;
  int direction = FLAGFFT_FORWARD;
  fs::path input;
  fs::path output_dir;
  Implementation implementation = Implementation::kBoth;
};

struct Layout {
  std::size_t transform_elements = 0;
  std::size_t input_elements_per_transform = 0;
  std::size_t output_elements_per_transform = 0;
  std::size_t scalar_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
};

struct FlagPlan {
  flagfftHandle handle = nullptr;

  ~FlagPlan() {
    if (handle != nullptr) {
      flagfftDestroy(handle);
    }
  }

  FlagPlan() = default;
  FlagPlan(const FlagPlan&) = delete;
  FlagPlan& operator=(const FlagPlan&) = delete;

  FlagPlan(FlagPlan&& other) noexcept : handle(other.handle) {
    other.handle = nullptr;
  }

  FlagPlan& operator=(FlagPlan&& other) noexcept {
    if (this != &other) {
      if (handle != nullptr) {
        flagfftDestroy(handle);
      }
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }
};

void usage() {
  std::cout << "Usage: numpy_fft_capture --api API --shape N[,N[,N]] --batch B "
               "--direction forward|inverse --input INPUT.bin --output-dir DIR "
               "[--implementation both|flagfft|platform]\n"
               "\n"
               "API is one of c2c, z2z, r2c, d2z, c2r, z2d.\n";
}

Implementation parse_implementation(const std::string& value) {
  if (value == "both") return Implementation::kBoth;
  if (value == "flagfft") return Implementation::kFlagFFT;
  if (value == "platform") return Implementation::kPlatform;
  throw std::runtime_error("unknown --implementation: " + value + " (expected both, flagfft, or platform)");
}

std::map<std::string, std::string> parse_arguments(int argc, char** argv) {
  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    }
    if (!arg.starts_with("--")) {
      throw std::runtime_error("unexpected argument: " + arg);
    }
    arg.erase(0, 2);
    const std::size_t equal = arg.find('=');
    if (equal != std::string::npos) {
      values[arg.substr(0, equal)] = arg.substr(equal + 1);
    } else {
      if (i + 1 >= argc) {
        throw std::runtime_error("--" + arg + " requires a value");
      }
      values[arg] = argv[++i];
    }
  }
  return values;
}

int parse_positive(const std::string& name, const std::string& value) {
  std::size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid value for --" + name + ": " + value);
  }
  if (consumed != value.size() || parsed <= 0) {
    throw std::runtime_error("--" + name + " must be a positive integer");
  }
  return parsed;
}

std::vector<int> parse_shape(const std::string& value) {
  std::vector<int> shape;
  std::size_t start = 0;
  while (start <= value.size()) {
    std::size_t end = value.find_first_of("xX,", start);
    const std::string part = value.substr(start, end == std::string::npos ? end : end - start);
    if (part.empty()) {
      throw std::runtime_error("invalid --shape: " + value);
    }
    shape.push_back(parse_positive("shape", part));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("--shape rank must be between 1 and 3");
  }
  return shape;
}

flagfftType parse_api(const std::string& value) {
  if (value == "c2c") return FLAGFFT_C2C;
  if (value == "z2z") return FLAGFFT_Z2Z;
  if (value == "r2c") return FLAGFFT_R2C;
  if (value == "d2z") return FLAGFFT_D2Z;
  if (value == "c2r") return FLAGFFT_C2R;
  if (value == "z2d") return FLAGFFT_Z2D;
  throw std::runtime_error("unknown --api: " + value);
}

bool is_double(flagfftType type) {
  return type == FLAGFFT_Z2Z || type == FLAGFFT_D2Z || type == FLAGFFT_Z2D;
}

bool is_complex(flagfftType type) {
  return type == FLAGFFT_C2C || type == FLAGFFT_Z2Z;
}

bool is_real_forward(flagfftType type) {
  return type == FLAGFFT_R2C || type == FLAGFFT_D2Z;
}

bool is_real_inverse(flagfftType type) {
  return type == FLAGFFT_C2R || type == FLAGFFT_Z2D;
}

std::size_t product(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t {1}, [](std::size_t lhs, int rhs) {
    return lhs * static_cast<std::size_t>(rhs);
  });
}

Layout make_layout(const Spec& spec) {
  const std::size_t transform_elements = product(spec.shape);
  const int last = spec.shape.back();
  const std::size_t half_transform_elements =
      transform_elements / static_cast<std::size_t>(last) * static_cast<std::size_t>(last / 2 + 1);
  const bool input_complex = is_complex(spec.type) || is_real_inverse(spec.type);
  const bool output_complex = is_complex(spec.type) || is_real_forward(spec.type);
  const std::size_t scalar_bytes = is_double(spec.type) ? sizeof(double) : sizeof(float);
  const std::size_t input_elements = is_real_inverse(spec.type)
                                         ? half_transform_elements * 2
                                         : (input_complex ? transform_elements * 2 : transform_elements);
  const std::size_t output_elements = is_real_forward(spec.type)
                                          ? half_transform_elements * 2
                                          : (output_complex ? transform_elements * 2 : transform_elements);

  Layout layout;
  layout.transform_elements = transform_elements;
  layout.input_elements_per_transform = input_elements;
  layout.output_elements_per_transform = output_elements;
  layout.scalar_bytes = scalar_bytes;
  layout.input_bytes = input_elements * static_cast<std::size_t>(spec.batch) * scalar_bytes;
  layout.output_bytes = output_elements * static_cast<std::size_t>(spec.batch) * scalar_bytes;
  return layout;
}

Spec parse_spec(const std::map<std::string, std::string>& args) {
  const auto required = [&](const char* name) -> const std::string& {
    auto it = args.find(name);
    if (it == args.end() || it->second.empty()) {
      throw std::runtime_error(std::string("missing --") + name);
    }
    return it->second;
  };

  Spec spec;
  spec.api = required("api");
  spec.type = parse_api(spec.api);
  spec.shape = parse_shape(required("shape"));
  spec.batch = parse_positive("batch", required("batch"));
  const std::string direction = required("direction");
  if (direction == "forward" || direction == "fwd") {
    spec.direction = FLAGFFT_FORWARD;
  } else if (direction == "inverse" || direction == "inv") {
    spec.direction = FLAGFFT_INVERSE;
  } else {
    throw std::runtime_error("unknown --direction: " + direction);
  }
  spec.input = required("input");
  spec.output_dir = required("output-dir");
  auto implementation = args.find("implementation");
  if (implementation != args.end()) {
    spec.implementation = parse_implementation(implementation->second);
  }

  if (is_real_forward(spec.type) && spec.direction != FLAGFFT_FORWARD) {
    throw std::runtime_error(spec.api + " only supports forward direction");
  }
  if (is_real_inverse(spec.type) && spec.direction != FLAGFFT_INVERSE) {
    throw std::runtime_error(spec.api + " only supports inverse direction");
  }
  if (spec.shape.size() == 3 && spec.batch != 1) {
    throw std::runtime_error("rank-3 capture currently requires --batch 1");
  }
  return spec;
}

// Keep the native capture's host and device allocations bounded.  The
// acceptance runner invokes this executable separately for FlagFFT and the
// platform reference, but a single invocation still used to materialize the
// entire batch in host memory and in two device buffers.  That exceeded the
// 8 GiB MUSA test cgroup for the largest double-complex batch cases.
constexpr std::size_t kMaxChunkBytes = 128ULL * 1024ULL * 1024ULL;

void validate_file_size(const fs::path& path, std::size_t expected) {
  std::error_code error;
  const auto actual = fs::file_size(path, error);
  if (error) {
    throw std::runtime_error("cannot stat file: " + path.string() + ": " + error.message());
  }
  if (actual != expected) {
    throw std::runtime_error("file byte count mismatch: expected " + std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

std::vector<std::uint8_t> read_input_chunk(std::ifstream& input, const fs::path& path, std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  if (bytes == 0) {
    return result;
  }
  input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  if (input.gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error("failed to read input chunk from: " + path.string());
  }
  return result;
}

void write_output_chunk(std::ofstream& output, const fs::path& path, const void* data, std::size_t bytes) {
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!output) {
    throw std::runtime_error("failed to write output chunk to: " + path.string());
  }
}

void release_file_cache(const fs::path& path, std::size_t offset, std::size_t bytes, bool sync_first) {
#if defined(__linux__)
  if (bytes == 0) {
    return;
  }
  const int flags = (sync_first ? O_WRONLY : O_RDONLY) | O_CLOEXEC;
  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("cannot open file for cache release: " + path.string() + ": " +
                             std::strerror(errno));
  }

  int result = 0;
  if (sync_first && ::fsync(fd) != 0) {
    result = errno;
  } else {
    result = ::posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
  }
  const int close_result = ::close(fd);
  if (result != 0) {
    throw std::runtime_error("cannot release file cache for " + path.string() + ": " + std::strerror(result));
  }
  if (close_result != 0) {
    throw std::runtime_error("cannot close file used for cache release: " + path.string() + ": " +
                             std::strerror(errno));
  }
#else
  (void)path;
  (void)offset;
  (void)bytes;
  (void)sync_first;
#endif
}

int chunk_batch_size(const Spec& spec, const Layout& full_layout) {
  if (spec.batch <= 1) {
    return spec.batch;
  }
  const std::size_t batch = static_cast<std::size_t>(spec.batch);
  const std::size_t bytes_per_batch = full_layout.input_bytes / batch + full_layout.output_bytes / batch;
  const std::size_t max_batch =
      bytes_per_batch == 0 ? batch : std::max<std::size_t>(1, kMaxChunkBytes / bytes_per_batch);
  return static_cast<int>(std::min(batch, max_batch));
}

void check_flagfft(flagfftResult result, const std::string& context) {
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error(context +
                             " failed with flagfftResult=" + std::to_string(static_cast<int>(result)));
  }
}

FlagPlan make_flag_plan(const Spec& spec, const Layout& layout) {
  FlagPlan plan;
  if (spec.shape.size() == 1) {
    check_flagfft(flagfftPlan1d(&plan.handle, spec.shape[0], spec.type, spec.batch), "flagfftPlan1d");
  } else if (spec.shape.size() == 2) {
    int n[2] = {spec.shape[0], spec.shape[1]};
    const int full = static_cast<int>(layout.transform_elements);
    const int half = spec.shape[0] * (spec.shape[1] / 2 + 1);
    const int idist = is_real_inverse(spec.type) ? half : full;
    const int odist = is_real_forward(spec.type) ? half : full;
    check_flagfft(
        flagfftPlanMany(&plan.handle, 2, n, nullptr, 1, idist, nullptr, 1, odist, spec.type, spec.batch),
        "flagfftPlanMany(rank=2)");
  } else {
    check_flagfft(flagfftPlan3d(&plan.handle, spec.shape[0], spec.shape[1], spec.shape[2], spec.type),
                  "flagfftPlan3d");
  }
  return plan;
}

RefPlanHandle make_reference_plan(const Spec& spec) {
  RefPlanHandle plan;
  if (spec.shape.size() == 1) {
    flagfft::test_adaptor::ref_plan_1d(plan, spec.shape[0], spec.type, spec.batch);
  } else if (spec.shape.size() == 2) {
    flagfft::test_adaptor::ref_plan_2d(plan, spec.shape[0], spec.shape[1], spec.type);
  } else {
    flagfft::test_adaptor::ref_plan_3d(plan, spec.shape[0], spec.shape[1], spec.shape[2], spec.type);
  }
  return plan;
}

template <typename T>
T* device_offset(void* base, std::size_t bytes) {
  return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) + bytes);
}

void execute_flagfft(flagfftHandle plan, const Spec& spec, void* input, void* output) {
  switch (spec.type) {
    case FLAGFFT_C2C:
      check_flagfft(flagfftExecC2C(plan,
                                   static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftComplex*>(output),
                                   spec.direction),
                    "flagfftExecC2C");
      return;
    case FLAGFFT_Z2Z:
      check_flagfft(flagfftExecZ2Z(plan,
                                   static_cast<flagfftDoubleComplex*>(input),
                                   static_cast<flagfftDoubleComplex*>(output),
                                   spec.direction),
                    "flagfftExecZ2Z");
      return;
    case FLAGFFT_R2C:
      check_flagfft(
          flagfftExecR2C(plan, static_cast<flagfftReal*>(input), static_cast<flagfftComplex*>(output)),
          "flagfftExecR2C");
      return;
    case FLAGFFT_D2Z:
      check_flagfft(flagfftExecD2Z(plan,
                                   static_cast<flagfftDoubleReal*>(input),
                                   static_cast<flagfftDoubleComplex*>(output)),
                    "flagfftExecD2Z");
      return;
    case FLAGFFT_C2R:
      check_flagfft(
          flagfftExecC2R(plan, static_cast<flagfftComplex*>(input), static_cast<flagfftReal*>(output)),
          "flagfftExecC2R");
      return;
    case FLAGFFT_Z2D:
      check_flagfft(flagfftExecZ2D(plan,
                                   static_cast<flagfftDoubleComplex*>(input),
                                   static_cast<flagfftDoubleReal*>(output)),
                    "flagfftExecZ2D");
      return;
  }
  throw std::runtime_error("unsupported FFT type");
}

void execute_reference_one(RefPlanHandle& plan, const Spec& spec, void* input, void* output) {
  switch (spec.type) {
    case FLAGFFT_C2C:
      flagfft::test_adaptor::ref_exec_c2c(plan,
                                          static_cast<flagfftComplex*>(input),
                                          static_cast<flagfftComplex*>(output),
                                          spec.direction);
      return;
    case FLAGFFT_Z2Z:
      flagfft::test_adaptor::ref_exec_z2z(plan,
                                          static_cast<flagfftDoubleComplex*>(input),
                                          static_cast<flagfftDoubleComplex*>(output),
                                          spec.direction);
      return;
    case FLAGFFT_R2C:
      flagfft::test_adaptor::ref_exec_r2c(plan,
                                          static_cast<flagfftReal*>(input),
                                          static_cast<flagfftComplex*>(output));
      return;
    case FLAGFFT_D2Z:
      flagfft::test_adaptor::ref_exec_d2z(plan,
                                          static_cast<flagfftDoubleReal*>(input),
                                          static_cast<flagfftDoubleComplex*>(output));
      return;
    case FLAGFFT_C2R:
      flagfft::test_adaptor::ref_exec_c2r(plan,
                                          static_cast<flagfftComplex*>(input),
                                          static_cast<flagfftReal*>(output));
      return;
    case FLAGFFT_Z2D:
      flagfft::test_adaptor::ref_exec_z2d(plan,
                                          static_cast<flagfftDoubleComplex*>(input),
                                          static_cast<flagfftDoubleReal*>(output));
      return;
  }
  throw std::runtime_error("unsupported FFT type");
}

void execute_reference(
    RefPlanHandle& plan, const Spec& spec, const Layout& layout, void* input, void* output) {
  // ref_plan_2d is intentionally a one-transform plan.  Use the same
  // per-batch execution convention as the existing 2D correctness tests.
  if (spec.shape.size() != 2 || spec.batch == 1) {
    execute_reference_one(plan, spec, input, output);
    return;
  }

  const std::size_t input_stride = layout.input_elements_per_transform * layout.scalar_bytes;
  const std::size_t output_stride = layout.output_elements_per_transform * layout.scalar_bytes;
  for (int batch = 0; batch < spec.batch; ++batch) {
    void* batch_input = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(input) +
                                                static_cast<std::size_t>(batch) * input_stride);
    void* batch_output = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(output) +
                                                 static_cast<std::size_t>(batch) * output_stride);
    execute_reference_one(plan, spec, batch_input, batch_output);
  }
}

void write_plan_description(flagfftHandle plan, const fs::path& path) {
  const char* description = flagfftGetPlanDescription(plan);
  if (description == nullptr) {
    return;
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open plan description: " + path.string());
  }
  output << description;
}

void run_implementation(const Spec& spec, Implementation implementation) {
  const Layout full_layout = make_layout(spec);
  validate_file_size(spec.input, full_layout.input_bytes);
  const int batch_chunk = chunk_batch_size(spec, full_layout);
  const fs::path output_path =
      spec.output_dir / (implementation == Implementation::kFlagFFT ? "flagfft.bin" : "platform.bin");

  std::ifstream input(spec.input, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open input file: " + spec.input.string());
  }
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open output file: " + output_path.string());
  }

  std::size_t input_offset = 0;
  std::size_t output_offset = 0;
  for (int batch_start = 0; batch_start < spec.batch; batch_start += batch_chunk) {
    const int current_batch = std::min(batch_chunk, spec.batch - batch_start);
    Spec chunk_spec = spec;
    chunk_spec.batch = current_batch;
    const Layout chunk_layout = make_layout(chunk_spec);
    std::vector<std::uint8_t> host_input = read_input_chunk(input, spec.input, chunk_layout.input_bytes);
    release_file_cache(spec.input, input_offset, chunk_layout.input_bytes, false);
    input_offset += chunk_layout.input_bytes;

    std::vector<std::uint8_t> host_output;

    {
      Memory device_input(chunk_layout.input_bytes);
      Memory device_output(chunk_layout.output_bytes);
      device_input.copy_from_host(host_input.data(), chunk_layout.input_bytes);
      host_input.clear();
      host_input.shrink_to_fit();

      Stream stream;
      FlagPlan flag_plan;
      std::optional<RefPlanHandle> reference_plan;
      if (implementation == Implementation::kFlagFFT) {
        flag_plan = make_flag_plan(chunk_spec, chunk_layout);
        check_flagfft(flagfftSetStream(flag_plan.handle, stream.get()), "flagfftSetStream");
        // Retain the chosen plan even when execution subsequently fails/hangs.
        // The successful path writes it again with compiled execution details.
        write_plan_description(flag_plan.handle, spec.output_dir / "flagfft_plan.txt");
        execute_flagfft(flag_plan.handle, chunk_spec, device_input.data(), device_output.data());
      } else {
        reference_plan.emplace(make_reference_plan(chunk_spec));
        flagfft::test_adaptor::ref_set_stream(*reference_plan, stream.get());
        execute_reference(*reference_plan,
                          chunk_spec,
                          chunk_layout,
                          device_input.data(),
                          device_output.data());
      }
      stream.sync();

      host_output.resize(chunk_layout.output_bytes);
      device_output.copy_to_host(host_output.data(), chunk_layout.output_bytes);
      if (implementation == Implementation::kFlagFFT) {
        write_plan_description(flag_plan.handle, spec.output_dir / "flagfft_plan.txt");
      }
    }

    write_output_chunk(output, output_path, host_output.data(), host_output.size());
    output.flush();
    if (!output) {
      throw std::runtime_error("failed to flush output chunk: " + output_path.string());
    }
    release_file_cache(output_path, output_offset, host_output.size(), true);
    output_offset += host_output.size();
  }

  output.flush();
  if (!output) {
    throw std::runtime_error("failed to flush output file: " + output_path.string());
  }
  output.close();
  validate_file_size(output_path, full_layout.output_bytes);
}

int run(const Spec& spec) {
  fs::create_directories(spec.output_dir);
  if (spec.implementation != Implementation::kPlatform) {
    run_implementation(spec, Implementation::kFlagFFT);
  }
  if (spec.implementation != Implementation::kFlagFFT) {
    run_implementation(spec, Implementation::kPlatform);
  }

  std::ofstream backend_file(spec.output_dir / "capture_backend.txt", std::ios::trunc);
  if (!backend_file.is_open()) {
    throw std::runtime_error("cannot write capture backend metadata");
  }
  backend_file << flagfft::test_adaptor::backend_name() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    return run(parse_spec(args));
  } catch (const std::exception& error) {
    std::cerr << "numpy_fft_capture: " << error.what() << '\n';
    return 2;
  }
}
