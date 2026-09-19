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
  bool input_from_stdin = false;
  bool output_to_stdout = false;
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
               "API is one of c2c, z2z, r2c, d2z, c2r, z2d.\n"
               "\n"
               "Passing `-` as --input reads the input from stdin and passing `-` as\n"
               "--output-dir writes the result to stdout, so a caller can compare a\n"
               "transform without materializing either side on disk.  In that mode the\n"
               "plan description goes to stderr, wrapped in FLAGFFT PLAN BEGIN/END\n"
               "delimiters, instead of to flagfft_plan.txt.\n";
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
  spec.input_from_stdin = spec.input == "-";
  spec.output_to_stdout = spec.output_dir == "-";
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
  if (spec.input_from_stdin && spec.implementation == Implementation::kBoth) {
    // stdin holds a single copy of the input, so it cannot be replayed for a
    // second library.  The runner always selects one library per invocation.
    throw std::runtime_error("--input=- requires --implementation=flagfft or platform");
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

std::string describe_input(const Spec& spec) {
  return spec.input_from_stdin ? std::string("stdin") : spec.input.string();
}

std::string describe_output(const Spec& spec) {
  return spec.output_to_stdout ? std::string("stdout") : spec.output_dir.string();
}

// A regular file delivers a whole chunk at once, but a pipe hands over
// whatever has arrived so far.  Both modes know the exact expected size, so
// loop until it is satisfied; any premature end is a hard error.
std::vector<std::uint8_t> read_input_chunk(std::istream& input, const Spec& spec, std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  std::size_t done = 0;
  while (done < bytes) {
    input.read(reinterpret_cast<char*>(result.data()) + done, static_cast<std::streamsize>(bytes - done));
    const std::streamsize got = input.gcount();
    if (got <= 0) {
      throw std::runtime_error("failed to read input chunk from " + describe_input(spec) + ": got " +
                               std::to_string(done) + " of " + std::to_string(bytes) + " bytes");
    }
    done += static_cast<std::size_t>(got);
  }
  return result;
}

void write_output_chunk(std::ostream& output, const Spec& spec, const void* data, std::size_t bytes) {
  // ostream::write is specified to emit every byte, including on a pipe.
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!output) {
    throw std::runtime_error("failed to write output chunk to " + describe_output(spec));
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

// In stdout mode the plan cannot be a sibling file, so it is emitted on stderr
// between these delimiters.  The runner extracts the text for the JSON and it
// also reaches the per-operator log, where it stays human-readable.
constexpr const char* kPlanBegin = "===== FLAGFFT PLAN BEGIN =====";
constexpr const char* kPlanEnd = "===== FLAGFFT PLAN END =====";

void write_plan_description(flagfftHandle plan, const Spec& spec) {
  const char* description = flagfftGetPlanDescription(plan);
  if (description == nullptr) {
    return;
  }
  if (spec.output_to_stdout) {
    std::cerr << kPlanBegin << '\n' << description << '\n' << kPlanEnd << '\n';
    std::cerr.flush();
    return;
  }
  const fs::path path = spec.output_dir / "flagfft_plan.txt";
  std::ofstream output(path, std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open plan description: " + path.string());
  }
  output << description;
}

void run_implementation(const Spec& spec, Implementation implementation) {
  const Layout full_layout = make_layout(spec);
  if (!spec.input_from_stdin) {
    validate_file_size(spec.input, full_layout.input_bytes);
  }
  const int batch_chunk = chunk_batch_size(spec, full_layout);
  const fs::path output_path =
      spec.output_dir / (implementation == Implementation::kFlagFFT ? "flagfft.bin" : "platform.bin");

  std::ifstream input_file;
  std::istream* input = &std::cin;
  if (!spec.input_from_stdin) {
    input_file.open(spec.input, std::ios::binary);
    if (!input_file.is_open()) {
      throw std::runtime_error("cannot open input file: " + spec.input.string());
    }
    input = &input_file;
  }
  std::ofstream output_file;
  std::ostream* output = &std::cout;
  if (!spec.output_to_stdout) {
    output_file.open(output_path, std::ios::binary | std::ios::trunc);
    if (!output_file.is_open()) {
      throw std::runtime_error("cannot open output file: " + output_path.string());
    }
    output = &output_file;
  }

  std::size_t input_offset = 0;
  std::size_t output_offset = 0;
  for (int batch_start = 0; batch_start < spec.batch; batch_start += batch_chunk) {
    const int current_batch = std::min(batch_chunk, spec.batch - batch_start);
    Spec chunk_spec = spec;
    chunk_spec.batch = current_batch;
    const Layout chunk_layout = make_layout(chunk_spec);
    std::vector<std::uint8_t> host_input = read_input_chunk(*input, spec, chunk_layout.input_bytes);
    if (!spec.input_from_stdin) {
      release_file_cache(spec.input, input_offset, chunk_layout.input_bytes, false);
    }
    input_offset += chunk_layout.input_bytes;

    std::vector<std::uint8_t> host_output;

    if (implementation == Implementation::kPlatform && flagfft::test_adaptor::reference_uses_host_memory()) {
      // The reference library exposes host pointers and performs its own
      // device transfers, so no device staging buffers are allocated here.
      host_output.resize(chunk_layout.output_bytes);
      Stream stream;
      RefPlanHandle reference_plan = make_reference_plan(chunk_spec);
      flagfft::test_adaptor::ref_set_stream(reference_plan, stream.get());
      execute_reference(reference_plan, chunk_spec, chunk_layout, host_input.data(), host_output.data());
      stream.sync();
    } else {
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
        write_plan_description(flag_plan.handle, spec);
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
        write_plan_description(flag_plan.handle, spec);
      }
    }

    write_output_chunk(*output, spec, host_output.data(), host_output.size());
    output->flush();
    if (!*output) {
      throw std::runtime_error("failed to flush output chunk to " + describe_output(spec));
    }
    if (!spec.output_to_stdout) {
      release_file_cache(output_path, output_offset, host_output.size(), true);
    }
    output_offset += host_output.size();
  }

  output->flush();
  if (!*output) {
    throw std::runtime_error("failed to flush output to " + describe_output(spec));
  }
  if (!spec.output_to_stdout) {
    output_file.close();
    validate_file_size(output_path, full_layout.output_bytes);
  }
}

int run(const Spec& spec) {
  if (!spec.output_to_stdout) {
    fs::create_directories(spec.output_dir);
  }
  if (spec.implementation != Implementation::kPlatform) {
    run_implementation(spec, Implementation::kFlagFFT);
  }
  if (spec.implementation != Implementation::kFlagFFT) {
    run_implementation(spec, Implementation::kPlatform);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    const int status = run(parse_spec(args));
    std::cout.flush();
    return status;
  } catch (const std::exception& error) {
    std::cout.flush();
    std::cerr << "numpy_fft_capture: " << error.what() << '\n';
    return 2;
  }
}
