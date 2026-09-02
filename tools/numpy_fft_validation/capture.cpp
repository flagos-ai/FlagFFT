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

// This is deliberately an out-of-tree validation executable.  It links to an
// already-built FlagFFT shared library and compiles the existing platform
// reference adaptor; it is not part of the FlagFFT library or its public API.

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "flagfft.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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
  throw std::runtime_error("unknown --implementation: " + value +
                           " (expected both, flagfft, or platform)");
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

std::vector<std::uint8_t> read_bytes(const fs::path& path, std::size_t expected) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open input file: " + path.string());
  }
  const std::streamoff size = input.tellg();
  if (size < 0 || static_cast<std::size_t>(size) != expected) {
    throw std::runtime_error("input byte count mismatch: expected " + std::to_string(expected) + ", got " +
                             std::to_string(size < 0 ? 0 : static_cast<std::size_t>(size)));
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(expected);
  if (expected > 0) {
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expected));
  }
  if (!input && !input.eof()) {
    throw std::runtime_error("failed to read input file: " + path.string());
  }
  return bytes;
}

void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("cannot open output file: " + path.string());
  }
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!output) {
    throw std::runtime_error("failed to write output file: " + path.string());
  }
}

void check_flagfft(flagfftResult result, const std::string& context) {
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error(context + " failed with flagfftResult=" +
                             std::to_string(static_cast<int>(result)));
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
    check_flagfft(flagfftPlanMany(&plan.handle,
                                  2,
                                  n,
                                  nullptr,
                                  1,
                                  idist,
                                  nullptr,
                                  1,
                                  odist,
                                  spec.type,
                                  spec.batch),
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
      check_flagfft(flagfftExecR2C(plan,
                                   static_cast<flagfftReal*>(input),
                                   static_cast<flagfftComplex*>(output)),
                    "flagfftExecR2C");
      return;
    case FLAGFFT_D2Z:
      check_flagfft(flagfftExecD2Z(plan,
                                   static_cast<flagfftDoubleReal*>(input),
                                   static_cast<flagfftDoubleComplex*>(output)),
                    "flagfftExecD2Z");
      return;
    case FLAGFFT_C2R:
      check_flagfft(flagfftExecC2R(plan,
                                   static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftReal*>(output)),
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

void execute_reference(RefPlanHandle& plan,
                       const Spec& spec,
                       const Layout& layout,
                       void* input,
                       void* output) {
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

int run(const Spec& spec) {
  const Layout layout = make_layout(spec);
  std::vector<std::uint8_t> host_input = read_bytes(spec.input, layout.input_bytes);

  fs::create_directories(spec.output_dir);
  const bool run_flagfft = spec.implementation != Implementation::kPlatform;
  const bool run_platform = spec.implementation != Implementation::kFlagFFT;

  Memory flag_input;
  Memory flag_output;
  Memory reference_input;
  Memory reference_output;
  if (run_flagfft) {
    flag_input.allocate(layout.input_bytes);
    flag_output.allocate(layout.output_bytes);
    flag_input.copy_from_host(host_input.data(), layout.input_bytes);
  }
  if (run_platform) {
    reference_input.allocate(layout.input_bytes);
    reference_output.allocate(layout.output_bytes);
    reference_input.copy_from_host(host_input.data(), layout.input_bytes);
  }

  Stream stream;
  FlagPlan flag_plan;
  std::optional<RefPlanHandle> reference_plan;
  if (run_flagfft) {
    flag_plan = make_flag_plan(spec, layout);
    check_flagfft(flagfftSetStream(flag_plan.handle, stream.get()), "flagfftSetStream");
  }
  if (run_platform) {
    reference_plan.emplace(make_reference_plan(spec));
    flagfft::test_adaptor::ref_set_stream(*reference_plan, stream.get());
  }

  if (run_flagfft) {
    execute_flagfft(flag_plan.handle, spec, flag_input.data(), flag_output.data());
    stream.sync();
  }
  if (run_platform) {
    execute_reference(*reference_plan, spec, layout, reference_input.data(), reference_output.data());
    stream.sync();
  }

  if (run_flagfft) {
    std::vector<std::uint8_t> host_flagfft(layout.output_bytes);
    flag_output.copy_to_host(host_flagfft.data(), layout.output_bytes);
    write_bytes(spec.output_dir / "flagfft.bin", host_flagfft.data(), host_flagfft.size());
    write_plan_description(flag_plan.handle, spec.output_dir / "flagfft_plan.txt");
  }
  if (run_platform) {
    std::vector<std::uint8_t> host_reference(layout.output_bytes);
    reference_output.copy_to_host(host_reference.data(), layout.output_bytes);
    write_bytes(spec.output_dir / "platform.bin", host_reference.data(), host_reference.size());
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
