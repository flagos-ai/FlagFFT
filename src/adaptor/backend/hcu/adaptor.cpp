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

#include "adaptor/adaptor.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

void check(hipError_t result, const std::string &context) {
  if (result == hipSuccess) {
    return;
  }
  std::ostringstream out;
  out << context << " failed";
  const char *name = hipGetErrorName(result);
  const char *message = hipGetErrorString(result);
  if (name != nullptr) {
    out << " (" << name << ")";
  }
  if (message != nullptr) {
    out << ": " << message;
  }
  throw std::runtime_error(out.str());
}

void *as_device_ptr(DevicePtr ptr) {
  return reinterpret_cast<void *>(ptr);
}

hipStream_t as_stream(StreamHandle stream) {
  return reinterpret_cast<hipStream_t>(stream);
}

hipEvent_t as_event(void *event) {
  return reinterpret_cast<hipEvent_t>(event);
}

hipGraph_t as_graph(void *graph) {
  return reinterpret_cast<hipGraph_t>(graph);
}

hipGraphExec_t as_graph_exec(void *exec) {
  return reinterpret_cast<hipGraphExec_t>(exec);
}

int ensure_current_device() {
  int device = 0;
  check(hipGetDevice(&device), "hipGetDevice");
  return device;
}

hipDeviceProp_t device_properties(int device_index) {
  hipDeviceProp_t properties {};
  check(hipGetDeviceProperties(&properties, device_index), "hipGetDeviceProperties");
  return properties;
}

std::string architecture_from_properties(const hipDeviceProp_t &properties) {
  std::string arch(properties.gcnArchName);
  const std::size_t suffix = arch.find(':');
  if (suffix != std::string::npos) {
    arch.resize(suffix);
  }
  return arch;
}

}  // namespace

Memory::Memory(std::size_t bytes) {
  allocate(bytes);
}

Memory::~Memory() {
  reset();
}

Memory::Memory(Memory &&other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = 0;
  other.bytes_ = 0;
}

Memory &Memory::operator=(Memory &&other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = 0;
    other.bytes_ = 0;
  }
  return *this;
}

void Memory::allocate(std::size_t bytes) {
  reset();
  if (bytes == 0) {
    return;
  }
  ensure_current_device();
  void *ptr = nullptr;
  check(hipMalloc(&ptr, bytes), "hipMalloc");
  ptr_ = reinterpret_cast<DevicePtr>(ptr);
  bytes_ = bytes;
}

void Memory::reset() {
  if (ptr_ != 0) {
    hipFree(as_device_ptr(ptr_));
    ptr_ = 0;
    bytes_ = 0;
  }
}

DevicePtr Memory::get() const noexcept {
  return ptr_;
}

void *Memory::data() const noexcept {
  return as_device_ptr(ptr_);
}

std::size_t Memory::size() const noexcept {
  return bytes_;
}

void Memory::copy_from_host(const void *source, std::size_t bytes) {
  if (bytes > bytes_) {
    throw std::runtime_error("host-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(hipMemcpy(data(), source, bytes, hipMemcpyHostToDevice), "hipMemcpy(HtoD)");
  }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("device-to-host copy exceeds allocation");
  }
  if (bytes > 0) {
    check(hipMemcpy(destination, data(), bytes, hipMemcpyDeviceToHost), "hipMemcpy(DtoH)");
  }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
  if (bytes > bytes_ || bytes > source.bytes_) {
    throw std::runtime_error("device-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(hipMemcpy(data(), source.data(), bytes, hipMemcpyDeviceToDevice), "hipMemcpy(DtoD)");
  }
}

Memory Memory::from_floats(const std::vector<float> &values) {
  Memory allocation(values.size() * sizeof(float));
  allocation.copy_from_host(values.data(), allocation.bytes_);
  return allocation;
}

Memory Memory::from_doubles(const std::vector<double> &values) {
  Memory allocation(values.size() * sizeof(double));
  allocation.copy_from_host(values.data(), allocation.bytes_);
  return allocation;
}

void copy_device_to_device(DevicePtr destination,
                           DevicePtr source,
                           std::size_t bytes,
                           StreamHandle stream) {
  if (bytes == 0) {
    return;
  }
  ensure_current_device();
  check(hipMemcpyAsync(as_device_ptr(destination),
                       as_device_ptr(source),
                       bytes,
                       hipMemcpyDeviceToDevice,
                       as_stream(stream)),
        "hipMemcpyAsync(DtoD)");
}

Stream::Stream() {
  ensure_current_device();
  hipStream_t stream = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    hipStreamDestroy(as_stream(stream_));
  }
}

StreamHandle Stream::get() const noexcept {
  return stream_;
}

void Stream::sync() {
  check(hipStreamSynchronize(as_stream(stream_)), "hipStreamSynchronize");
}

CudaGraph::~CudaGraph() {
  if (exec_ != nullptr) {
    hipGraphExecDestroy(as_graph_exec(exec_));
  }
  if (graph_ != nullptr) {
    hipGraphDestroy(as_graph(graph_));
  }
}

void CudaGraph::begin_capture(StreamHandle stream) {
  ensure_current_device();
  check(hipStreamBeginCapture(as_stream(stream), hipStreamCaptureModeRelaxed),
        "hipStreamBeginCapture");
}

void CudaGraph::end_capture(StreamHandle stream) {
  ensure_current_device();
  hipGraph_t graph = nullptr;
  check(hipStreamEndCapture(as_stream(stream), &graph), "hipStreamEndCapture");
  graph_ = reinterpret_cast<void *>(graph);
  hipGraphExec_t exec = nullptr;
  check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "hipGraphInstantiate");
  check(hipGraphDestroy(graph), "hipGraphDestroy");
  graph_ = nullptr;
  exec_ = reinterpret_cast<void *>(exec);
}

void CudaGraph::launch(StreamHandle stream) {
  ensure_current_device();
  if (exec_ == nullptr) {
    throw std::runtime_error("CudaGraph::launch called before end_capture");
  }
  check(hipGraphLaunch(as_graph_exec(exec_), as_stream(stream)), "hipGraphLaunch");
}

bool CudaGraph::valid() const noexcept {
  return exec_ != nullptr;
}

EventTimer::EventTimer() {
  ensure_current_device();
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipEventCreate(&start), "hipEventCreate(start)");
  try {
    check(hipEventCreate(&stop), "hipEventCreate(stop)");
  } catch (...) {
    hipEventDestroy(start);
    throw;
  }
  start_ = reinterpret_cast<void *>(start);
  stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
  if (start_ != nullptr) {
    hipEventDestroy(as_event(start_));
  }
  if (stop_ != nullptr) {
    hipEventDestroy(as_event(stop_));
  }
}

void EventTimer::start(StreamHandle stream) {
  check(hipEventRecord(as_event(start_), as_stream(stream)), "hipEventRecord(start)");
}

void EventTimer::stop(StreamHandle stream) {
  check(hipEventRecord(as_event(stop_), as_stream(stream)), "hipEventRecord(stop)");
}

float EventTimer::elapsed_ms() {
  check(hipEventSynchronize(as_event(stop_)), "hipEventSynchronize(stop)");
  float milliseconds = 0.0F;
  check(hipEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)),
        "hipEventElapsedTime");
  return milliseconds;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
  try {
    device_index = ensure_current_device();
    device_arch = device_architecture(device_index);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &) {
    return FLAGFFT_INVALID_DEVICE;
  }
}

int device_count() {
  int count = 0;
  check(hipGetDeviceCount(&count), "hipGetDeviceCount");
  return count;
}

std::string device_architecture(int device_index) {
  return architecture_from_properties(device_properties(device_index));
}

int64_t max_dynamic_smem_bytes(int device_index) {
  constexpr int64_t fallback = 48 * 1024;
  try {
    const hipDeviceProp_t properties = device_properties(device_index);
    const int64_t reported = static_cast<int64_t>(properties.sharedMemPerBlock);
    return reported > 0 ? reported : fallback;
  } catch (const std::exception &) {
    return fallback;
  }
}

int64_t max_launch_blocks() {
  return 2147483647;
}

void synchronize() {
  ensure_current_device();
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

std::string backend_name() {
  return "hcu";
}

std::string triton_target(const std::string &device_arch) {
  std::string arch = device_arch;
  const std::size_t suffix = arch.find(':');
  if (suffix != std::string::npos) {
    arch.resize(suffix);
  }
  return backend_name() + ":" + arch + ":64";
}

std::string device_capabilities_json() {
  int index = 0;
  std::string arch;
  if (ensure_device(index, arch) != FLAGFFT_SUCCESS) {
    throw std::runtime_error("cannot query current device");
  }
  const hipDeviceProp_t properties = device_properties(index);
  nlohmann::json result = {
      {"schema_version", 1},
      {"source", "driver_query"},
      {"backend", backend_name()},
      {"device_index", index},
      {"device_name", properties.name},
      {"device_arch", arch},
      {"warp_size", properties.warpSize},
      {"max_threads_per_block", properties.maxThreadsPerBlock},
      {"shared_memory_per_block", static_cast<int64_t>(properties.sharedMemPerBlock)},
      {"max_dynamic_shared_memory", max_dynamic_smem_bytes(index)},
      {"multiprocessor_count", properties.multiProcessorCount},
  };
  return result.dump();
}

}  // namespace flagfft::adaptor
