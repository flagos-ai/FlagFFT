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

#include <mcr/mc_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

  void check(mcError_t result, const std::string &context) {
    if (result == mcSuccess) {
      return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    name = mcGetErrorName(result);
    message = mcGetErrorString(result);
    std::ostringstream out;
    out << context << " failed";
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

  mcStream_t as_stream(StreamHandle stream) {
    return reinterpret_cast<mcStream_t>(stream);
  }

  mcEvent_t as_event(void *event) {
    return reinterpret_cast<mcEvent_t>(event);
  }

  mcGraph_t as_graph(void *graph) {
    return reinterpret_cast<mcGraph_t>(graph);
  }

  mcGraphExec_t as_graph_exec(void *exec) {
    return reinterpret_cast<mcGraphExec_t>(exec);
  }

  int ensure_current_context() {
    int device = 0;
    check(mcGetDevice(&device), "mcGetDevice");
    mcCtx_t context = nullptr;
    check(mcCtxGetCurrent(&context), "mcCtxGetCurrent");
    if (context == nullptr) {
      check(mcDevicePrimaryCtxRetain(&context, device), "mcDevicePrimaryCtxRetain");
      check(mcCtxSetCurrent(context), "mcCtxSetCurrent");
    }
    return device;
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
  ensure_current_context();
  void *ptr = nullptr;
  check(mcMalloc(&ptr, bytes), "mcMalloc");
  ptr_ = reinterpret_cast<DevicePtr>(ptr);
  bytes_ = bytes;
}

void Memory::reset() {
  if (ptr_ != 0) {
    mcFree(as_device_ptr(ptr_));
    ptr_ = 0;
    bytes_ = 0;
  }
}

DevicePtr Memory::get() const noexcept {
  return ptr_;
}

void *Memory::data() const noexcept {
  return reinterpret_cast<void *>(ptr_);
}

std::size_t Memory::size() const noexcept {
  return bytes_;
}

void Memory::copy_from_host(const void *source, std::size_t bytes) {
  if (bytes > bytes_) {
    throw std::runtime_error("host-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(mcMemcpy(as_device_ptr(ptr_), source, bytes, mcMemcpyHostToDevice), "mcMemcpy(HtoD)");
  }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("device-to-host copy exceeds allocation");
  }
  if (bytes > 0) {
    check(mcMemcpy(destination, as_device_ptr(ptr_), bytes, mcMemcpyDeviceToHost), "mcMemcpy(DtoH)");
  }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
  if (bytes > bytes_ || bytes > source.bytes_) {
    throw std::runtime_error("device-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(mcMemcpy(as_device_ptr(ptr_), as_device_ptr(source.ptr_), bytes, mcMemcpyDeviceToDevice),
          "mcMemcpy(DtoD)");
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

void copy_device_to_device(DevicePtr destination, DevicePtr source, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) {
    return;
  }
  ensure_current_context();
  check(mcMemcpyAsync(as_device_ptr(destination), as_device_ptr(source), bytes,
                      mcMemcpyDeviceToDevice, as_stream(stream)), "mcMemcpyAsync(DtoD)");
}

Stream::Stream() {
  ensure_current_context();
  mcStream_t stream = nullptr;
  check(mcStreamCreate(&stream), "mcStreamCreate");
  stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    mcStreamDestroy(as_stream(stream_));
  }
}

StreamHandle Stream::get() const noexcept {
  return stream_;
}

void Stream::sync() {
  check(mcStreamSynchronize(as_stream(stream_)), "mcStreamSynchronize");
}

CudaGraph::~CudaGraph() {
  if (exec_ != nullptr) {
    mcGraphExecDestroy(as_graph_exec(exec_));
  }
  if (graph_ != nullptr) {
    mcGraphDestroy(as_graph(graph_));
  }
}

void CudaGraph::begin_capture(StreamHandle stream) {
  ensure_current_context();
  check(mcStreamBeginCapture(as_stream(stream), mcStreamCaptureModeRelaxed), "mcStreamBeginCapture");
}

void CudaGraph::end_capture(StreamHandle stream) {
  ensure_current_context();
  mcGraph_t graph = nullptr;
  check(mcStreamEndCapture(as_stream(stream), &graph), "mcStreamEndCapture");
  graph_ = reinterpret_cast<void *>(graph);
  mcGraphExec_t exec = nullptr;
  check(mcGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "mcGraphInstantiate");
  if (exec_ != nullptr) {
    mcGraphExecDestroy(as_graph_exec(exec_));
  }
  exec_ = reinterpret_cast<void *>(exec);
  check(mcGraphDestroy(graph), "mcGraphDestroy");
  graph_ = nullptr;
}

void CudaGraph::launch(StreamHandle stream) {
  ensure_current_context();
  if (exec_ == nullptr) {
    throw std::runtime_error("CudaGraph::launch called before end_capture");
  }
  check(mcGraphLaunch(as_graph_exec(exec_), as_stream(stream)), "mcGraphLaunch");
}

bool CudaGraph::valid() const noexcept {
  return exec_ != nullptr;
}

EventTimer::EventTimer() {
  ensure_current_context();
  mcEvent_t start = nullptr;
  mcEvent_t stop = nullptr;
  check(mcEventCreate(&start), "mcEventCreate(start)");
  try {
    check(mcEventCreate(&stop), "mcEventCreate(stop)");
  } catch (...) {
    mcEventDestroy(start);
    throw;
  }
  start_ = reinterpret_cast<void *>(start);
  stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
  if (start_ != nullptr) {
    mcEventDestroy(as_event(start_));
  }
  if (stop_ != nullptr) {
    mcEventDestroy(as_event(stop_));
  }
}

void EventTimer::start(StreamHandle stream) {
  check(mcEventRecord(as_event(start_), as_stream(stream)), "mcEventRecord(start)");
}

void EventTimer::stop(StreamHandle stream) {
  check(mcEventRecord(as_event(stop_), as_stream(stream)), "mcEventRecord(stop)");
}

float EventTimer::elapsed_ms() {
  check(mcEventSynchronize(as_event(stop_)), "mcEventSynchronize(stop)");
  float milliseconds = 0.0F;
  check(mcEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)), "mcEventElapsedTime");
  return milliseconds;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
  try {
    int device = ensure_current_context();
    device_index = static_cast<int>(device);
    device_arch = device_architecture(device_index);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &) {
    return FLAGFFT_INVALID_DEVICE;
  }
}

int device_count() {
  int count = 0;
  check(mcGetDeviceCount(&count), "mcGetDeviceCount");
  return count;
}

std::string device_architecture(int device_index) {
  mcDeviceProp_t properties{};
  check(mcGetDeviceProperties(&properties, device_index), "mcGetDeviceProperties");
  return std::to_string(properties.major) + std::to_string(properties.minor);
}

int64_t max_dynamic_smem_bytes(int device_index) {
  mcDeviceProp_t properties{};
  check(mcGetDeviceProperties(&properties, device_index), "mcGetDeviceProperties");
  return static_cast<int64_t>(std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin));
}

void synchronize() {
  ensure_current_context();
  check(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

std::string backend_name() {
  return "maca";
}

std::string triton_target(const std::string &device_arch) {
  // Native C550 capability is 10.2; MetaX Torch/Triton expose the CUDA
  // compatibility target 8.0. Keep the cache key aligned with the compiler.
  (void)device_arch;
  return backend_name() + ":80:64";
}

}  // namespace flagfft::adaptor
