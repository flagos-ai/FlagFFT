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

#include <musa.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

  void check(MUresult result, const std::string &context) {
    if (result == MUSA_SUCCESS) {
      return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    muGetErrorName(result, &name);
    muGetErrorString(result, &message);
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

  MUdeviceptr as_device_ptr(DevicePtr ptr) {
    return static_cast<MUdeviceptr>(ptr);
  }

  MUstream as_stream(StreamHandle stream) {
    return reinterpret_cast<MUstream>(stream);
  }

  MUevent as_event(void *event) {
    return reinterpret_cast<MUevent>(event);
  }

  MUgraph as_graph(void *graph) {
    return reinterpret_cast<MUgraph>(graph);
  }

  MUgraphExec as_graph_exec(void *exec) {
    return reinterpret_cast<MUgraphExec>(exec);
  }

  MUdevice ensure_current_context() {
    check(muInit(0), "muInit");
    MUcontext context = nullptr;
    check(muCtxGetCurrent(&context), "muCtxGetCurrent");
    MUdevice device = 0;
    if (context == nullptr) {
      check(muDeviceGet(&device, 0), "muDeviceGet");
      check(muDevicePrimaryCtxRetain(&context, device), "muDevicePrimaryCtxRetain");
      check(muCtxSetCurrent(context), "muCtxSetCurrent");
    } else {
      check(muCtxGetDevice(&device), "muCtxGetDevice");
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
  MUdeviceptr ptr = 0;
  check(muMemAlloc(&ptr, bytes), "muMemAlloc");
  ptr_ = static_cast<DevicePtr>(ptr);
  bytes_ = bytes;
}

void Memory::reset() {
  if (ptr_ != 0) {
    muMemFree(as_device_ptr(ptr_));
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
    check(muMemcpyHtoD(as_device_ptr(ptr_), source, bytes), "muMemcpyHtoD");
  }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("device-to-host copy exceeds allocation");
  }
  if (bytes > 0) {
    check(muMemcpyDtoH(destination, as_device_ptr(ptr_), bytes), "muMemcpyDtoH");
  }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
  if (bytes > bytes_ || bytes > source.bytes_) {
    throw std::runtime_error("device-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(muMemcpyDtoD(as_device_ptr(ptr_), as_device_ptr(source.ptr_), bytes), "muMemcpyDtoD");
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
  check(muMemcpyDtoDAsync(as_device_ptr(destination), as_device_ptr(source), bytes, as_stream(stream)),
        "muMemcpyDtoDAsync");
}

Stream::Stream() {
  ensure_current_context();
  MUstream stream = nullptr;
  check(muStreamCreate(&stream, MU_STREAM_DEFAULT), "muStreamCreate");
  stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    muStreamDestroy(as_stream(stream_));
  }
}

StreamHandle Stream::get() const noexcept {
  return stream_;
}

void Stream::sync() {
  check(muStreamSynchronize(as_stream(stream_)), "muStreamSynchronize");
}

CudaGraph::~CudaGraph() {
  if (exec_ != nullptr) {
    muGraphExecDestroy(as_graph_exec(exec_));
  }
  if (graph_ != nullptr) {
    muGraphDestroy(as_graph(graph_));
  }
}

void CudaGraph::begin_capture(StreamHandle stream) {
  ensure_current_context();
  check(muStreamBeginCapture(as_stream(stream), MU_STREAM_CAPTURE_MODE_RELAXED), "muStreamBeginCapture");
}

void CudaGraph::end_capture(StreamHandle stream) {
  ensure_current_context();
  MUgraph graph = nullptr;
  check(muStreamEndCapture(as_stream(stream), &graph), "muStreamEndCapture");
  MUgraphExec exec = nullptr;
  check(muGraphInstantiate(&exec, graph, 0), "muGraphInstantiate");
  check(muGraphDestroy(graph), "muGraphDestroy");
  graph_ = nullptr;
  exec_ = reinterpret_cast<void *>(exec);
}

void CudaGraph::launch(StreamHandle stream) {
  ensure_current_context();
  if (exec_ == nullptr) {
    throw std::runtime_error("CudaGraph::launch called before end_capture");
  }
  check(muGraphLaunch(as_graph_exec(exec_), as_stream(stream)), "muGraphLaunch");
}

bool CudaGraph::valid() const noexcept {
  return exec_ != nullptr;
}

EventTimer::EventTimer() {
  ensure_current_context();
  MUevent start = nullptr;
  MUevent stop = nullptr;
  check(muEventCreate(&start, MU_EVENT_DEFAULT), "muEventCreate(start)");
  try {
    check(muEventCreate(&stop, MU_EVENT_DEFAULT), "muEventCreate(stop)");
  } catch (...) {
    muEventDestroy(start);
    throw;
  }
  start_ = reinterpret_cast<void *>(start);
  stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
  if (start_ != nullptr) {
    muEventDestroy(as_event(start_));
  }
  if (stop_ != nullptr) {
    muEventDestroy(as_event(stop_));
  }
}

void EventTimer::start(StreamHandle stream) {
  check(muEventRecord(as_event(start_), as_stream(stream)), "muEventRecord(start)");
}

void EventTimer::stop(StreamHandle stream) {
  check(muEventRecord(as_event(stop_), as_stream(stream)), "muEventRecord(stop)");
}

float EventTimer::elapsed_ms() {
  check(muEventSynchronize(as_event(stop_)), "muEventSynchronize(stop)");
  float milliseconds = 0.0F;
  check(muEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)), "muEventElapsedTime");
  return milliseconds;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
  try {
    MUdevice device = ensure_current_context();
    device_index = static_cast<int>(device);
    device_arch = device_architecture(device_index);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &) {
    return FLAGFFT_INVALID_DEVICE;
  }
}

int device_count() {
  check(muInit(0), "muInit");
  int count = 0;
  check(muDeviceGetCount(&count), "muDeviceGetCount");
  return count;
}

std::string device_architecture(int device_index) {
  check(muInit(0), "muInit");
  MUdevice device = 0;
  check(muDeviceGet(&device, device_index), "muDeviceGet");
  int major = 0;
  int minor = 0;
  check(muDeviceGetAttribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "muDeviceGetAttribute(major)");
  check(muDeviceGetAttribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "muDeviceGetAttribute(minor)");
  // MUSA capability, e.g. (3, 1) -> "31" (matches the FlagTree mtgpu target arch)
  return std::to_string(major) + std::to_string(minor);
}

int64_t max_dynamic_smem_bytes(int device_index) {
  constexpr int64_t fallback = 48 * 1024;
  try {
    check(muInit(0), "muInit");
    MUdevice device = 0;
    check(muDeviceGet(&device, device_index), "muDeviceGet");
    int value = 0;
    if (muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) ==
            MUSA_SUCCESS &&
        value > 0) {
      return value;
    }
    check(muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device),
          "muDeviceGetAttribute(shared)");
    return value > 0 ? value : fallback;
  } catch (const std::exception &) {
    return fallback;
  }
}

void synchronize() {
  ensure_current_context();
  check(muCtxSynchronize(), "muCtxSynchronize");
}

std::string backend_name() {
  return "musa";
}

std::string triton_target(const std::string &device_arch) {
  return backend_name() + ":" + device_arch + ":32";
}

}  // namespace flagfft::adaptor
