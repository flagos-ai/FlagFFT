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

#include <hggc.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

  void check(HGresult result, const std::string &context) {
    if (result == HGGC_SUCCESS) {
      return;
    }
    const char *message = nullptr;
    hgGetErrorString(result, &message);
    std::ostringstream out;
    out << context << " failed (HGresult " << result << ")";
    if (message != nullptr) {
      out << ": " << message;
    }
    throw std::runtime_error(out.str());
  }

  HGdeviceptr as_device_ptr(DevicePtr ptr) {
    return static_cast<HGdeviceptr>(ptr);
  }

  HGstream as_stream(StreamHandle stream) {
    return reinterpret_cast<HGstream>(stream);
  }

  HGevent as_event(void *event) {
    return reinterpret_cast<HGevent>(event);
  }

  HGgraph as_graph(void *graph) {
    return reinterpret_cast<HGgraph>(graph);
  }

  HGgraphExec as_graph_exec(void *exec) {
    return reinterpret_cast<HGgraphExec>(exec);
  }

  HGdevice ensure_current_context() {
    check(hgInit(0), "hgInit");
    HGcontext context = nullptr;
    check(hgCtxGetCurrent(&context), "hgCtxGetCurrent");
    HGdevice device = 0;
    if (context == nullptr) {
      check(hgDeviceGet(&device, 0), "hgDeviceGet");
      check(hgDevicePrimaryCtxRetain(&context, device), "hgDevicePrimaryCtxRetain");
      check(hgCtxSetCurrent(context), "hgCtxSetCurrent");
    } else {
      check(hgCtxGetDevice(&device), "hgCtxGetDevice");
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
  HGdeviceptr ptr = 0;
  check(hgMemAlloc_v2(&ptr, bytes), "hgMemAlloc_v2");
  ptr_ = static_cast<DevicePtr>(ptr);
  bytes_ = bytes;
}

void Memory::reset() {
  if (ptr_ != 0) {
    hgMemFree_v2(as_device_ptr(ptr_));
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
    check(hgMemcpyHtoD(as_device_ptr(ptr_), source, bytes), "hgMemcpyHtoD");
  }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("device-to-host copy exceeds allocation");
  }
  if (bytes > 0) {
    check(hgMemcpyDtoH(destination, as_device_ptr(ptr_), bytes), "hgMemcpyDtoH");
  }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
  if (bytes > bytes_ || bytes > source.bytes_) {
    throw std::runtime_error("device-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(hgMemcpyDtoD(as_device_ptr(ptr_), as_device_ptr(source.ptr_), bytes), "hgMemcpyDtoD");
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
  check(hgMemcpyDtoDAsync(as_device_ptr(destination), as_device_ptr(source), bytes, as_stream(stream)),
        "hgMemcpyDtoDAsync");
}

Stream::Stream() {
  ensure_current_context();
  HGstream stream = nullptr;
  check(hgStreamCreate(&stream, HG_STREAM_DEFAULT), "hgStreamCreate");
  stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    hgStreamDestroy(as_stream(stream_));
  }
}

StreamHandle Stream::get() const noexcept {
  return stream_;
}

void Stream::sync() {
  check(hgStreamSynchronize(as_stream(stream_)), "hgStreamSynchronize");
}

CudaGraph::~CudaGraph() {
  if (exec_ != nullptr) {
    hgGraphExecDestroy(as_graph_exec(exec_));
  }
  if (graph_ != nullptr) {
    hgGraphDestroy(as_graph(graph_));
  }
}

void CudaGraph::begin_capture(StreamHandle stream) {
  ensure_current_context();
  check(hgStreamBeginCapture(as_stream(stream), HG_STREAM_CAPTURE_MODE_RELAXED), "hgStreamBeginCapture");
}

void CudaGraph::end_capture(StreamHandle stream) {
  ensure_current_context();
  HGgraph graph = nullptr;
  check(hgStreamEndCapture(as_stream(stream), &graph), "hgStreamEndCapture");
  HGgraphExec exec = nullptr;
  check(hgGraphInstantiate(&exec, graph, 0), "hgGraphInstantiate");
  check(hgGraphDestroy(graph), "hgGraphDestroy");
  graph_ = nullptr;
  exec_ = reinterpret_cast<void *>(exec);
}

void CudaGraph::launch(StreamHandle stream) {
  ensure_current_context();
  if (exec_ == nullptr) {
    throw std::runtime_error("CudaGraph::launch called before end_capture");
  }
  check(hgGraphLaunch(as_graph_exec(exec_), as_stream(stream)), "hgGraphLaunch");
}

bool CudaGraph::valid() const noexcept {
  return exec_ != nullptr;
}

EventTimer::EventTimer() {
  ensure_current_context();
  HGevent start = nullptr;
  HGevent stop = nullptr;
  check(hgEventCreate(&start, HG_EVENT_DEFAULT), "hgEventCreate(start)");
  try {
    check(hgEventCreate(&stop, HG_EVENT_DEFAULT), "hgEventCreate(stop)");
  } catch (...) {
    hgEventDestroy(start);
    throw;
  }
  start_ = reinterpret_cast<void *>(start);
  stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
  if (start_ != nullptr) {
    hgEventDestroy(as_event(start_));
  }
  if (stop_ != nullptr) {
    hgEventDestroy(as_event(stop_));
  }
}

void EventTimer::start(StreamHandle stream) {
  check(hgEventRecord(as_event(start_), as_stream(stream)), "hgEventRecord(start)");
}

void EventTimer::stop(StreamHandle stream) {
  check(hgEventRecord(as_event(stop_), as_stream(stream)), "hgEventRecord(stop)");
}

float EventTimer::elapsed_ms() {
  check(hgEventSynchronize(as_event(stop_)), "hgEventSynchronize(stop)");
  float milliseconds = 0.0F;
  check(hgEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)), "hgEventElapsedTime");
  return milliseconds;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
  try {
    HGdevice device = ensure_current_context();
    device_index = static_cast<int>(device);
    device_arch = device_architecture(device_index);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &) {
    return FLAGFFT_INVALID_DEVICE;
  }
}

int device_count() {
  check(hgInit(0), "hgInit");
  int count = 0;
  check(hgDeviceGetCount(&count), "hgDeviceGetCount");
  return count;
}

std::string device_architecture(int device_index) {
  check(hgInit(0), "hgInit");
  HGdevice device = 0;
  check(hgDeviceGet(&device, device_index), "hgDeviceGet");
  int major = 0;
  int minor = 0;
  check(hgDeviceGetAttribute(&major, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "hgDeviceGetAttribute(major)");
  check(hgDeviceGetAttribute(&minor, HG_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "hgDeviceGetAttribute(minor)");
  // PPU capability, e.g. (8, 0) -> "80" (matches the FlagTree ppu target arch)
  return std::to_string(major) + std::to_string(minor);
}

int64_t max_dynamic_smem_bytes(int device_index) {
  constexpr int64_t fallback = 48 * 1024;
  try {
    check(hgInit(0), "hgInit");
    HGdevice device = 0;
    check(hgDeviceGet(&device, device_index), "hgDeviceGet");
    int value = 0;
    if (hgDeviceGetAttribute(&value, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) ==
            HGGC_SUCCESS &&
        value > 0) {
      return value;
    }
    check(hgDeviceGetAttribute(&value, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device),
          "hgDeviceGetAttribute(shared)");
    return value > 0 ? value : fallback;
  } catch (const std::exception &) {
    return fallback;
  }
}

void synchronize() {
  ensure_current_context();
  check(hgCtxSynchronize(), "hgCtxSynchronize");
}

std::string backend_name() {
  return "ppu";
}

std::string triton_target(const std::string &device_arch) {
  return backend_name() + ":" + device_arch + ":32";
}

}  // namespace flagfft::adaptor
