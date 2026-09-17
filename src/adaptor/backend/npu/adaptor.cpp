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

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstdlib>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

  void check(aclError result, const std::string &context) {
    if (result == ACL_SUCCESS) {
      return;
    }
    std::ostringstream out;
    out << context << " failed (aclError " << static_cast<int>(result) << ")";
    const char *message = aclGetRecentErrMsg();
    if (message != nullptr && *message != '\0') {
      out << ": " << message;
    }
    throw std::runtime_error(out.str());
  }

  aclrtStream as_stream(StreamHandle stream) {
    return reinterpret_cast<aclrtStream>(stream);
  }

  aclrtEvent as_event(void *event) {
    return reinterpret_cast<aclrtEvent>(event);
  }

  void ensure_acl() {
    static std::once_flag init_once;
    static aclError init_result = ACL_SUCCESS;
    std::call_once(init_once, [] { init_result = aclInit(nullptr); });
    if (init_result != ACL_SUCCESS && init_result != ACL_ERROR_REPEAT_INITIALIZE) {
      check(init_result, "aclInit");
    }
  }

  int ensure_current_device() {
    ensure_acl();
    int32_t device = -1;
    aclError result = aclrtGetDevice(&device);
    if (result != ACL_SUCCESS || device < 0) {
      check(aclrtSetDevice(0), "aclrtSetDevice(0)");
      check(aclrtGetDevice(&device), "aclrtGetDevice");
    }
    return static_cast<int>(device);
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
  check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
  ptr_ = reinterpret_cast<DevicePtr>(ptr);
  bytes_ = bytes;
}

void Memory::reset() {
  if (ptr_ != 0) {
    aclrtFree(reinterpret_cast<void *>(ptr_));
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
    check(aclrtMemcpy(data(), bytes_, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
  }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("device-to-host copy exceeds allocation");
  }
  if (bytes > 0) {
    check(aclrtMemcpy(destination, bytes, data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");
  }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
  if (bytes > bytes_ || bytes > source.bytes_) {
    throw std::runtime_error("device-to-device copy exceeds allocation");
  }
  if (bytes > 0) {
    check(aclrtMemcpy(data(), bytes_, source.data(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
          "aclrtMemcpy(D2D)");
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
  aclrtStream copy_stream = as_stream(stream);
  if (stream == nullptr) {
    // A synchronous memcpy is not ordered after kernels queued on the
    // implicit runtime stream. Enqueue the copy on that same stream instead.
    check(aclrtCtxGetCurrentDefaultStream(&copy_stream), "aclrtCtxGetCurrentDefaultStream");
  }
  check(aclrtMemcpyAsync(reinterpret_cast<void *>(destination),
                         bytes,
                         reinterpret_cast<const void *>(source),
                         bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE,
                         copy_stream),
        "aclrtMemcpyAsync(D2D)");
}

Stream::Stream() {
  ensure_current_device();
  aclrtStream stream = nullptr;
  check(aclrtCreateStream(&stream), "aclrtCreateStream");
  stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    aclrtSynchronizeStream(as_stream(stream_));
    aclrtDestroyStream(as_stream(stream_));
  }
}

StreamHandle Stream::get() const noexcept {
  return stream_;
}

void Stream::sync() {
  check(aclrtSynchronizeStream(as_stream(stream_)), "aclrtSynchronizeStream");
}

// Ascend Graph/ACL graph capture is intentionally not used in the first NPU
// backend.  The execution layer catches this exception and keeps the direct
// launch path, which also avoids introducing a graph-specific dependency into
// the plain-Triton MVP.
CudaGraph::~CudaGraph() = default;

void CudaGraph::begin_capture(StreamHandle) {
  throw std::runtime_error("Ascend graph capture is not enabled in the plain-Triton MVP");
}

void CudaGraph::end_capture(StreamHandle) {
  throw std::runtime_error("Ascend graph capture is not enabled in the plain-Triton MVP");
}

void CudaGraph::launch(StreamHandle) {
  throw std::runtime_error("Ascend graph replay is not enabled in the plain-Triton MVP");
}

bool CudaGraph::valid() const noexcept {
  return false;
}

EventTimer::EventTimer() {
  ensure_current_device();
  aclrtEvent start = nullptr;
  aclrtEvent stop = nullptr;
  check(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
  try {
    check(aclrtCreateEvent(&stop), "aclrtCreateEvent(stop)");
  } catch (...) {
    aclrtDestroyEvent(start);
    throw;
  }
  start_ = reinterpret_cast<void *>(start);
  stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
  if (start_ != nullptr) {
    aclrtDestroyEvent(as_event(start_));
  }
  if (stop_ != nullptr) {
    aclrtDestroyEvent(as_event(stop_));
  }
}

void EventTimer::start(StreamHandle stream) {
  check(aclrtRecordEvent(as_event(start_), as_stream(stream)), "aclrtRecordEvent(start)");
}

void EventTimer::stop(StreamHandle stream) {
  check(aclrtRecordEvent(as_event(stop_), as_stream(stream)), "aclrtRecordEvent(stop)");
}

float EventTimer::elapsed_ms() {
  check(aclrtSynchronizeEvent(as_event(stop_)), "aclrtSynchronizeEvent(stop)");
  float milliseconds = 0.0F;
  check(aclrtEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)),
        "aclrtEventElapsedTime");
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
  ensure_acl();
  uint32_t count = 0;
  check(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
  return static_cast<int>(count);
}

std::string device_architecture(int device_index) {
  ensure_acl();
  check(aclrtSetDevice(device_index), "aclrtSetDevice");
  const char *soc_name = aclrtGetSocName();
  if (soc_name != nullptr && *soc_name != '\0') {
    return std::string(soc_name);
  }
  const char *override_arch = std::getenv("FLAGFFT_NPU_ARCH");
  if (override_arch != nullptr && *override_arch != '\0') {
    return std::string(override_arch);
  }
  // The current baai-ascend host is an Ascend 910B4 system.  Keep a usable
  // cache/target key if an older ACL runtime does not expose aclrtGetSocName.
  return "Ascend910B4";
}

int64_t max_dynamic_smem_bytes(int) {
  // The initial route is direct DFT and does not use shared memory.  This
  // conservative 192 KiB value also keeps the generic planner from selecting
  // an oversized TLE leaf before the NPU planner gate is expanded.
  return 192 * 1024;
}

int64_t max_launch_blocks() {
  // ACL's rtKernelLaunch accepts blockDim in [1, 65535]; exec nodes split
  // larger grids into chunked launches.
  return 65535;
}

std::string device_capabilities_json() {
  int index = 0;
  std::string arch;
  if (ensure_device(index, arch) != FLAGFFT_SUCCESS)
    throw std::runtime_error("cannot query current device");
  // libtriton_jit's NPU backend uses WARP_SIZE = 1, so a kernel's num_warps is
  // the literal block dimension. The ACL block dimension is far above the
  // 1..8 warps the planner emits; the shared-memory figures reuse the value
  // the planner already gates NPU leaf selection on.
  constexpr int64_t kMaxThreadsPerBlock = 65535;
  nlohmann::json result = {
      {"schema_version", 1}, {"source", "backend_default"},
      {"backend", backend_name()}, {"device_index", index},
      {"device_name", "Ascend NPU"}, {"device_arch", arch},
      {"warp_size", 1}, {"max_threads_per_block", kMaxThreadsPerBlock},
      {"shared_memory_per_block", max_dynamic_smem_bytes(index)},
      {"max_dynamic_shared_memory", max_dynamic_smem_bytes(index)},
      {"multiprocessor_count", nullptr}};
  return result.dump();
}

void synchronize() {
  ensure_current_device();
  check(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
}

std::string backend_name() {
  return "npu";
}

std::string triton_target(const std::string &device_arch) {
  return backend_name() + ":" + device_arch + ":1";
}

}  // namespace flagfft::adaptor
