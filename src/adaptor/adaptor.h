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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flagfft.h"

namespace flagfft::adaptor {

using DevicePtr = std::uintptr_t;
using StreamHandle = flagfftStream_t;

class Memory {
 public:
  Memory() = default;
  explicit Memory(std::size_t bytes);
  ~Memory();

  Memory(const Memory &) = delete;
  Memory &operator=(const Memory &) = delete;
  Memory(Memory &&other) noexcept;
  Memory &operator=(Memory &&other) noexcept;

  void allocate(std::size_t bytes);
  void reset();
  DevicePtr get() const noexcept;
  void *data() const noexcept;
  std::size_t size() const noexcept;

  void copy_from_host(const void *source, std::size_t bytes);
  void copy_to_host(void *destination, std::size_t bytes) const;
  void copy_from_device(const Memory &source, std::size_t bytes);

  static Memory from_floats(const std::vector<float> &values);
  static Memory from_doubles(const std::vector<double> &values);

 private:
  DevicePtr ptr_ = 0;
  std::size_t bytes_ = 0;
};

class Stream {
 public:
  Stream();
  ~Stream();

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  StreamHandle get() const noexcept;
  void sync();

 private:
  StreamHandle stream_ = nullptr;
};

class EventTimer {
 public:
  EventTimer();
  ~EventTimer();

  EventTimer(const EventTimer &) = delete;
  EventTimer &operator=(const EventTimer &) = delete;

  void start(StreamHandle stream);
  void stop(StreamHandle stream);
  float elapsed_ms();

 private:
  void *start_ = nullptr;
  void *stop_ = nullptr;
};

class CudaGraph {
 public:
  CudaGraph() = default;
  ~CudaGraph();

  CudaGraph(const CudaGraph &) = delete;
  CudaGraph &operator=(const CudaGraph &) = delete;

  // Capture subsequent kernel launches on `stream` into a graph and
  // instantiate it.  No host-side work or allocation may happen between
  // begin_capture and end_capture.
  void begin_capture(StreamHandle stream);
  void end_capture(StreamHandle stream);

  void launch(StreamHandle stream);
  bool valid() const noexcept;

 private:
  void *graph_ = nullptr;
  void *exec_ = nullptr;
};

void copy_device_to_device(DevicePtr destination, DevicePtr source, std::size_t bytes, StreamHandle stream);

flagfftResult ensure_device(int &device_index, std::string &device_arch);
int device_count();
std::string device_architecture(int device_index);
int64_t max_dynamic_smem_bytes(int device_index);
void synchronize();
std::string backend_name();
std::string triton_target(const std::string &device_arch);

}  // namespace flagfft::adaptor
