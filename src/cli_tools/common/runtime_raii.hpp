#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace flagfft::cli {

class DeviceMemory {
public:
    DeviceMemory() = default;
    explicit DeviceMemory(std::size_t bytes);
    ~DeviceMemory();

    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;
    DeviceMemory(DeviceMemory &&other) noexcept;
    DeviceMemory &operator=(DeviceMemory &&other) noexcept;

    void allocate(std::size_t bytes);
    void reset();
    void *get() const noexcept;
    std::size_t size() const noexcept;

private:
    void *ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

class Stream {
public:
    Stream();
    ~Stream();

    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    cudaStream_t get() const noexcept;
    void sync();

private:
    cudaStream_t stream_ = nullptr;
};

class Timer {
public:
    Timer();
    ~Timer();

    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;

    void start(cudaStream_t stream);
    void stop(cudaStream_t stream);
    float elapsed_ms();

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace flagfft::cli
