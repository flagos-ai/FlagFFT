#include "runtime_raii.hpp"

#include <stdexcept>
#include <sstream>

namespace flagfft::cli {

namespace {

void check_cuda_impl(cudaError_t result, const std::string &context) {
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << context << ": CUDA " << static_cast<int>(result)
            << " (" << cudaGetErrorString(result) << ")";
        throw std::runtime_error(oss.str());
    }
}

}  // namespace

DeviceMemory::DeviceMemory(std::size_t bytes) {
    allocate(bytes);
}

DeviceMemory::~DeviceMemory() {
    reset();
}

DeviceMemory::DeviceMemory(DeviceMemory &&other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

DeviceMemory &DeviceMemory::operator=(DeviceMemory &&other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void DeviceMemory::allocate(std::size_t bytes) {
    reset();
    if (bytes == 0) {
        return;
    }
    check_cuda_impl(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    bytes_ = bytes;
}

void DeviceMemory::reset() {
    if (ptr_ != nullptr) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
}

void *DeviceMemory::get() const noexcept {
    return ptr_;
}

std::size_t DeviceMemory::size() const noexcept {
    return bytes_;
}

Stream::Stream() {
    check_cuda_impl(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

Stream::~Stream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

cudaStream_t Stream::get() const noexcept {
    return stream_;
}

void Stream::sync() {
    check_cuda_impl(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

Timer::Timer() {
    check_cuda_impl(cudaEventCreate(&start_), "cudaEventCreate(start)");
    check_cuda_impl(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
}

Timer::~Timer() {
    if (start_ != nullptr) {
        cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
    }
}

void Timer::start(cudaStream_t stream) {
    check_cuda_impl(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
}

void Timer::stop(cudaStream_t stream) {
    check_cuda_impl(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
}

float Timer::elapsed_ms() {
    check_cuda_impl(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
    float ms = 0.0f;
    check_cuda_impl(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
    return ms;
}

}  // namespace flagfft::cli
