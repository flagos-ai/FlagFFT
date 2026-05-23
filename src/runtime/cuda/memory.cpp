#include "runtime/memory.hpp"

#include <cuda.h>

#include <stdexcept>

#include "runtime/check.hpp"

namespace flagfft::runtime {

Memory::Memory(std::size_t bytes) {
    if (bytes > 0) {
        CUdeviceptr ptr = 0;
        check(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
        ptr_ = ptr;
        bytes_ = bytes;
    }
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

void Memory::reset() {
    if (ptr_ != 0) {
        cuMemFree(ptr_);
        ptr_ = 0;
        bytes_ = 0;
    }
}

DevicePtr Memory::get() const noexcept {
    return ptr_;
}

std::size_t Memory::size() const noexcept {
    return bytes_;
}

Memory Memory::allocate(std::size_t bytes) {
    return Memory(bytes);
}

Memory Memory::from_floats(const std::vector<float> &values) {
    Memory allocation = allocate(values.size() * sizeof(float));
    if (allocation.ptr_ != 0) {
        check(cuMemcpyHtoD(allocation.ptr_, values.data(), allocation.bytes_), "cuMemcpyHtoD");
    }
    return allocation;
}

Memory Memory::from_doubles(const std::vector<double> &values) {
    Memory allocation = allocate(values.size() * sizeof(double));
    if (allocation.ptr_ != 0) {
        check(cuMemcpyHtoD(allocation.ptr_, values.data(), allocation.bytes_), "cuMemcpyHtoD");
    }
    return allocation;
}

}  // namespace flagfft::runtime
