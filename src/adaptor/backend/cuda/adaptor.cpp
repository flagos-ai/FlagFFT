#include "adaptor/adaptor.h"

#include <cuda.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flagfft::adaptor {
namespace {

void check(CUresult result, const std::string &context) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
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

CUdeviceptr as_device_ptr(DevicePtr ptr) {
    return static_cast<CUdeviceptr>(ptr);
}

CUstream as_stream(StreamHandle stream) {
    return reinterpret_cast<CUstream>(stream);
}

CUevent as_event(void *event) {
    return reinterpret_cast<CUevent>(event);
}

CUdevice ensure_current_context() {
    check(cuInit(0), "cuInit");
    CUcontext context = nullptr;
    check(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    CUdevice device = 0;
    if (context == nullptr) {
        check(cuDeviceGet(&device, 0), "cuDeviceGet");
        check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    } else {
        check(cuCtxGetDevice(&device), "cuCtxGetDevice");
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
    CUdeviceptr ptr = 0;
    check(cuMemAlloc(&ptr, bytes), "cuMemAlloc");
    ptr_ = static_cast<DevicePtr>(ptr);
    bytes_ = bytes;
}

void Memory::reset() {
    if (ptr_ != 0) {
        cuMemFree(as_device_ptr(ptr_));
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
        check(cuMemcpyHtoD(as_device_ptr(ptr_), source, bytes), "cuMemcpyHtoD");
    }
}

void Memory::copy_to_host(void *destination, std::size_t bytes) const {
    if (bytes > bytes_) {
        throw std::runtime_error("device-to-host copy exceeds allocation");
    }
    if (bytes > 0) {
        check(cuMemcpyDtoH(destination, as_device_ptr(ptr_), bytes), "cuMemcpyDtoH");
    }
}

void Memory::copy_from_device(const Memory &source, std::size_t bytes) {
    if (bytes > bytes_ || bytes > source.bytes_) {
        throw std::runtime_error("device-to-device copy exceeds allocation");
    }
    if (bytes > 0) {
        check(cuMemcpyDtoD(as_device_ptr(ptr_), as_device_ptr(source.ptr_), bytes), "cuMemcpyDtoD");
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

Stream::Stream() {
    ensure_current_context();
    CUstream stream = nullptr;
    check(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
    stream_ = reinterpret_cast<StreamHandle>(stream);
}

Stream::~Stream() {
    if (stream_ != nullptr) {
        cuStreamDestroy(as_stream(stream_));
    }
}

StreamHandle Stream::get() const noexcept {
    return stream_;
}

void Stream::sync() {
    check(cuStreamSynchronize(as_stream(stream_)), "cuStreamSynchronize");
}

EventTimer::EventTimer() {
    ensure_current_context();
    CUevent start = nullptr;
    CUevent stop = nullptr;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    try {
        check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
    } catch (...) {
        cuEventDestroy(start);
        throw;
    }
    start_ = reinterpret_cast<void *>(start);
    stop_ = reinterpret_cast<void *>(stop);
}

EventTimer::~EventTimer() {
    if (start_ != nullptr) {
        cuEventDestroy(as_event(start_));
    }
    if (stop_ != nullptr) {
        cuEventDestroy(as_event(stop_));
    }
}

void EventTimer::start(StreamHandle stream) {
    check(cuEventRecord(as_event(start_), as_stream(stream)), "cuEventRecord(start)");
}

void EventTimer::stop(StreamHandle stream) {
    check(cuEventRecord(as_event(stop_), as_stream(stream)), "cuEventRecord(stop)");
}

float EventTimer::elapsed_ms() {
    check(cuEventSynchronize(as_event(stop_)), "cuEventSynchronize(stop)");
    float milliseconds = 0.0F;
    check(cuEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_)), "cuEventElapsedTime");
    return milliseconds;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
    try {
        CUdevice device = ensure_current_context();
        device_index = static_cast<int>(device);
        device_arch = device_architecture(device_index);
        return FLAGFFT_SUCCESS;
    } catch (const std::exception &) {
        return FLAGFFT_INVALID_DEVICE;
    }
}

int device_count() {
    check(cuInit(0), "cuInit");
    int count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    return count;
}

std::string device_architecture(int device_index) {
    check(cuInit(0), "cuInit");
    CUdevice device = 0;
    check(cuDeviceGet(&device, device_index), "cuDeviceGet");
    int major = 0;
    int minor = 0;
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
          "cuDeviceGetAttribute(major)");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
          "cuDeviceGetAttribute(minor)");
    return "sm_" + std::to_string(major) + std::to_string(minor);
}

int64_t max_dynamic_smem_bytes(int device_index) {
    constexpr int64_t fallback = 48 * 1024;
    try {
        check(cuInit(0), "cuInit");
        CUdevice device = 0;
        check(cuDeviceGet(&device, device_index), "cuDeviceGet");
        int value = 0;
        if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                 device) == CUDA_SUCCESS &&
            value > 0) {
            return value;
        }
        check(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device),
              "cuDeviceGetAttribute(shared)");
        return value > 0 ? value : fallback;
    } catch (const std::exception &) {
        return fallback;
    }
}

void synchronize() {
    ensure_current_context();
    check(cuCtxSynchronize(), "cuCtxSynchronize");
}

std::string backend_name() {
    return "cuda";
}

std::string triton_target(const std::string &device_arch) {
    const std::string prefix = "sm_";
    const std::string arch = device_arch.starts_with(prefix)
                                 ? device_arch.substr(prefix.size()) : device_arch;
    return backend_name() + ":" + arch + ":32";
}

}  // namespace flagfft::adaptor
