#include "flagfft_test.h"

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace flagfft_test::adaptor {

// =========================================================================
// RefHandle
// =========================================================================

// cufftHandle is int; store as uintptr_t via double-cast.
static cufftHandle to_cufft(uintptr_t v) {
    return static_cast<cufftHandle>(static_cast<intptr_t>(v));
}
static uintptr_t from_cufft(cufftHandle h) {
    return static_cast<uintptr_t>(static_cast<intptr_t>(h));
}

RefHandle::RefHandle() : impl_(0) {
    cufftHandle h;
    if (cufftCreate(&h) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "RefHandle: cufftCreate failed\n");
        return;
    }
    impl_ = from_cufft(h);
}

RefHandle::~RefHandle() {
    if (impl_) cufftDestroy(to_cufft(impl_));
}

RefHandle::RefHandle(RefHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = 0;
}

RefHandle& RefHandle::operator=(RefHandle&& other) noexcept {
    if (this != &other) {
        if (impl_) cufftDestroy(to_cufft(impl_));
        impl_ = other.impl_;
        other.impl_ = 0;
    }
    return *this;
}

uintptr_t RefHandle::get() const { return impl_; }

uintptr_t* RefHandle::ptr() { return &impl_; }

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize() {}
std::string backend_name() { return "cuda"; }

// =========================================================================
// Device memory (CUDA runtime)
// =========================================================================

void* allocate_device(std::size_t bytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        std::fprintf(stderr, "allocate_device: cudaMalloc(%zu) failed\n", bytes);
        return nullptr;
    }
    return ptr;
}

void free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void copy_host_to_device(const void* src, void* dst, std::size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void copy_device_to_host(const void* src, void* dst, std::size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

// =========================================================================
// Helpers — flagfftType to cufftType (same underlying values)
// =========================================================================

static cufftType to_cufft_type(flagfftType type) {
    return static_cast<cufftType>(static_cast<int>(type));
}

// =========================================================================
// Reference plan creation
// =========================================================================

void ref_plan_1d(RefHandle& plan, int nx, flagfftType type, int batch) {
    auto r = cufftPlan1d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx,
                         to_cufft_type(type), batch);
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftPlan1d(nx=" << nx << ", type=" << type << ", batch=" << batch
               << ") failed with code " << r;
    }
}

void ref_plan_2d(RefHandle& plan, int nx, int ny, flagfftType type) {
    auto r = cufftPlan2d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx, ny,
                         to_cufft_type(type));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftPlan2d(nx=" << nx << ", ny=" << ny << ", type=" << type
               << ") failed with code " << r;
    }
}

void ref_plan_3d(RefHandle& plan, int nx, int ny, int nz, flagfftType type) {
    auto r = cufftPlan3d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx, ny, nz,
                         to_cufft_type(type));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftPlan3d(nx=" << nx << ", ny=" << ny << ", nz=" << nz
               << ", type=" << type << ") failed with code " << r;
    }
}

// =========================================================================
// Reference execution
// =========================================================================

void ref_exec_c2c(RefHandle& plan, flagfftComplex* idata, flagfftComplex* odata,
                  int direction) {
    auto r = cufftExecC2C(to_cufft(plan.get()),
                          reinterpret_cast<cufftComplex*>(idata),
                          reinterpret_cast<cufftComplex*>(odata), direction);
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecC2C failed with code " << r;
    }
}

void ref_exec_z2z(RefHandle& plan, flagfftDoubleComplex* idata,
                  flagfftDoubleComplex* odata, int direction) {
    auto r = cufftExecZ2Z(to_cufft(plan.get()),
                          reinterpret_cast<cufftDoubleComplex*>(idata),
                          reinterpret_cast<cufftDoubleComplex*>(odata), direction);
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecZ2Z failed with code " << r;
    }
}

void ref_exec_r2c(RefHandle& plan, flagfftReal* idata, flagfftComplex* odata) {
    auto r = cufftExecR2C(to_cufft(plan.get()),
                          reinterpret_cast<cufftReal*>(idata),
                          reinterpret_cast<cufftComplex*>(odata));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecR2C failed with code " << r;
    }
}

void ref_exec_d2z(RefHandle& plan, flagfftDoubleReal* idata,
                  flagfftDoubleComplex* odata) {
    auto r = cufftExecD2Z(to_cufft(plan.get()),
                          reinterpret_cast<cufftDoubleReal*>(idata),
                          reinterpret_cast<cufftDoubleComplex*>(odata));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecD2Z failed with code " << r;
    }
}

void ref_exec_c2r(RefHandle& plan, flagfftComplex* idata, flagfftReal* odata) {
    auto r = cufftExecC2R(to_cufft(plan.get()),
                          reinterpret_cast<cufftComplex*>(idata),
                          reinterpret_cast<cufftReal*>(odata));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecC2R failed with code " << r;
    }
}

void ref_exec_z2d(RefHandle& plan, flagfftDoubleComplex* idata,
                  flagfftDoubleReal* odata) {
    auto r = cufftExecZ2D(to_cufft(plan.get()),
                          reinterpret_cast<cufftDoubleComplex*>(idata),
                          reinterpret_cast<cufftDoubleReal*>(odata));
    if (r != CUFFT_SUCCESS) {
        FAIL() << "cufftExecZ2D failed with code " << r;
    }
}

}  // namespace flagfft_test::adaptor
