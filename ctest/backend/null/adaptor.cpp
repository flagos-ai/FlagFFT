#include "flagfft_test.h"

#include <cstdlib>
#include <cstring>

namespace flagfft_test::adaptor {

// =========================================================================
// RefHandle — null backend: no-op
// =========================================================================

RefHandle::RefHandle() : impl_(0) {}
RefHandle::~RefHandle() {}
RefHandle::RefHandle(RefHandle&& other) noexcept : impl_(other.impl_) {
    other.impl_ = 0;
}
RefHandle& RefHandle::operator=(RefHandle&& other) noexcept {
    if (this != &other) {
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
std::string backend_name() { return "null"; }

// =========================================================================
// Device memory — host malloc (no GPU)
// =========================================================================

void* allocate_device(std::size_t bytes) { return std::malloc(bytes); }
void free_device(void* ptr) { std::free(ptr); }
void copy_host_to_device(const void* src, void* dst, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}
void copy_device_to_host(const void* src, void* dst, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

// =========================================================================
// Reference FFT — all no-ops (no reference library available)
// =========================================================================

void ref_plan_1d(RefHandle&, int, flagfftType, int) {}
void ref_plan_2d(RefHandle&, int, int, flagfftType) {}
void ref_plan_3d(RefHandle&, int, int, int, flagfftType) {}
void ref_exec_c2c(RefHandle&, flagfftComplex*, flagfftComplex*, int) {}
void ref_exec_z2z(RefHandle&, flagfftDoubleComplex*, flagfftDoubleComplex*, int) {}
void ref_exec_r2c(RefHandle&, flagfftReal*, flagfftComplex*) {}
void ref_exec_d2z(RefHandle&, flagfftDoubleReal*, flagfftDoubleComplex*) {}
void ref_exec_c2r(RefHandle&, flagfftComplex*, flagfftReal*) {}
void ref_exec_z2d(RefHandle&, flagfftDoubleComplex*, flagfftDoubleReal*) {}

}  // namespace flagfft_test::adaptor
