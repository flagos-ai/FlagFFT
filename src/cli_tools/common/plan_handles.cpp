#include "plan_handles.hpp"

namespace flagfft::cli {

FlagfftPlanHandle::FlagfftPlanHandle(flagfftHandle handle) : handle_(handle) {
}

FlagfftPlanHandle::~FlagfftPlanHandle() {
  reset();
}

FlagfftPlanHandle::FlagfftPlanHandle(FlagfftPlanHandle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

FlagfftPlanHandle &FlagfftPlanHandle::operator=(FlagfftPlanHandle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

flagfftHandle FlagfftPlanHandle::get() const noexcept {
  return handle_;
}

flagfftHandle FlagfftPlanHandle::release() noexcept {
  flagfftHandle out = handle_;
  handle_ = nullptr;
  return out;
}

void FlagfftPlanHandle::reset(flagfftHandle handle) {
  if (handle_ != nullptr) {
    flagfftDestroy(handle_);
  }
  handle_ = handle;
}

CufftPlanHandle::CufftPlanHandle(cufftHandle handle) : handle_(handle) {
}

CufftPlanHandle::~CufftPlanHandle() {
  reset();
}

CufftPlanHandle::CufftPlanHandle(CufftPlanHandle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = 0;
}

CufftPlanHandle &CufftPlanHandle::operator=(CufftPlanHandle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

cufftHandle CufftPlanHandle::get() const noexcept {
  return handle_;
}

cufftHandle *CufftPlanHandle::put() {
  reset();
  return &handle_;
}

void CufftPlanHandle::reset(cufftHandle handle) {
  if (handle_ != 0) {
    cufftDestroy(handle_);
  }
  handle_ = handle;
}

}  // namespace flagfft::cli
