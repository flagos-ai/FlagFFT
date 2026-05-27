#include "plan_handles.hpp"

namespace flagfft::cli {

FlagfftPlanHandle::FlagfftPlanHandle(flagfftHandle handle) : handle_(handle) {
}

FlagfftPlanHandle::~FlagfftPlanHandle() {
  reset();
}

FlagfftPlanHandle::FlagfftPlanHandle(FlagfftPlanHandle&& other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

FlagfftPlanHandle& FlagfftPlanHandle::operator=(FlagfftPlanHandle&& other) noexcept {
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

}  // namespace flagfft::cli
