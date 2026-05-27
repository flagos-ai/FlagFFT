#pragma once

#include "flagfft.h"

namespace flagfft::cli {

class FlagfftPlanHandle {
 public:
  FlagfftPlanHandle() = default;
  explicit FlagfftPlanHandle(flagfftHandle handle);
  ~FlagfftPlanHandle();

  FlagfftPlanHandle(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle& operator=(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle(FlagfftPlanHandle&& other) noexcept;
  FlagfftPlanHandle& operator=(FlagfftPlanHandle&& other) noexcept;

  flagfftHandle get() const noexcept;
  flagfftHandle release() noexcept;
  void reset(flagfftHandle handle = nullptr);

 private:
  flagfftHandle handle_ = nullptr;
};

}  // namespace flagfft::cli
