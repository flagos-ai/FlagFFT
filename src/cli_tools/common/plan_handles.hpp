#pragma once

#include <cufft.h>

#include "flagfft.h"

namespace flagfft::cli {

class FlagfftPlanHandle {
public:
    FlagfftPlanHandle() = default;
    explicit FlagfftPlanHandle(flagfftHandle handle);
    ~FlagfftPlanHandle();

    FlagfftPlanHandle(const FlagfftPlanHandle &) = delete;
    FlagfftPlanHandle &operator=(const FlagfftPlanHandle &) = delete;
    FlagfftPlanHandle(FlagfftPlanHandle &&other) noexcept;
    FlagfftPlanHandle &operator=(FlagfftPlanHandle &&other) noexcept;

    flagfftHandle get() const noexcept;
    flagfftHandle release() noexcept;
    void reset(flagfftHandle handle = nullptr);

private:
    flagfftHandle handle_ = nullptr;
};

class CufftPlanHandle {
public:
    CufftPlanHandle() = default;
    explicit CufftPlanHandle(cufftHandle handle);
    ~CufftPlanHandle();

    CufftPlanHandle(const CufftPlanHandle &) = delete;
    CufftPlanHandle &operator=(const CufftPlanHandle &) = delete;
    CufftPlanHandle(CufftPlanHandle &&other) noexcept;
    CufftPlanHandle &operator=(CufftPlanHandle &&other) noexcept;

    cufftHandle get() const noexcept;
    cufftHandle *put();
    void reset(cufftHandle handle = 0);

private:
    cufftHandle handle_ = 0;
};

}  // namespace flagfft::cli
