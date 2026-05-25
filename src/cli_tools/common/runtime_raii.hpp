#pragma once

#include "adaptor/adaptor.h"

namespace flagfft::cli {

class DeviceMemory : public adaptor::Memory {
public:
    using adaptor::Memory::Memory;
    void *get() const noexcept { return data(); }
};

using Stream = adaptor::Stream;
using Timer = adaptor::EventTimer;

}  // namespace flagfft::cli
