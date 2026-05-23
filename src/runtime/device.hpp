#pragma once

#include <cstdint>
#include <string>

#include "flagfft/flagfft.h"

namespace flagfft::runtime {

int64_t max_dynamic_smem_bytes(int device_index);

flagfftResult ensure_device(int &device_index, std::string &device_arch);

}  // namespace flagfft::runtime
