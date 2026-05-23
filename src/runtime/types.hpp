#pragma once

#include <cuda.h>

namespace flagfft::runtime {

using DevicePtr = CUdeviceptr;
using StreamHandle = CUstream;

}  // namespace flagfft::runtime
