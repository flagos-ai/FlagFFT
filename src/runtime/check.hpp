#pragma once

#include <cuda.h>

#include <string>

namespace flagfft::runtime {

void check(CUresult result, const std::string &context);

}  // namespace flagfft::runtime
