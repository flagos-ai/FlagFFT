#pragma once

#include "flagfft/plan.hpp"

namespace flagfft {

int64_t ceil_div(int64_t numerator, int64_t denominator);
int64_t cuda_device_max_dynamic_shared_memory_bytes(int64_t device_index);

}  // namespace flagfft
