#pragma once

#include <cstdint>
#include <vector>

namespace flagfft {

bool is_prime_length(int64_t n);
int64_t find_primitive_root(int64_t p);
std::vector<int64_t> build_rader_index_table(int64_t p, int64_t root);

}  // namespace flagfft
