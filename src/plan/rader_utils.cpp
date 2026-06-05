#include "rader_utils.hpp"

#include <stdexcept>

namespace flagfft {
namespace {

  int64_t modpow(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
    base %= mod;
    while (exp > 0) {
      if ((exp & 1) != 0) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exp >>= 1;
    }
    return result;
  }

  std::vector<int64_t> prime_factors_unique(int64_t n) {
    std::vector<int64_t> factors;
    for (int64_t factor = 2; factor * factor <= n; ++factor) {
      if (n % factor != 0) {
        continue;
      }
      factors.push_back(factor);
      while (n % factor == 0) {
        n /= factor;
      }
    }
    if (n > 1) {
      factors.push_back(n);
    }
    return factors;
  }

}  // namespace

bool is_prime_length(int64_t n) {
  if (n < 2) {
    return false;
  }
  if (n % 2 == 0) {
    return n == 2;
  }
  for (int64_t divisor = 3; divisor * divisor <= n; divisor += 2) {
    if (n % divisor == 0) {
      return false;
    }
  }
  return true;
}

int64_t find_primitive_root(int64_t p) {
  if (!is_prime_length(p)) {
    throw std::runtime_error("Rader primitive root requires a prime length");
  }
  if (p == 2) {
    return 1;
  }
  const int64_t phi = p - 1;
  std::vector<int64_t> factors = prime_factors_unique(phi);
  for (int64_t root = 2; root < p; ++root) {
    bool primitive = true;
    for (int64_t factor : factors) {
      if (modpow(root, phi / factor, p) == 1) {
        primitive = false;
        break;
      }
    }
    if (primitive) {
      return root;
    }
  }
  throw std::runtime_error("failed to find primitive root for prime length");
}

std::vector<int64_t> build_rader_index_table(int64_t p, int64_t root) {
  if (p < 2) {
    throw std::runtime_error("Rader index table requires length >= 2");
  }
  std::vector<int64_t> idx(static_cast<std::size_t>(p - 1));
  int64_t value = 1;
  for (int64_t i = 0; i < p - 1; ++i) {
    idx[static_cast<std::size_t>(i)] = value;
    value = (value * root) % p;
  }
  return idx;
}

}  // namespace flagfft
