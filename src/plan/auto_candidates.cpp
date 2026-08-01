// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flagfft/core.hpp"

#include "rader_utils.hpp"

namespace flagfft {
namespace {

  constexpr int64_t kBluesteinConvolutionSearchWindow = 4096;
  // vkFFT fixMaxRaderPrimeFFT: primes beyond this use Bluestein, not Rader.
  constexpr int64_t kMaxRaderPrime = 16384;
  // vkFFT fixMaxRadixBluestein=2 (CUDA default): pad Bluestein convolution to a
  // power of two for fp32 while the padded length stays at or below 1M.
  constexpr int64_t kBluesteinPow2ConvMaxLength = 1048576;
  constexpr int64_t kThreadLocalCrossRadix = 32;
  constexpr int64_t kThreadLocalColLength = 1024;
  const std::vector<int64_t> kThreadLocalRegisterRadices = {18, 20, 24, 25, 27, 28, 30, 32};

}  // namespace

std::vector<int64_t> PlanBuilder::enumerate_divisors(int64_t n) {
  auto it = divisor_cache_.find(n);
  if (it != divisor_cache_.end()) {
    return it->second;
  }
  std::vector<int64_t> divisors;
  int64_t root = static_cast<int64_t>(std::sqrt(static_cast<double>(n)));
  for (int64_t divisor = 1; divisor <= root; ++divisor) {
    if (n % divisor != 0) {
      continue;
    }
    divisors.push_back(divisor);
    int64_t mate = n / divisor;
    if (mate != divisor) {
      divisors.push_back(mate);
    }
  }
  std::sort(divisors.begin(), divisors.end());
  divisor_cache_[n] = divisors;
  return divisors;
}

int64_t PlanBuilder::next_supported_convolution_length(int64_t minimum) {
  if (minimum <= 1) {
    return 1;
  }
  int64_t power = ceil_power_of_two(minimum);
  const RequestContext &context = request_context();
  const bool is_fp32 = context.input_dtype == "complex64" || context.input_dtype == "float32";
  if (is_fp32 && power <= kBluesteinPow2ConvMaxLength) {
    // fp32 four-step kernels are measurably faster for power-of-two convolution
    // lengths; this mirrors vkFFT's CUDA default padding to a 2-smooth sequence.
    return power;
  }
  std::vector<int64_t> candidates;
  auto add_supported_candidate = [&](int64_t candidate) {
    if (std::find(candidates.begin(), candidates.end(), candidate) != candidates.end()) {
      return;
    }
    Factorization factorization = factorize_supported_radices(candidate);
    if (factorization.remainder == 1 && !factorization.factors.empty()) {
      candidates.push_back(candidate);
    }
  };

  int64_t local_limit = power;
  if (power - minimum > kBluesteinConvolutionSearchWindow) {
    local_limit = minimum + kBluesteinConvolutionSearchWindow;
  }
  for (int64_t candidate = minimum; candidate <= local_limit; ++candidate) {
    add_supported_candidate(candidate);
  }
  add_supported_candidate(power);

  if (candidates.empty()) {
    return power;
  }

  int64_t best = candidates.front();
  double best_cost = 3.0 * cost_for(best) + static_cast<double>(best);
  for (int64_t candidate : candidates) {
    double candidate_cost = 3.0 * cost_for(candidate) + static_cast<double>(candidate);
    if (candidate_cost < best_cost || (candidate_cost == best_cost && candidate < best)) {
      best = candidate;
      best_cost = candidate_cost;
    }
  }
  return best;
}

PlanNodePtr PlanBuilder::make_bluestein_plan(int64_t n) {
  int64_t conv_length = next_supported_convolution_length(2 * n - 1);
  PlanNodePtr fft_plan = build_auto_node(conv_length);
  return std::make_shared<BluesteinPlanNode>(n, conv_length, std::move(fft_plan));
}

PlanNodePtr PlanBuilder::make_rader_plan(int64_t n) {
  int64_t root = find_primitive_root(n);
  std::vector<int64_t> idx = build_rader_index_table(n, root);
  PlanNodePtr conv_plan = build_auto_node(n - 1);
  return std::make_shared<RaderPlanNode>(n, root, std::move(idx), std::move(conv_plan));
}

std::vector<PlanCandidate> PlanBuilder::build_auto_candidates(int64_t n) {
  if (n <= 0) {
    throw std::runtime_error("FFT length must be positive");
  }

  std::vector<PlanCandidate> candidates;
  Factorization factorization = factorize_supported_radices(n);
  if (factorization.remainder == 1 && !factorization.factors.empty() &&
      should_use_leaf(n, factorization.factors)) {
    PlanNodePtr node = make_leaf_plan(n, select_leaf_factors(n));
    candidates.push_back({node, estimate_leaf_warm_cost(n), priority(node)});
  }

  if (n <= kDirectDftMaxN) {
    PlanNodePtr node = std::make_shared<DirectDFTPlanNode>(n);
    candidates.push_back({node, estimate_direct_dft_cost(n), priority(node)});
  }

  if (n == 16384 && should_use_leaf(64, std::vector<int64_t> {4, 4, 4}) &&
      should_use_leaf(256, std::vector<int64_t> {4, 8, 8})) {
    PlanNodePtr row = make_leaf_plan(64, std::vector<int64_t> {4, 4, 4});
    PlanNodePtr col = make_leaf_plan(256, std::vector<int64_t> {4, 8, 8});
    PlanNodePtr node = std::make_shared<FourStepPlanNode>(n, 64, 256, row, col);
    candidates.push_back({node, four_step_cost(64, 256) * 0.5, priority(node)});
  }

  const RequestContext &context = request_context();
  if (context.input_dtype == "complex64" && context.output_dtype == "complex64" &&
      n % kThreadLocalColLength == 0) {
    const int64_t n1 = n / kThreadLocalColLength;
    const int64_t register_radix = n1 / kThreadLocalCrossRadix;
    if (n1 % kThreadLocalCrossRadix == 0 && contains(kThreadLocalRegisterRadices, register_radix)) {
      auto make_thread_local_leaf = [](int64_t local_register_radix) -> PlanNodePtr {
        const int64_t length = local_register_radix * kThreadLocalCrossRadix;
        const int64_t num_warps = local_register_radix == 32 ? 2 : 1;
        const std::vector<int64_t> generic_radices =
            local_register_radix == 32 ? std::vector<int64_t> {32} : std::vector<int64_t> {};
        return std::make_shared<LeafPlanNode>(
            length,
            std::vector<int64_t> {local_register_radix, kThreadLocalCrossRadix},
            1,
            local_register_radix,
            num_warps,
            generic_radices,
            ceil_power_of_two(length));
      };
      PlanNodePtr node = std::make_shared<FourStepPlanNode>(n,
                                                            n1,
                                                            kThreadLocalColLength,
                                                            make_thread_local_leaf(register_radix),
                                                            make_thread_local_leaf(32));
      candidates.push_back({node, four_step_cost(n1, kThreadLocalColLength) * 0.25, priority(node)});
    }
  }

  for (int64_t n1 : enumerate_divisors(n)) {
    if (n1 <= 1 || n1 >= n) {
      continue;
    }
    int64_t n2 = n / n1;
    try {
      PlanNodePtr row = build_auto_node(n1);
      PlanNodePtr col = build_auto_node(n2);
      PlanNodePtr node = std::make_shared<FourStepPlanNode>(n, n1, n2, row, col);
      double balance = std::abs(std::log(static_cast<double>(n1)) - std::log(static_cast<double>(n2)));
      candidates.push_back({node, four_step_cost(n1, n2) + balance, priority(node)});
    } catch (const std::exception &) {
    }
  }
  if (candidates.empty() && n > kDirectDftMaxN) {
    PlanNodePtr node = make_bluestein_plan(n);
    auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node);
    const double bluestein_candidate_cost = bluestein_cost(n, bluestein->conv_length);
    candidates.push_back({node, bluestein_candidate_cost, priority(node)});
    if (is_prime_length(n) && n <= kMaxRaderPrime) {
      PlanNodePtr rader = make_rader_plan(n);
      double rader_candidate_cost = rader_cost(n);
      const Factorization rader_factorization = factorize_supported_radices(n - 1);
      if (rader_factorization.remainder == 1 && !rader_factorization.factors.empty()) {
        rader_candidate_cost = std::min(rader_candidate_cost, bluestein_candidate_cost * 0.99);
      }
      candidates.push_back({rader, rader_candidate_cost, priority(rader)});
    }
  }
  return candidates;
}

}  // namespace flagfft
