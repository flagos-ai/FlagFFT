#include "adaptor/adaptor.h"
#include "flagfft/core.hpp"

namespace flagfft {

bool PlanBuilder::RequestContext::operator==(const RequestContext &other) const {
  return input_dtype == other.input_dtype && output_dtype == other.output_dtype &&
         device_index == other.device_index && device_arch == other.device_arch &&
         batch == other.batch && max_dynamic_smem_bytes == other.max_dynamic_smem_bytes;
}

PlanBuilder::RequestContext PlanBuilder::make_request_context(const FFTRequest &request) const {
  RequestContext context;
  context.input_dtype = request.input_dtype;
  context.output_dtype = request.output_dtype;
  context.device_index = request.device_index;
  context.device_arch = request.device_arch;
  context.batch = request.batch;
  if (request.device_type == adaptor::backend_name()) {
    context.max_dynamic_smem_bytes = adaptor::max_dynamic_smem_bytes(request.device_index);
  }
  return context;
}

void PlanBuilder::set_request_context(const FFTRequest &request) {
  RequestContext next = make_request_context(request);
  if (request_context_.has_value() && *request_context_ == next) {
    return;
  }
  request_context_ = std::move(next);
  node_cache_.clear();
  cost_cache_.clear();
  tune_candidate_cache_.clear();
}

const PlanBuilder::RequestContext &PlanBuilder::request_context() const {
  if (!request_context_.has_value()) {
    throw std::runtime_error("PlanBuilder request context is not initialized");
  }
  return *request_context_;
}

PlanNodePtr PlanBuilder::build(int64_t n, const FFTRequest &request) {
  set_request_context(request);
  if (n <= 0) {
    throw std::runtime_error("FFT length must be positive");
  }
  return build_auto_node(n);
}

double PlanBuilder::cost_for(int64_t n, const FFTRequest &request) {
  set_request_context(request);
  return cost_for(n);
}

double PlanBuilder::cost_for(int64_t n) {
  auto it = cost_cache_.find(n);
  if (it != cost_cache_.end()) {
    return it->second;
  }
  std::vector<PlanCandidate> candidates = build_auto_candidates(n);
  if (candidates.empty()) {
    throw std::runtime_error("length " + std::to_string(n) + " has no supported FFT implementation route");
  }
  auto best = select_candidate(candidates);
  cost_cache_[n] = best.cost;
  return best.cost;
}

}  // namespace flagfft
