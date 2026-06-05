#pragma once

#include <string>

#include "flagfft/keys.hpp"

namespace flagfft {

struct Factorization {
  std::vector<int64_t> factors;
  int64_t remainder = 1;
};

struct PlanNode {
  explicit PlanNode(int64_t length, PlanNodeKind kind);
  virtual ~PlanNode();

  virtual std::string describe(int indent = 0) const = 0;

  int64_t length;
  PlanNodeKind kind;
};

struct LeafPlanNode final : PlanNode {
  LeafPlanNode(int64_t length,
               std::vector<int64_t> factors,
               int64_t remainder,
               int64_t lanes,
               int64_t num_warps,
               std::vector<int64_t> generic_radices,
               int64_t smem_size);

  std::string describe(int indent = 0) const override;

  std::vector<int64_t> factors;
  int64_t remainder;
  int64_t lanes;
  int64_t num_warps;
  std::vector<int64_t> generic_radices;
  int64_t smem_size;
};

struct DirectDFTPlanNode final : PlanNode {
  explicit DirectDFTPlanNode(int64_t length);
  std::string describe(int indent = 0) const override;
};

struct StockhamPlanNode final : PlanNode {
  StockhamPlanNode(int64_t length, std::vector<int64_t> factors);
  std::string describe(int indent = 0) const override;

  std::vector<int64_t> factors;
};

struct FourStepPlanNode final : PlanNode {
  FourStepPlanNode(int64_t length, int64_t n1, int64_t n2, PlanNodePtr row, PlanNodePtr col);
  std::string describe(int indent = 0) const override;

  int64_t n1;
  int64_t n2;
  PlanNodePtr row_plan;
  PlanNodePtr col_plan;
};

struct BluesteinPlanNode final : PlanNode {
  BluesteinPlanNode(int64_t length, int64_t conv_length, PlanNodePtr fft_plan);
  std::string describe(int indent = 0) const override;

  int64_t conv_length;
  PlanNodePtr fft_plan;
};

struct RaderPlanNode final : PlanNode {
  RaderPlanNode(int64_t prime, int64_t root, std::vector<int64_t> idx, PlanNodePtr conv_plan);
  std::string describe(int indent = 0) const override;

  int64_t prime;
  int64_t root;
  std::vector<int64_t> idx;
  PlanNodePtr conv_plan;
};

enum class TwoDimStrategy { RTRT };

struct TwoDimPlanNode final : PlanNode {
  TwoDimPlanNode(int64_t n0, int64_t n1, TwoDimStrategy strategy, PlanNodePtr row_plan, PlanNodePtr col_plan);
  std::string describe(int indent = 0) const override;

  int64_t n0;
  int64_t n1;
  TwoDimStrategy strategy;
  PlanNodePtr row_plan;
  PlanNodePtr col_plan;
};

std::string two_dim_strategy_name(TwoDimStrategy strategy);

struct PlanCandidate {
  PlanNodePtr node;
  double cost = 0.0;
  int priority = 99;
};

class PlanBuilder {
 public:
  PlanNodePtr build(int64_t n, const FFTRequest &request);
  double cost_for(int64_t n, const FFTRequest &request);
  int64_t choose_lanes(int64_t n, const std::vector<int64_t> &factors);
  int64_t choose_num_warps(int64_t lanes);
  std::vector<PlanCandidate> build_tune_candidates(int64_t n, int64_t depth);

 private:
  struct RequestContext {
    std::string input_dtype;
    std::string output_dtype;
    int64_t device_index = -1;
    std::string device_arch;
    int64_t max_dynamic_smem_bytes = kDynamicSmemFallbackBytes;

    bool operator==(const RequestContext &other) const;
  };

  RequestContext make_request_context(const FFTRequest &request) const;
  void set_request_context(const FFTRequest &request);
  const RequestContext &request_context() const;
  Factorization factorize_supported_radices(int64_t n);
  std::vector<std::vector<int64_t>> enumerate_supported_factorizations(int64_t n);
  std::vector<int64_t> factorize_or_raise(int64_t n);
  std::vector<int64_t> score_leaf_factorization(int64_t n, const std::vector<int64_t> &factors);
  std::vector<int64_t> select_leaf_factors(int64_t n);
  std::optional<int64_t> leaf_smem_elements(int64_t n,
                                            const std::vector<int64_t> &factors,
                                            const std::string &input_dtype);
  std::optional<int64_t> leaf_smem_bytes(int64_t n,
                                         const std::vector<int64_t> &factors,
                                         const std::string &input_dtype);
  bool should_use_leaf(int64_t n, const std::vector<int64_t> &factors);
  PlanNodePtr make_leaf_plan(int64_t n, const std::vector<int64_t> &factors, int64_t rem = 1);
  double estimate_leaf_warm_cost(int64_t n, const std::vector<int64_t> &factors);
  double estimate_leaf_warm_cost(int64_t n);
  double estimate_direct_dft_cost(int64_t n);
  double four_step_cost(int64_t n1, int64_t n2);
  double bluestein_cost(int64_t n, int64_t conv_length);
  double rader_cost(int64_t n);
  int priority(const PlanNodePtr &node);
  std::vector<int64_t> enumerate_divisors(int64_t n);
  int64_t next_supported_convolution_length(int64_t minimum);
  PlanNodePtr make_bluestein_plan(int64_t n);
  PlanNodePtr make_rader_plan(int64_t n);
  std::vector<PlanCandidate> build_auto_candidates(int64_t n);
  std::vector<PlanCandidate> build_leaf_tune_candidates(int64_t n);
  PlanCandidate select_candidate(const std::vector<PlanCandidate> &candidates);
  PlanNodePtr build_auto_node(int64_t n);
  double cost_for(int64_t n);
  std::vector<PlanCandidate> top_candidates(std::vector<PlanCandidate> candidates, int64_t limit);

  std::optional<RequestContext> request_context_;
  std::unordered_map<int64_t, PlanNodePtr> node_cache_;
  std::unordered_map<int64_t, double> cost_cache_;
  std::unordered_map<int64_t, std::vector<int64_t>> divisor_cache_;
  std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> factorization_cache_;
  std::unordered_map<int64_t, std::vector<int64_t>> best_leaf_factors_cache_;
  std::unordered_map<int64_t, std::vector<PlanCandidate>> tune_candidate_cache_;
};

}  // namespace flagfft
