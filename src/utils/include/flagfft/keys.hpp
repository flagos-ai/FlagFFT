#pragma once

#include "flagfft/base.hpp"

namespace flagfft {

struct ProblemKey {
    int64_t fft_length = 0;
    std::vector<int64_t> input_shape;
    std::optional<int64_t> n;
    int64_t requested_n = 0;
    int64_t normalized_dim = -1;
    std::string norm;
    std::string input_dtype;
    std::string output_dtype;
    std::string device_type;
    int64_t device_index = -1;
    std::string device_arch;
    std::vector<int64_t> input_strides;
    std::string input_layout;
    bool requires_contiguous_copy = false;
    std::string direction;

    static ProblemKey from_request(const FFTRequest &request);
    bool operator==(const ProblemKey &other) const;
    std::string repr() const;
};

struct ProblemKeyHash {
    std::size_t operator()(const ProblemKey &key) const;
};

struct PlanKey {
    int64_t schema_version = kPlanSchemaVersion;
    PlanNodeKind root_kind = PlanNodeKind::CtLeaf;
    int64_t length = 0;
    std::vector<int64_t> factors;
    int64_t remainder = 1;
    int64_t lanes = 0;
    int64_t num_warps = 0;
    std::vector<int64_t> generic_radices;
    int64_t smem_size = 0;
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t conv_length = 0;
    std::vector<std::string> child_keys;

    static PlanKey from_node(const PlanNodePtr &node);
    bool operator==(const PlanKey &other) const;
    std::string repr() const;
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey &key) const;
};

struct KernelKey {
    KernelKind kind = KernelKind::Leaf;
    std::string target;
    std::string direction = "forward";
    int64_t length = 0;
    std::vector<int64_t> factors;
    int64_t lanes = 0;
    int64_t num_warps = 0;
    std::vector<int64_t> generic_radices;
    int64_t smem_size = 0;
    int64_t four_step_n1 = 0;
    int64_t four_step_n2 = 0;
    int64_t bluestein_n = 0;
    int64_t bluestein_m = 0;

    static KernelKey leaf(std::string target,
                          std::string direction,
                          int64_t length,
                          std::vector<int64_t> factors,
                          int64_t lanes,
                          int64_t num_warps,
                          std::vector<int64_t> generic_radices,
                          int64_t smem_size);
    static KernelKey four_step_row(std::string target,
                                   std::string direction,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size);
    static KernelKey four_step_col(std::string target,
                                   std::string direction,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size);
    static KernelKey transpose(std::string target);
    static KernelKey twiddle_transpose(std::string target);
    static KernelKey bluestein_prepare(std::string target, int64_t n, int64_t m);
    static KernelKey bluestein_pointwise(std::string target, int64_t n, int64_t m);
    static KernelKey bluestein_finalize(std::string target, int64_t n, int64_t m);
    bool operator==(const KernelKey &other) const;
    std::string repr() const;
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey &key) const;
};

std::string output_dtype_for(const std::string &input_dtype);

}  // namespace flagfft
