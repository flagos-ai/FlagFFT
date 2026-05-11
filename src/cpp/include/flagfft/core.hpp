#pragma once

#include <Python.h>
#include <cuda.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace flagfft {

inline constexpr int64_t kPlanSchemaVersion = 2;
inline constexpr int64_t kDirectDftMaxN = 64;
inline constexpr int64_t kLeafMaxN = 4096;
inline constexpr int64_t kDynamicSmemFallbackBytes = 48 * 1024;
inline constexpr int64_t kMaxLanes = 128;
inline constexpr double kPi = 3.14159265358979323846264338327950288;
inline constexpr int64_t kFourStepTileRows = 32;
inline constexpr int64_t kFourStepTileCols = 32;
inline constexpr int64_t kLeafTuneTopK = 8;
inline constexpr int64_t kLeafTuneLargeTopK = 16;
inline constexpr int64_t kLeafTuneLargeN = 1024;
inline constexpr int64_t kTuneOrdersPerFactorMultiset = 3;
inline constexpr int64_t kTuneFourStepPairTopK = 8;
inline constexpr int64_t kTuneFourStepCombosPerPair = 4;
inline constexpr int64_t kTuneFourStepTopK = 16;
inline constexpr int64_t kTuneStaticPlanTopK = 32;

inline const std::vector<int64_t> kSupportedRadices = {19, 17, 16, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2};
inline const std::vector<int64_t> kSpecializedButterflyRadices = {2, 4, 8, 16};
inline const std::vector<int64_t> kSpecializedDirectCodeletRadices = {3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 17, 19};

[[noreturn]] void raise_python(PyObject *type, const std::string &message);
bool contains(const std::vector<int64_t> &values, int64_t value);
std::string py_str(nb::handle object);
std::string strip_torch_prefix(std::string value);
std::vector<int64_t> int64_vector_from_sequence(nb::handle sequence);
int64_t product(const std::vector<int64_t> &values);
std::string join_ints(const std::vector<int64_t> &values);
int64_t ceil_power_of_two(int64_t value);
int64_t lane_block_for(int64_t lanes);
std::string shell_quote(const std::string &value);
void cuda_check(CUresult result, const std::string &context);
void hash_combine(std::size_t &seed, std::size_t value);

template <typename T>
void hash_value(std::size_t &seed, const T &value) {
    hash_combine(seed, std::hash<T>{}(value));
}

template <typename T>
void hash_vector(std::size_t &seed, const std::vector<T> &values) {
    hash_value(seed, values.size());
    for (const T &value : values) {
        hash_value(seed, value);
    }
}

struct FFTRequest {
    int64_t fft_length = 0;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    std::optional<int64_t> n;
    int64_t requested_n = 0;
    int64_t raw_dim = -1;
    int64_t normalized_dim = -1;
    std::string norm = "backward";
    std::string input_dtype;
    std::string output_dtype;
    std::string device_type;
    int64_t device_index = -1;
    std::string device_arch;
    std::string input_layout;
    bool requires_contiguous_copy = false;
    std::string direction = "forward";
    int64_t batch = 0;
};

enum class PlanNodeKind { CtLeaf, FourStep, DirectDft, StockhamAutosort, Bluestein };
enum class KernelKind {
    Leaf,
    FourStepRow,
    FourStepCol,
    Transpose,
    TwiddleTranspose,
    BluesteinPrepare,
    BluesteinPointwise,
    BluesteinFinalize
};

std::string plan_node_kind_name(PlanNodeKind kind);
std::string kernel_kind_name(KernelKind kind);

struct PlanNode;
using PlanNodePtr = std::shared_ptr<PlanNode>;

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

nb::dict request_to_dict(const FFTRequest &request);
nb::dict problem_key_to_dict(const ProblemKey &key);
nb::dict plan_key_to_dict(const PlanKey &key);
nb::dict kernel_key_to_dict(const KernelKey &key);
std::string output_dtype_for(const std::string &input_dtype);
FFTRequest build_request(nb::object input,
                         nb::object n_obj,
                         int64_t dim,
                         nb::object norm_obj,
                         std::string direction = "forward");
void validate_request(const FFTRequest &request);

struct Factorization {
    std::vector<int64_t> factors;
    int64_t remainder = 1;
};

struct PlanNode {
    explicit PlanNode(int64_t length, PlanNodeKind kind);
    virtual ~PlanNode();
    virtual nb::dict to_dict() const = 0;

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
    nb::dict to_dict() const override;

    std::vector<int64_t> factors;
    int64_t remainder;
    int64_t lanes;
    int64_t num_warps;
    std::vector<int64_t> generic_radices;
    int64_t smem_size;
};

struct DirectDFTPlanNode final : PlanNode {
    explicit DirectDFTPlanNode(int64_t length);
    nb::dict to_dict() const override;
};

struct StockhamPlanNode final : PlanNode {
    StockhamPlanNode(int64_t length, std::vector<int64_t> factors);
    nb::dict to_dict() const override;

    std::vector<int64_t> factors;
};

struct FourStepPlanNode final : PlanNode {
    FourStepPlanNode(int64_t length, int64_t n1, int64_t n2, PlanNodePtr row, PlanNodePtr col);
    nb::dict to_dict() const override;

    int64_t n1;
    int64_t n2;
    PlanNodePtr row_plan;
    PlanNodePtr col_plan;
};

struct BluesteinPlanNode final : PlanNode {
    BluesteinPlanNode(int64_t length, int64_t conv_length, PlanNodePtr fft_plan);
    nb::dict to_dict() const override;

    int64_t conv_length;
    PlanNodePtr fft_plan;
};

struct PlanCandidate {
    PlanNodePtr node;
    double cost = 0.0;
    int priority = 99;
};

class PlanBuilder {
public:
    PlanNodePtr build(int64_t n, const FFTRequest &request);
    double cost_for(int64_t n, const FFTRequest &request);
    nb::dict wrap_plan_dict(const PlanNodePtr &root, const FFTRequest &request);
    nb::list enumerate_candidate_plans(int64_t n, const FFTRequest &request);
    PlanNodePtr node_from_dict(nb::dict node);
    nb::dict wrap_forced_plan_dict(const PlanNodePtr &root, const FFTRequest &request, std::string source);

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
    int64_t choose_lanes(int64_t n, const std::vector<int64_t> &factors);
    std::vector<int64_t> score_leaf_factorization(int64_t n, const std::vector<int64_t> &factors);
    std::vector<int64_t> select_leaf_factors(int64_t n);
    int64_t choose_num_warps(int64_t lanes);
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
    int priority(const PlanNodePtr &node);
    std::vector<int64_t> enumerate_divisors(int64_t n);
    int64_t next_supported_convolution_length(int64_t minimum);
    PlanNodePtr make_bluestein_plan(int64_t n);
    std::vector<PlanCandidate> build_auto_candidates(int64_t n);
    std::vector<PlanCandidate> build_tune_candidates(int64_t n, int64_t depth);
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

std::pair<std::vector<int64_t>, std::vector<int64_t>> decode_stage_codelet(
    int64_t codelet, const std::vector<int64_t> &radices, int64_t stage);
int64_t mixed_radix_value(const std::vector<int64_t> &digits,
                          const std::vector<int64_t> &radices,
                          std::size_t limit);
std::pair<std::vector<float>, std::vector<float>> build_stage_twiddles(
    const std::vector<int64_t> &radices, int64_t stage, int64_t lanes, const std::string &direction);
std::pair<std::vector<float>, std::vector<float>> build_dft_matrix(int64_t radix,
                                                                   const std::string &direction);
nb::object build_four_step_twiddle_tensor(const FFTRequest &request, int64_t n1, int64_t n2);
nb::object build_bluestein_chirp_tensor(const FFTRequest &request, int64_t n, bool inverse_sign);
nb::object build_bluestein_b_tensor(const FFTRequest &request, int64_t n, int64_t m);

nb::object tensor_from_float_vector(const std::vector<float> &values, const FFTRequest &request);
std::string torch_device_string(const FFTRequest &request);
nb::object tensor_from_complex_vectors(const std::vector<float> &real,
                                       const std::vector<float> &imag,
                                       const FFTRequest &request,
                                       nb::tuple shape);
nb::object empty_complex64_tensor(const FFTRequest &request, nb::tuple shape);
int64_t ceil_div(int64_t numerator, int64_t denominator);
int64_t tensor_numel(const nb::object &tensor);
int64_t tensor_size(const nb::object &tensor, int64_t dim);
int64_t tensor_stride(const nb::object &tensor, int64_t dim);
CUdeviceptr tensor_data_ptr(const nb::object &tensor);
CUstream current_cuda_stream(const FFTRequest &request);
int64_t cuda_device_max_dynamic_shared_memory_bytes(int64_t device_index);

enum class AotArgKind { DevicePtr, Int32, Int64 };

struct AotKernelArg {
    static AotKernelArg device(CUdeviceptr value);
    static AotKernelArg i32(int32_t value);
    static AotKernelArg i64(int64_t value);

    AotArgKind kind = AotArgKind::DevicePtr;
    CUdeviceptr device_ptr = 0;
    int32_t int32_value = 0;
    int64_t int64_value = 0;
};

struct AotKernel {
    ~AotKernel();
    void load();
    void launch(CUstream stream,
                const std::vector<AotKernelArg> &kernel_args,
                int64_t grid_x,
                int64_t grid_y,
                int64_t grid_z);

    std::string kernel_name;
    std::vector<unsigned char> cubin;
    int64_t shared = 0;
    int64_t num_warps = 1;
    int64_t batch_per_block = 1;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::mutex mutex;
};

struct ExecutionContext {
    const FFTRequest &request;
    CUstream stream = nullptr;
};

struct CompiledNode {
    virtual ~CompiledNode() = default;
    virtual nb::object execute(const nb::object &input, const ExecutionContext &context) const = 0;
};

struct CompiledLeafNode final : CompiledNode {
    CompiledLeafNode(int64_t length, std::shared_ptr<AotKernel> kernel, std::vector<nb::object> tables);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;

    int64_t length;
    std::shared_ptr<AotKernel> kernel;
    std::vector<nb::object> tables;
};

struct CompiledFourStepNode final : CompiledNode {
    CompiledFourStepNode(int64_t length,
                         int64_t n1,
                         int64_t n2,
                         std::shared_ptr<CompiledNode> row,
                         std::shared_ptr<CompiledNode> col,
                         std::shared_ptr<AotKernel> transpose_kernel,
                         std::shared_ptr<AotKernel> twiddle_transpose_kernel,
                         nb::object twiddle,
                         nb::object stage0,
                         nb::object stage2);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void launch_transpose(CUstream stream, const nb::object &src, const nb::object &dst) const;
    void launch_twiddle_transpose(CUstream stream,
                                  const nb::object &src,
                                  const nb::object &twiddle,
                                  const nb::object &dst) const;

    int64_t length;
    int64_t n1;
    int64_t n2;
    std::shared_ptr<CompiledNode> row;
    std::shared_ptr<CompiledNode> col;
    std::shared_ptr<AotKernel> transpose_kernel;
    std::shared_ptr<AotKernel> twiddle_transpose_kernel;
    nb::object twiddle;
    nb::object stage0;
    nb::object stage2;
};

struct CompiledFourStepFusedNode final : CompiledNode {
    CompiledFourStepFusedNode(int64_t length,
                              int64_t n1,
                              int64_t n2,
                              std::shared_ptr<AotKernel> row_kernel,
                              std::vector<nb::object> row_tables,
                              std::shared_ptr<AotKernel> col_kernel,
                              std::vector<nb::object> col_tables,
                              nb::object twiddle,
                              nb::object stage1);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void launch_row(CUstream stream, const nb::object &src, const nb::object &dst, int64_t batch) const;
    void launch_col(CUstream stream, const nb::object &src, const nb::object &dst, int64_t batch) const;

    int64_t length;
    int64_t n1;
    int64_t n2;
    std::shared_ptr<AotKernel> row_kernel;
    std::vector<nb::object> row_tables;
    std::shared_ptr<AotKernel> col_kernel;
    std::vector<nb::object> col_tables;
    nb::object twiddle;
    nb::object stage1;
};

struct CompiledBluesteinNode final : CompiledNode {
    CompiledBluesteinNode(int64_t length,
                          int64_t conv_length,
                          std::shared_ptr<CompiledNode> fft,
                          std::shared_ptr<AotKernel> prepare_kernel,
                          std::shared_ptr<AotKernel> pointwise_kernel,
                          std::shared_ptr<AotKernel> finalize_kernel,
                          nb::object chirp,
                          nb::object b_time,
                          nb::object a_buf,
                          nb::object work_buf,
                          nb::object b_fft_buf);
    nb::object execute(const nb::object &input, const ExecutionContext &context) const override;
    void ensure_b_fft(const ExecutionContext &context) const;

    int64_t length;
    int64_t conv_length;
    std::shared_ptr<CompiledNode> fft;
    std::shared_ptr<AotKernel> prepare_kernel;
    std::shared_ptr<AotKernel> pointwise_kernel;
    std::shared_ptr<AotKernel> finalize_kernel;
    nb::object chirp;
    nb::object b_time;
    nb::object a_buf;
    nb::object work_buf;
    mutable nb::object b_fft_buf;
    mutable bool b_fft_ready = false;
    mutable std::mutex b_fft_mutex;
};

class TritonCompiler {
public:
    std::shared_ptr<CompiledNode> compile_node(const PlanNodePtr &node,
                                               const FFTRequest &request,
                                               int64_t batch);
    static void clear_kernel_cache();
    static nb::dict kernel_cache_info();
    static nb::list kernel_cache_keys();

private:
    std::shared_ptr<CompiledNode> compile_leaf(const LeafPlanNode &leaf, const FFTRequest &request);
    std::shared_ptr<AotKernel> compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t n1,
                                                            int64_t n2);
    std::shared_ptr<AotKernel> compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                            const FFTRequest &request,
                                                            int64_t n1,
                                                            int64_t n2);
    std::shared_ptr<AotKernel> compile_transpose_kernel(const FFTRequest &request);
    std::shared_ptr<AotKernel> compile_twiddle_transpose_kernel(const FFTRequest &request);
    std::shared_ptr<AotKernel> compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                int64_t n,
                                                                int64_t m);
    std::shared_ptr<AotKernel> compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                  int64_t n,
                                                                  int64_t m);
    std::shared_ptr<AotKernel> compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                 int64_t n,
                                                                 int64_t m);
    std::shared_ptr<AotKernel> compile_kernel(const KernelKey &key, const std::string &command) const;
    std::filesystem::path out_dir() const;
    std::string python_executable() const;
    std::string triton_aot_entrypoint() const;

    std::shared_ptr<AotKernel> transpose_kernel;
    std::shared_ptr<AotKernel> twiddle_transpose_kernel;
};

std::string triton_target_for_request(const FFTRequest &request);
nb::list kernel_keys_for_plan(const PlanNodePtr &node, const FFTRequest &request);

enum class ExecutionBackend { AotCuda, TorchFFT };

struct ExecutablePlan {
    ProblemKey problem_key;
    PlanKey plan_key;
    FFTRequest request;
    PlanNodePtr root;
    nb::dict plan_dict;
    ExecutionBackend backend;
    std::shared_ptr<CompiledNode> compiled_root;

    nb::object execute(nb::object input) const;
};

class PlanCache {
public:
    std::shared_ptr<ExecutablePlan> get_or_create(const FFTRequest &request);
    void clear();
    nb::dict info() const;
    nb::dict keys() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<ProblemKey, std::shared_ptr<ExecutablePlan>, ProblemKeyHash> problem_cache_;
    std::unordered_map<PlanKey, PlanNodePtr, PlanKeyHash> plan_cache_;
    int64_t problem_hits_ = 0;
    int64_t problem_misses_ = 0;
    int64_t plan_hits_ = 0;
    int64_t plan_misses_ = 0;
    int64_t tuned_db_lookups_ = 0;
};

PlanCache &plan_cache();
std::shared_ptr<ExecutablePlan> resolve_plan(nb::object input,
                                             nb::object n_obj,
                                             int64_t dim,
                                             nb::object norm_obj,
                                             std::string direction);
nb::object fft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj);
nb::object ifft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj);
nb::object fft_with_plan(nb::object input,
                         nb::dict plan,
                         nb::object n_obj,
                         int64_t dim,
                         nb::object norm_obj);
nb::object ifft_with_plan(nb::object input,
                          nb::dict plan,
                          nb::object n_obj,
                          int64_t dim,
                          nb::object norm_obj);
nb::dict debug_request(nb::object input,
                       nb::object n_obj,
                       int64_t dim,
                       nb::object norm_obj,
                       std::string direction);
nb::dict debug_keys(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction);
nb::dict debug_plan(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction);
nb::dict debug_resolved_plan(nb::object input,
                             nb::object n_obj,
                             int64_t dim,
                             nb::object norm_obj,
                             std::string direction);
nb::dict debug_forced_plan(nb::object input,
                           nb::dict plan,
                           nb::object n_obj,
                           int64_t dim,
                           nb::object norm_obj,
                           std::string direction);
nb::list enumerate_plan_candidates(nb::object input,
                                   nb::object n_obj,
                                   int64_t dim,
                                   nb::object norm_obj,
                                   std::string direction);
nb::dict tune_fingerprints();

}  // namespace flagfft
