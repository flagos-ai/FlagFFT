#pragma once

#include <flagfft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <initializer_list>
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

namespace flagfft {

inline constexpr int64_t kPlanSchemaVersion = 3;
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

bool contains(const std::vector<int64_t> &values, int64_t value);
int64_t product(const std::vector<int64_t> &values);
std::string join_ints(const std::vector<int64_t> &values);
int64_t ceil_power_of_two(int64_t value);
int64_t ceil_div(int64_t numerator, int64_t denominator);
int64_t lane_block_for(int64_t lanes);
std::string shell_quote(const std::string &value);
std::filesystem::path executable_directory();
std::filesystem::path default_cache_dir();
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
    BluesteinFinalize,
    ReshapePack,
    TwiddleReshapePack,
    RealToComplex,
    R2CHalfPack,
    CompactToHermitianFull,
    ComplexToReal
};

int64_t complex_element_bytes(const std::string &input_dtype);
std::string complex_dtype_for(const std::string &input_dtype);

std::string plan_node_kind_name(PlanNodeKind kind);
std::string kernel_kind_name(KernelKind kind);

struct PlanNode;
using PlanNodePtr = std::shared_ptr<PlanNode>;

}  // namespace flagfft
