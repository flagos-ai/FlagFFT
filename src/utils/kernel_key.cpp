#include "flagfft/core.hpp"

namespace flagfft {

KernelKey KernelKey::leaf(std::string target,
                          std::string direction,
                          int64_t length,
                          std::vector<int64_t> factors,
                          int64_t lanes,
                          int64_t num_warps,
                          std::vector<int64_t> generic_radices,
                          int64_t smem_size) {
    KernelKey key;
    key.kind = KernelKind::Leaf;
    key.target = std::move(target);
    key.direction = std::move(direction);
    key.length = length;
    key.factors = std::move(factors);
    key.lanes = lanes;
    key.num_warps = num_warps;
    key.generic_radices = std::move(generic_radices);
    key.smem_size = smem_size;
    return key;
}

KernelKey KernelKey::four_step_row(std::string target,
                                   std::string direction,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
    KernelKey key = KernelKey::leaf(std::move(target),
                                    std::move(direction),
                                    length,
                                    std::move(factors),
                                    lanes,
                                    num_warps,
                                    std::move(generic_radices),
                                    smem_size);
    key.kind = KernelKind::FourStepRow;
    key.four_step_n1 = n1;
    key.four_step_n2 = n2;
    return key;
}

KernelKey KernelKey::four_step_col(std::string target,
                                   std::string direction,
                                   int64_t n1,
                                   int64_t n2,
                                   int64_t length,
                                   std::vector<int64_t> factors,
                                   int64_t lanes,
                                   int64_t num_warps,
                                   std::vector<int64_t> generic_radices,
                                   int64_t smem_size) {
    KernelKey key = KernelKey::leaf(std::move(target),
                                    std::move(direction),
                                    length,
                                    std::move(factors),
                                    lanes,
                                    num_warps,
                                    std::move(generic_radices),
                                    smem_size);
    key.kind = KernelKind::FourStepCol;
    key.four_step_n1 = n1;
    key.four_step_n2 = n2;
    return key;
}

KernelKey KernelKey::transpose(std::string target) {
    KernelKey key;
    key.kind = KernelKind::Transpose;
    key.target = std::move(target);
    return key;
}

KernelKey KernelKey::twiddle_transpose(std::string target) {
    KernelKey key;
    key.kind = KernelKind::TwiddleTranspose;
    key.target = std::move(target);
    return key;
}

KernelKey KernelKey::bluestein_prepare(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinPrepare;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

KernelKey KernelKey::bluestein_pointwise(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinPointwise;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

KernelKey KernelKey::bluestein_finalize(std::string target, int64_t n, int64_t m) {
    KernelKey key;
    key.kind = KernelKind::BluesteinFinalize;
    key.target = std::move(target);
    key.bluestein_n = n;
    key.bluestein_m = m;
    return key;
}

bool KernelKey::operator==(const KernelKey &other) const {
    return kind == other.kind && target == other.target && direction == other.direction &&
           length == other.length && factors == other.factors && lanes == other.lanes &&
           num_warps == other.num_warps && generic_radices == other.generic_radices &&
           smem_size == other.smem_size && four_step_n1 == other.four_step_n1 &&
           four_step_n2 == other.four_step_n2 && bluestein_n == other.bluestein_n &&
           bluestein_m == other.bluestein_m;
}

std::string KernelKey::repr() const {
    std::ostringstream out;
    out << "kind=" << kernel_kind_name(kind) << ";target=" << target;
    if (kind == KernelKind::Leaf || kind == KernelKind::FourStepRow ||
        kind == KernelKind::FourStepCol) {
        out << ";direction=" << direction << ";length=" << length << ";factors=["
            << join_ints(factors) << "]"
            << ";lanes=" << lanes << ";num_warps=" << num_warps
            << ";generic_radices=[" << join_ints(generic_radices)
            << "];smem_size=" << smem_size;
        if (kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol) {
            out << ";four_step_n1=" << four_step_n1 << ";four_step_n2=" << four_step_n2;
        }
    }
    if (kind == KernelKind::BluesteinPrepare || kind == KernelKind::BluesteinPointwise ||
        kind == KernelKind::BluesteinFinalize) {
        out << ";bluestein_n=" << bluestein_n << ";bluestein_m=" << bluestein_m;
    }
    return out.str();
}

std::size_t KernelKeyHash::operator()(const KernelKey &key) const {
    std::size_t seed = 0;
    hash_value(seed, static_cast<int64_t>(key.kind));
    hash_value(seed, key.target);
    hash_value(seed, key.direction);
    hash_value(seed, key.length);
    hash_vector(seed, key.factors);
    hash_value(seed, key.lanes);
    hash_value(seed, key.num_warps);
    hash_vector(seed, key.generic_radices);
    hash_value(seed, key.smem_size);
    hash_value(seed, key.four_step_n1);
    hash_value(seed, key.four_step_n2);
    hash_value(seed, key.bluestein_n);
    hash_value(seed, key.bluestein_m);
    return seed;
}

}  // namespace flagfft
