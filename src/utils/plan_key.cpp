#include "flagfft/core.hpp"

namespace flagfft {

bool PlanKey::operator==(const PlanKey &other) const {
  return schema_version == other.schema_version && root_kind == other.root_kind && length == other.length &&
         factors == other.factors && remainder == other.remainder && lanes == other.lanes &&
         num_warps == other.num_warps && generic_radices == other.generic_radices &&
         smem_size == other.smem_size && n1 == other.n1 && n2 == other.n2 &&
         conv_length == other.conv_length && child_keys == other.child_keys;
}

std::string PlanKey::repr() const {
  std::ostringstream out;
  out << "schema=" << schema_version << ";kind=" << plan_node_kind_name(root_kind) << ";length=" << length;
  if (!factors.empty()) {
    out << ";factors=[" << join_ints(factors) << "]";
  }
  if (root_kind == PlanNodeKind::CtLeaf) {
    out << ";remainder=" << remainder << ";lanes=" << lanes << ";num_warps=" << num_warps
        << ";generic_radices=[" << join_ints(generic_radices) << "];smem_size=" << smem_size;
  }
  if (root_kind == PlanNodeKind::FourStep) {
    out << ";n1=" << n1 << ";n2=" << n2;
  }
  if (root_kind == PlanNodeKind::Bluestein) {
    out << ";conv_length=" << conv_length;
  }
  if (!child_keys.empty()) {
    out << ";children=[";
    for (std::size_t i = 0; i < child_keys.size(); ++i) {
      if (i != 0) {
        out << "|";
      }
      out << "{" << child_keys[i] << "}";
    }
    out << "]";
  }
  return out.str();
}

std::size_t PlanKeyHash::operator()(const PlanKey &key) const {
  std::size_t seed = 0;
  hash_value(seed, key.schema_version);
  hash_value(seed, static_cast<int64_t>(key.root_kind));
  hash_value(seed, key.length);
  hash_vector(seed, key.factors);
  hash_value(seed, key.remainder);
  hash_value(seed, key.lanes);
  hash_value(seed, key.num_warps);
  hash_vector(seed, key.generic_radices);
  hash_value(seed, key.smem_size);
  hash_value(seed, key.n1);
  hash_value(seed, key.n2);
  hash_value(seed, key.conv_length);
  hash_vector(seed, key.child_keys);
  return seed;
}

}  // namespace flagfft
