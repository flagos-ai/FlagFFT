#pragma once

#include "flagfft/codegen.hpp"

namespace flagfft {

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2, const std::string &dtype);
std::string batch_bucket(int64_t batch);
bool env_flag_enabled(const char *value);
std::optional<std::filesystem::path> tuned_db_path();

}  // namespace flagfft
