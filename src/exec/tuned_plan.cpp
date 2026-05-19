#include "flagfft/core.hpp"

namespace flagfft {

inline constexpr int64_t kFourStepColInnerPack = 2;
inline constexpr int64_t kFourStepColInnerPackMinN1 = 128;

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2) {
    (void)n2;
    if (n1 < kFourStepColInnerPackMinN1) {
        return 1;
    }
    return kFourStepColInnerPack;
}

std::string batch_bucket(int64_t batch) {
    if (batch <= 1) {
        return "1";
    }
    if (batch <= 8) {
        return "2-8";
    }
    if (batch <= 64) {
        return "9-64";
    }
    if (batch <= 512) {
        return "65-512";
    }
    return "513+";
}

bool env_flag_enabled(const char *value) {
    if (value == nullptr) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(normalized.empty() || normalized == "0" || normalized == "false" ||
             normalized == "off" || normalized == "no");
}

std::optional<std::filesystem::path> tuned_db_path() {
    if (env_flag_enabled(std::getenv("FLAGFFT_TUNE_DISABLE"))) {
        return std::nullopt;
    }
    const char *override_path = std::getenv("FLAGFFT_TUNE_DB");
    if (override_path != nullptr && std::string(override_path).size() > 0) {
        return std::filesystem::path(override_path);
    }
    return default_cache_dir() / "tuned_plans.sqlite";
}

}  // namespace flagfft
