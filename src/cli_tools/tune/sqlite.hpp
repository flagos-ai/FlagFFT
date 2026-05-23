#pragma once

#include <cstdint>
#include <string>

namespace flagfft {
namespace tune {

struct TuneMeasurement {
    int64_t fft_length = 0;
    int64_t batch = 0;
    std::string direction;
    std::string device_arch;
    std::string plan_json;
    std::string plan_key;
    double compile_ms = 0.0;
    double first_call_ms = 0.0;
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double max_abs_err = 0.0;
    double rms_err = 0.0;
    std::string status;
    std::string failure_reason;
};

void init_tune_db(const std::string &db_path);
bool lookup_tune_winner(const std::string &db_path, int64_t fft_length,
                        const std::string &batch_bucket, int64_t batch,
                        const std::string &direction, const std::string &device_arch,
                        std::string &plan_json_out);
void insert_measurement(const std::string &db_path, const TuneMeasurement &m);
void mark_superseded(const std::string &db_path, int64_t fft_length,
                     const std::string &batch_bucket, const std::string &direction,
                     const std::string &device_arch);

}  // namespace tune
}  // namespace flagfft
