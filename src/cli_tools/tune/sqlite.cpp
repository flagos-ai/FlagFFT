#include "sqlite.hpp"

#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "sqlite_wrapper.hpp"

namespace flagfft {
namespace tune {

  void init_tune_db(const std::string &db_path) {
    SqliteDb db(db_path);
    db.exec(
        "CREATE TABLE IF NOT EXISTS tuned_measurements ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  schema_version INTEGER NOT NULL,"
        "  device_arch TEXT NOT NULL,"
        "  fft_length INTEGER NOT NULL,"
        "  batch_bucket TEXT NOT NULL,"
        "  batch INTEGER NOT NULL,"
        "  dtype TEXT NOT NULL,"
        "  direction TEXT NOT NULL,"
        "  norm TEXT NOT NULL,"
        "  input_layout TEXT NOT NULL,"
        "  planner_fingerprint TEXT NOT NULL,"
        "  codegen_fingerprint TEXT NOT NULL,"
        "  runtime_fingerprint TEXT NOT NULL,"
        "  benchmark_fingerprint TEXT NOT NULL,"
        "  plan_key TEXT NOT NULL,"
        "  plan_json TEXT NOT NULL,"
        "  status TEXT NOT NULL,"
        "  rank INTEGER,"
        "  compile_ms REAL,"
        "  first_call_ms REAL,"
        "  median_ms REAL,"
        "  p90_ms REAL,"
        "  max_abs_err REAL,"
        "  rms_err REAL,"
        "  failure_reason TEXT,"
        "  measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")");
    db.exec(
        "CREATE INDEX IF NOT EXISTS tuned_measurements_lookup "
        "ON tuned_measurements ("
        "  schema_version, status, rank, device_arch, fft_length, batch_bucket, dtype,"
        "  direction, norm, input_layout, planner_fingerprint, codegen_fingerprint,"
        "  runtime_fingerprint"
        ")");
  }

  bool lookup_tune_winner(const std::string &db_path,
                          int64_t fft_length,
                          const std::string &batch_bucket,
                          int64_t batch,
                          const std::string &direction,
                          const std::string &device_arch,
                          std::string &plan_json_out) {
    auto fps = tune_fingerprints();
    auto db_opt = tuned_db_path();
    (void)db_opt;

    SqliteDb db(db_path);
    SqliteStmt stmt(db,
                    "SELECT plan_json FROM tuned_measurements "
                    "WHERE schema_version=? AND status='valid' AND rank=0 "
                    "AND device_arch=? AND fft_length=? AND batch_bucket=? AND dtype=? "
                    "AND direction=? AND norm=? AND input_layout=? "
                    "AND planner_fingerprint=? AND codegen_fingerprint=? "
                    "AND runtime_fingerprint=? "
                    "ORDER BY measured_at DESC LIMIT 1");

    stmt.bind_int64(1, kPlanSchemaVersion);
    stmt.bind_text(2, device_arch);
    stmt.bind_int64(3, fft_length);
    stmt.bind_text(4, batch_bucket);
    stmt.bind_text(5, "complex64");
    stmt.bind_text(6, direction);
    stmt.bind_text(7, "backward");
    stmt.bind_text(8, "contiguous");
    stmt.bind_text(9, fps.planner);
    stmt.bind_text(10, fps.codegen);
    stmt.bind_text(11, fps.runtime);

    if (!stmt.step()) {
      return false;
    }
    plan_json_out = stmt.column_text(0);
    return true;
  }

  void insert_measurement(const std::string &db_path, const TuneMeasurement &m) {
    auto fps = tune_fingerprints();
    SqliteDb db(db_path);

    SqliteStmt stmt(db,
                    "INSERT INTO tuned_measurements ("
                    "  schema_version, device_arch, fft_length, batch_bucket, batch, dtype,"
                    "  direction, norm, input_layout, planner_fingerprint, codegen_fingerprint,"
                    "  runtime_fingerprint, benchmark_fingerprint, plan_key, plan_json,"
                    "  status, rank, compile_ms, first_call_ms, median_ms, p90_ms,"
                    "  max_abs_err, rms_err, failure_reason"
                    ") VALUES ("
                    "  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
                    ")");

    stmt.bind_int64(1, kPlanSchemaVersion);
    stmt.bind_text(2, m.device_arch);
    stmt.bind_int64(3, m.fft_length);
    stmt.bind_text(4, batch_bucket(m.batch));
    stmt.bind_int64(5, m.batch);
    stmt.bind_text(6, "complex64");
    stmt.bind_text(7, m.direction);
    stmt.bind_text(8, "backward");
    stmt.bind_text(9, "contiguous");
    stmt.bind_text(10, fps.planner);
    stmt.bind_text(11, fps.codegen);
    stmt.bind_text(12, fps.runtime);
    stmt.bind_text(13, fps.benchmark);
    stmt.bind_text(14, m.plan_key);
    stmt.bind_text(15, m.plan_json);
    stmt.bind_text(16, m.status);
    if (m.status == "ok" || m.status == "valid") {
      stmt.bind_int64(17, m.status == "valid" ? 0 : 99);
    } else {
      stmt.bind_null(17);
    }
    stmt.bind_double(18, m.compile_ms);
    stmt.bind_double(19, m.first_call_ms);
    stmt.bind_double(20, m.median_ms);
    stmt.bind_double(21, m.p90_ms);
    stmt.bind_double(22, m.max_abs_err);
    stmt.bind_double(23, m.rms_err);
    if (m.failure_reason.empty()) {
      stmt.bind_null(24);
    } else {
      stmt.bind_text(24, m.failure_reason);
    }
    stmt.step();
  }

  void mark_superseded(const std::string &db_path,
                       int64_t fft_length,
                       const std::string &batch_bucket,
                       const std::string &direction,
                       const std::string &device_arch) {
    auto fps = tune_fingerprints();
    SqliteDb db(db_path);

    SqliteStmt stmt(db,
                    "UPDATE tuned_measurements SET status='superseded' "
                    "WHERE schema_version=? AND device_arch=? AND fft_length=? "
                    "AND batch_bucket=? AND dtype=? AND direction=? AND norm=? "
                    "AND input_layout=? AND planner_fingerprint=? AND codegen_fingerprint=? "
                    "AND runtime_fingerprint=? AND status='valid'");

    stmt.bind_int64(1, kPlanSchemaVersion);
    stmt.bind_text(2, device_arch);
    stmt.bind_int64(3, fft_length);
    stmt.bind_text(4, batch_bucket);
    stmt.bind_text(5, "complex64");
    stmt.bind_text(6, direction);
    stmt.bind_text(7, "backward");
    stmt.bind_text(8, "contiguous");
    stmt.bind_text(9, fps.planner);
    stmt.bind_text(10, fps.codegen);
    stmt.bind_text(11, fps.runtime);
    stmt.step();
  }

}  // namespace tune
}  // namespace flagfft
