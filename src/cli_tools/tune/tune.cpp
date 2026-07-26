// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cli_tools/tune/tune.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "c_api_internal.hpp"
#include "cli_tools/common/cli_utils.hpp"
#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "sqlite_wrapper.hpp"

namespace flagfft::cli::tune {
namespace {

  constexpr double kCorrectnessRelL2Limit = 2.0e-4;
  constexpr double kCorrectnessRelLinfLimit = 2.0e-4;

  struct AccuracyResult {
    bool valid = false;
    double rel_l2 = std::numeric_limits<double>::infinity();
    double rel_linf = std::numeric_limits<double>::infinity();
  };

  struct TimingResult {
    double median_ms = 0.0;
    double p90_ms = 0.0;
    std::vector<float> samples;
  };

  struct PhaseResult {
    TimingResult forward;
    TimingResult inverse;
    double objective_ms = std::numeric_limits<double>::infinity();
  };

  struct CandidateResult {
    PlanCandidate candidate;
    std::shared_ptr<CompiledRawNode> forward;
    std::shared_ptr<CompiledRawNode> inverse;
    std::string key;
    int64_t n1 = 0;
    int64_t n2 = 0;
    std::string status = "pending";
    std::string error;
    AccuracyResult forward_accuracy;
    AccuracyResult inverse_accuracy;
    PhaseResult screen;
    std::optional<PhaseResult> final;
  };

  double percentile(std::vector<float> samples, double fraction) {
    if (samples.empty()) {
      return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    std::size_t index = static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1));
    return samples[index];
  }

  TimingResult summarize(std::vector<float> samples) {
    TimingResult result;
    result.median_ms = percentile(samples, 0.5);
    result.p90_ms = percentile(samples, 0.9);
    result.samples = std::move(samples);
    return result;
  }

  void bind_finite_or_null(SqliteStmt& statement, int index, double value) {
    if (std::isfinite(value)) {
      statement.bind_double(index, value);
    } else {
      statement.bind_null(index);
    }
  }

  nlohmann::json timing_json(const TimingResult& timing) {
    return {
        {"median_ms", timing.median_ms},
        {   "p90_ms",    timing.p90_ms},
        {  "samples",   timing.samples},
    };
  }

  nlohmann::json accuracy_json(const AccuracyResult& accuracy) {
    return {
        {     "valid",           accuracy.valid},
        {    "rel_l2",          accuracy.rel_l2},
        {  "rel_linf",        accuracy.rel_linf},
        {  "limit_l2",   kCorrectnessRelL2Limit},
        {"limit_linf", kCorrectnessRelLinfLimit},
    };
  }

  nlohmann::json phase_json(const PhaseResult& phase) {
    return {
        {"objective_ms",         phase.objective_ms},
        {     "forward", timing_json(phase.forward)},
        {     "inverse", timing_json(phase.inverse)},
    };
  }

  FlagFFTPlanDesc make_desc(const TuneOptions& options) {
    FlagFFTPlanDesc desc;
    desc.rank = 1;
    desc.n = {options.length};
    desc.inembed = desc.n;
    desc.onembed = desc.n;
    desc.istride = 1;
    desc.ostride = 1;
    desc.idist = options.length;
    desc.odist = options.length;
    desc.batch = options.batch;
    desc.type = FLAGFFT_C2C;
    desc.precision = FlagFFTPrecision::Float32;
    desc.kind = FlagFFTTransformKind::C2C;
    int device_index = 0;
    std::string device_arch;
    check_flagfft(adaptor::ensure_device(device_index, device_arch), "initialize tune device");
    desc.device_index = device_index;
    desc.device_arch = std::move(device_arch);
    return desc;
  }

  std::vector<flagfftComplex> make_input(int64_t count) {
    std::vector<flagfftComplex> input(static_cast<std::size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
      double phase = static_cast<double>(index + 1) * 0.173;
      input[static_cast<std::size_t>(index)] = {
          static_cast<float>(std::sin(phase)),
          static_cast<float>(std::cos(phase * 0.731)),
      };
    }
    return input;
  }

  AccuracyResult compare_outputs(const std::vector<flagfftComplex>& output,
                                 const std::vector<flagfftComplex>& reference) {
    long double error_sq = 0.0L;
    long double reference_sq = 0.0L;
    long double max_error = 0.0L;
    long double max_reference = 0.0L;
    bool finite = output.size() == reference.size();
    for (std::size_t index = 0; index < output.size() && index < reference.size(); ++index) {
      long double dx =
          static_cast<long double>(output[index].x) - static_cast<long double>(reference[index].x);
      long double dy =
          static_cast<long double>(output[index].y) - static_cast<long double>(reference[index].y);
      long double error = std::hypot(dx, dy);
      long double ref = std::hypot(static_cast<long double>(reference[index].x),
                                   static_cast<long double>(reference[index].y));
      finite = finite && std::isfinite(error) && std::isfinite(ref);
      error_sq += error * error;
      reference_sq += ref * ref;
      max_error = std::max(max_error, error);
      max_reference = std::max(max_reference, ref);
    }

    AccuracyResult result;
    result.rel_l2 = reference_sq == 0.0L ? static_cast<double>(std::sqrt(error_sq))
                                         : static_cast<double>(std::sqrt(error_sq / reference_sq));
    result.rel_linf = max_reference == 0.0L ? static_cast<double>(max_error)
                                            : static_cast<double>(max_error / max_reference);
    result.valid =
        finite && result.rel_l2 <= kCorrectnessRelL2Limit && result.rel_linf <= kCorrectnessRelLinfLimit;
    return result;
  }

  class TuneHarness {
   public:
    TuneHarness(const TuneOptions& options, FFTRequest forward_request, FFTRequest inverse_request)
        : options_(options),
          forward_request_(std::move(forward_request)),
          inverse_request_(std::move(inverse_request)),
          element_count_(static_cast<int64_t>(options.length) * options.batch),
          bytes_(static_cast<std::size_t>(element_count_) * sizeof(flagfftComplex)),
          input_(bytes_),
          output_(bytes_),
          reference_output_(bytes_),
          host_input_(make_input(element_count_)),
          host_output_(static_cast<std::size_t>(element_count_)),
          forward_reference_(static_cast<std::size_t>(element_count_)),
          inverse_reference_(static_cast<std::size_t>(element_count_)) {
      input_.copy_from_host(host_input_.data(), bytes_);
      test_adaptor::ref_plan_1d(reference_plan_, options.length, FLAGFFT_C2C, options.batch);
      test_adaptor::ref_set_stream(reference_plan_, stream_.get());
      build_reference(FLAGFFT_FORWARD, forward_reference_);
      build_reference(FLAGFFT_INVERSE, inverse_reference_);
    }

    AccuracyResult verify(const std::shared_ptr<CompiledRawNode>& compiled, int direction) {
      const FFTRequest& request = direction == FLAGFFT_INVERSE ? inverse_request_ : forward_request_;
      RawExecutionContext context {request, stream_.get(), options_.batch, options_.length, options_.length};
      check_flagfft(compiled->execute(input_.get(), output_.get(), context),
                    "execute tune correctness candidate");
      stream_.sync();
      output_.copy_to_host(host_output_.data(), bytes_);
      const auto& reference = direction == FLAGFFT_INVERSE ? inverse_reference_ : forward_reference_;
      return compare_outputs(host_output_, reference);
    }

    TimingResult benchmark(const std::shared_ptr<CompiledRawNode>& compiled,
                           int direction,
                           int warmup,
                           int iters) {
      const FFTRequest& request = direction == FLAGFFT_INVERSE ? inverse_request_ : forward_request_;
      RawExecutionContext context {request, stream_.get(), options_.batch, options_.length, options_.length};
      for (int iteration = 0; iteration < warmup; ++iteration) {
        check_flagfft(compiled->execute(input_.get(), output_.get(), context), "execute tune warmup");
      }
      stream_.sync();

      std::vector<float> samples;
      samples.reserve(static_cast<std::size_t>(iters));
      for (int iteration = 0; iteration < iters; ++iteration) {
        timer_.start(stream_.get());
        check_flagfft(compiled->execute(input_.get(), output_.get(), context), "execute tune benchmark");
        timer_.stop(stream_.get());
        samples.push_back(timer_.elapsed_ms());
      }
      stream_.sync();
      return summarize(std::move(samples));
    }

    TimingResult benchmark_reference(int direction, int warmup, int iters) {
      for (int iteration = 0; iteration < warmup; ++iteration) {
        execute_reference(direction);
      }
      stream_.sync();

      std::vector<float> samples;
      samples.reserve(static_cast<std::size_t>(iters));
      for (int iteration = 0; iteration < iters; ++iteration) {
        timer_.start(stream_.get());
        execute_reference(direction);
        timer_.stop(stream_.get());
        samples.push_back(timer_.elapsed_ms());
      }
      stream_.sync();
      return summarize(std::move(samples));
    }

   private:
    void execute_reference(int direction) {
      test_adaptor::ref_exec_c2c(reference_plan_,
                                 static_cast<flagfftComplex*>(input_.data()),
                                 static_cast<flagfftComplex*>(reference_output_.data()),
                                 direction);
    }

    void build_reference(int direction, std::vector<flagfftComplex>& host) {
      execute_reference(direction);
      stream_.sync();
      reference_output_.copy_to_host(host.data(), bytes_);
    }

    TuneOptions options_;
    FFTRequest forward_request_;
    FFTRequest inverse_request_;
    int64_t element_count_;
    std::size_t bytes_;
    adaptor::Memory input_;
    adaptor::Memory output_;
    adaptor::Memory reference_output_;
    std::vector<flagfftComplex> host_input_;
    std::vector<flagfftComplex> host_output_;
    std::vector<flagfftComplex> forward_reference_;
    std::vector<flagfftComplex> inverse_reference_;
    test_adaptor::RefPlanHandle reference_plan_;
    adaptor::Stream stream_;
    adaptor::EventTimer timer_;
  };

  PhaseResult benchmark_phase(TuneHarness& harness, const CandidateResult& candidate, int warmup, int iters) {
    PhaseResult result;
    result.forward = harness.benchmark(candidate.forward, FLAGFFT_FORWARD, warmup, iters);
    result.inverse = harness.benchmark(candidate.inverse, FLAGFFT_INVERSE, warmup, iters);
    result.objective_ms = 0.5 * (result.forward.median_ms + result.inverse.median_ms);
    return result;
  }

  std::filesystem::path result_db_path(const TuneOptions& options) {
    if (!options.db_path.empty()) {
      return options.db_path;
    }
    auto path = tuned_db_path();
    if (!path.has_value()) {
      throw AssertionFailure("tuned-plan persistence is disabled; use --no-save or set --db");
    }
    return *path;
  }

  int64_t unix_time_millis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  void create_tune_schema(SqliteDb& db) {
    db.exec(
        "CREATE TABLE IF NOT EXISTS tune_trials ("
        "run_id TEXT NOT NULL, candidate_rank INTEGER NOT NULL, status TEXT NOT NULL, "
        "device_arch TEXT NOT NULL, fft_length INTEGER NOT NULL, batch INTEGER NOT NULL, "
        "n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, plan_key TEXT NOT NULL, "
        "screen_objective_ms REAL, final_objective_ms REAL, "
        "forward_rel_l2 REAL, forward_rel_linf REAL, "
        "inverse_rel_l2 REAL, inverse_rel_linf REAL, "
        "plan_json TEXT NOT NULL, error TEXT, measured_at INTEGER NOT NULL)");
    db.exec(
        "CREATE TABLE IF NOT EXISTS tuned_measurements ("
        "schema_version INTEGER NOT NULL, status TEXT NOT NULL, rank INTEGER NOT NULL, "
        "device_arch TEXT NOT NULL, fft_length INTEGER NOT NULL, batch_bucket TEXT NOT NULL, "
        "dtype TEXT NOT NULL, direction TEXT NOT NULL, norm TEXT NOT NULL, "
        "input_layout TEXT NOT NULL, planner_fingerprint TEXT NOT NULL, "
        "codegen_fingerprint TEXT NOT NULL, runtime_fingerprint TEXT NOT NULL, "
        "benchmark_fingerprint TEXT NOT NULL, plan_json TEXT NOT NULL, "
        "plan_key TEXT NOT NULL, median_ms REAL NOT NULL, p90_ms REAL NOT NULL, "
        "measured_at INTEGER NOT NULL)");
  }

  void persist_results(const std::filesystem::path& path,
                       const std::string& run_id,
                       const std::vector<CandidateResult>& candidates,
                       const CandidateResult& winner,
                       const FFTRequest& forward_request,
                       const FFTRequest& inverse_request,
                       const TuneOptions& options) {
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    SqliteDb db(path.string());
    create_tune_schema(db);
    db.exec("BEGIN IMMEDIATE");
    const int64_t measured_at = unix_time_millis();

    SqliteStmt trial(db,
                     "INSERT INTO tune_trials "
                     "(run_id,candidate_rank,status,device_arch,fft_length,batch,n1,n2,"
                     "plan_key,screen_objective_ms,final_objective_ms,forward_rel_l2,"
                     "forward_rel_linf,inverse_rel_l2,inverse_rel_linf,plan_json,error,measured_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const CandidateResult& candidate = candidates[index];
      nlohmann::json plan = wrap_plan_json(candidate.candidate.node, forward_request, "decomposition-tune");
      trial.bind_text(1, run_id);
      trial.bind_int64(2, static_cast<int64_t>(index));
      trial.bind_text(3, candidate.status);
      trial.bind_text(4, forward_request.device_arch);
      trial.bind_int64(5, options.length);
      trial.bind_int64(6, options.batch);
      trial.bind_int64(7, candidate.n1);
      trial.bind_int64(8, candidate.n2);
      trial.bind_text(9, candidate.key);
      if (std::isfinite(candidate.screen.objective_ms)) {
        trial.bind_double(10, candidate.screen.objective_ms);
      } else {
        trial.bind_null(10);
      }
      if (candidate.final.has_value()) {
        trial.bind_double(11, candidate.final->objective_ms);
      } else {
        trial.bind_null(11);
      }
      bind_finite_or_null(trial, 12, candidate.forward_accuracy.rel_l2);
      bind_finite_or_null(trial, 13, candidate.forward_accuracy.rel_linf);
      bind_finite_or_null(trial, 14, candidate.inverse_accuracy.rel_l2);
      bind_finite_or_null(trial, 15, candidate.inverse_accuracy.rel_linf);
      trial.bind_text(16, plan.dump());
      if (candidate.error.empty()) {
        trial.bind_null(17);
      } else {
        trial.bind_text(17, candidate.error);
      }
      trial.bind_int64(18, measured_at);
      trial.step();
      trial.reset();
    }

    const TuneFingerprints fingerprints = tune_fingerprints();
    auto write_winner = [&](const FFTRequest& request, const TimingResult& timing) {
      nlohmann::json plan = wrap_plan_json(winner.candidate.node, request, "decomposition-tune");
      SqliteStmt remove(db,
                        "DELETE FROM tuned_measurements WHERE schema_version=? AND rank=0 "
                        "AND device_arch=? AND fft_length=? AND batch_bucket=? AND dtype=? "
                        "AND direction=? AND norm=? AND input_layout=? AND planner_fingerprint=? "
                        "AND codegen_fingerprint=? AND runtime_fingerprint=?");
      remove.bind_int64(1, kPlanSchemaVersion);
      remove.bind_text(2, request.device_arch);
      remove.bind_int64(3, request.requested_n);
      remove.bind_text(4, batch_bucket(request.batch));
      remove.bind_text(5, request.input_dtype);
      remove.bind_text(6, request.direction);
      remove.bind_text(7, request.norm);
      remove.bind_text(8, request.input_layout);
      remove.bind_text(9, fingerprints.planner);
      remove.bind_text(10, fingerprints.codegen);
      remove.bind_text(11, fingerprints.runtime);
      remove.step();

      SqliteStmt insert(db,
                        "INSERT INTO tuned_measurements "
                        "(schema_version,status,rank,device_arch,fft_length,batch_bucket,dtype,"
                        "direction,norm,input_layout,planner_fingerprint,codegen_fingerprint,"
                        "runtime_fingerprint,benchmark_fingerprint,plan_json,plan_key,median_ms,"
                        "p90_ms,measured_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
      insert.bind_int64(1, kPlanSchemaVersion);
      insert.bind_text(2, "valid");
      insert.bind_int64(3, 0);
      insert.bind_text(4, request.device_arch);
      insert.bind_int64(5, request.requested_n);
      insert.bind_text(6, batch_bucket(request.batch));
      insert.bind_text(7, request.input_dtype);
      insert.bind_text(8, request.direction);
      insert.bind_text(9, request.norm);
      insert.bind_text(10, request.input_layout);
      insert.bind_text(11, fingerprints.planner);
      insert.bind_text(12, fingerprints.codegen);
      insert.bind_text(13, fingerprints.runtime);
      insert.bind_text(14, fingerprints.benchmark);
      insert.bind_text(15, plan.dump());
      insert.bind_text(16, winner.key);
      insert.bind_double(17, timing.median_ms);
      insert.bind_double(18, timing.p90_ms);
      insert.bind_int64(19, measured_at);
      insert.step();
    };

    const PhaseResult& final = winner.final.value();
    write_winner(forward_request, final.forward);
    write_winner(inverse_request, final.inverse);
    db.exec("COMMIT");
  }

  nlohmann::json candidate_json(const CandidateResult& candidate, std::size_t rank) {
    nlohmann::json out = {
        {       "rank",  rank                       },
        {     "status",             candidate.status},
        {   "plan_key",                candidate.key},
        {      "split", {candidate.n1, candidate.n2}},
        {"correctness",
         {
         {"forward", accuracy_json(candidate.forward_accuracy)},
         {"inverse", accuracy_json(candidate.inverse_accuracy)},
         }                                          },
    };
    if (!candidate.error.empty()) {
      out["error"] = candidate.error;
    }
    if (std::isfinite(candidate.screen.objective_ms)) {
      out["screen"] = phase_json(candidate.screen);
    }
    if (candidate.final.has_value()) {
      out["final"] = phase_json(*candidate.final);
    }
    return out;
  }

}  // namespace

nlohmann::json run_decomposition_tune(const TuneOptions& options) {
  if (options.batch != 1) {
    throw AssertionFailure("decomposition tuner v1 currently supports only --batch 1");
  }
  if (options.max_candidates <= 0 || options.finalists <= 0 || options.finalists > options.max_candidates) {
    throw AssertionFailure("--finalists must be between 1 and --max-candidates");
  }
  if (options.screen_warmup < 0 || options.final_warmup < 0 || options.screen_iters <= 0 ||
      options.final_iters <= 0) {
    throw AssertionFailure("tune warmups must be non-negative and iterations must be positive");
  }

  FlagFFTPlanDesc desc = make_desc(options);
  FFTRequest forward_request = request_from_desc(desc, "forward");
  FFTRequest inverse_request = request_from_desc(desc, "inverse");
  PlanBuilder builder;
  std::vector<PlanCandidate> plans =
      builder.build_decomposition_tune_candidates(options.length, forward_request, options.max_candidates);
  if (plans.empty()) {
    throw CliException("no decomposition candidates were generated", kExitFailed);
  }

  TuneHarness harness(options, forward_request, inverse_request);
  std::vector<CandidateResult> candidates;
  candidates.reserve(plans.size());
  TritonCompiler compiler;
  for (std::size_t index = 0; index < plans.size(); ++index) {
    CandidateResult candidate;
    candidate.candidate = plans[index];
    candidate.key = PlanKey::from_node(candidate.candidate.node).repr();
    auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(candidate.candidate.node);
    if (four_step != nullptr) {
      candidate.n1 = four_step->n1;
      candidate.n2 = four_step->n2;
    }

    std::cerr << "[tune] candidate " << index + 1 << "/" << plans.size() << " split=" << candidate.n1 << "x"
              << candidate.n2 << " compiling\n";
    try {
      candidate.forward = compiler.compile_raw_node(candidate.candidate.node, forward_request, options.batch);
      candidate.inverse = compiler.compile_raw_node(candidate.candidate.node, inverse_request, options.batch);
      candidate.forward_accuracy = harness.verify(candidate.forward, FLAGFFT_FORWARD);
      candidate.inverse_accuracy = harness.verify(candidate.inverse, FLAGFFT_INVERSE);
      if (!candidate.forward_accuracy.valid || !candidate.inverse_accuracy.valid) {
        candidate.status = "invalid";
        candidate.error = "correctness gate failed";
      } else {
        candidate.screen = benchmark_phase(harness, candidate, options.screen_warmup, options.screen_iters);
        candidate.status = "screened";
        std::cerr << "[tune] split=" << candidate.n1 << "x" << candidate.n2
                  << " screen=" << candidate.screen.objective_ms << " ms\n";
      }
    } catch (const std::exception& error) {
      candidate.status = "error";
      candidate.error = error.what();
      std::cerr << "[tune] split=" << candidate.n1 << "x" << candidate.n2 << " failed: " << error.what()
                << "\n";
    }
    candidates.push_back(std::move(candidate));
  }

  std::vector<std::size_t> valid_indices;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (candidates[index].status == "screened") {
      valid_indices.push_back(index);
    }
  }
  if (valid_indices.empty()) {
    throw CliException("all decomposition candidates failed", kExitFailed);
  }
  std::sort(valid_indices.begin(), valid_indices.end(), [&](std::size_t lhs, std::size_t rhs) {
    return candidates[lhs].screen.objective_ms < candidates[rhs].screen.objective_ms;
  });
  valid_indices.resize(std::min(valid_indices.size(), static_cast<std::size_t>(options.finalists)));

  for (std::size_t index : valid_indices) {
    CandidateResult& candidate = candidates[index];
    std::cerr << "[tune] finalist split=" << candidate.n1 << "x" << candidate.n2 << " benchmarking\n";
    candidate.final = benchmark_phase(harness, candidate, options.final_warmup, options.final_iters);
    candidate.status = "valid";
  }
  std::sort(valid_indices.begin(), valid_indices.end(), [&](std::size_t lhs, std::size_t rhs) {
    return candidates[lhs].final->objective_ms < candidates[rhs].final->objective_ms;
  });
  CandidateResult& winner = candidates[valid_indices.front()];
  winner.status = "winner";

  TimingResult reference_forward =
      harness.benchmark_reference(FLAGFFT_FORWARD, options.final_warmup, options.final_iters);
  TimingResult reference_inverse =
      harness.benchmark_reference(FLAGFFT_INVERSE, options.final_warmup, options.final_iters);

  const std::string run_id = std::to_string(unix_time_millis());
  std::optional<std::filesystem::path> db_path;
  if (options.save) {
    db_path = result_db_path(options);
    persist_results(*db_path, run_id, candidates, winner, forward_request, inverse_request, options);
  }

  nlohmann::json candidate_reports = nlohmann::json::array();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    candidate_reports.push_back(candidate_json(candidates[index], index));
  }
  const PhaseResult& winner_final = winner.final.value();
  return {
      {         "status",            "passed"                         },
      {        "command",                                       "tune"},
      {           "mode",                              "decomposition"},
      {         "run_id",                                       run_id},
      {          "shape",                             {options.length}},
      {            "api",                                        "c2c"},
      {          "batch",                                options.batch},
      {"candidate_count",                            candidates.size()},
      {     "candidates",                 std::move(candidate_reports)},
      {         "winner",
       {
       {"plan_key", winner.key},
       {"split", {winner.n1, winner.n2}},
       {"timing", phase_json(winner_final)},
       {"plan", plan_node_to_json(winner.candidate.node)},
       }                                                              },
      {      "reference",
       {
       {"forward", timing_json(reference_forward)},
       {"inverse", timing_json(reference_inverse)},
       }                                                              },
      {        "db_path", db_path.has_value() ? db_path->string() : ""},
  };
}

}  // namespace flagfft::cli::tune
