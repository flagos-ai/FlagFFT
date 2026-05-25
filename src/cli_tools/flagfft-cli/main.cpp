#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cli_tools/common/case.hpp"
#include "cli_tools/common/runner.hpp"
#include "cli_tools/tune/driver.hpp"
#include "flagfft/core.hpp"

namespace {

using namespace flagfft::cli;

struct Args {
    std::string command;
    std::string suite = "correctness";
    std::vector<CaseSpec> cases;
    CaseSpec prototype;
    TimingConfig timing;
    std::string db;
    int static_limit = 32;
    int finalists = 3;
    bool retune = false;
    bool print_path = false;
    bool json_output = false;
};

void print_usage() {
    std::cout <<
        "Usage: flagfft-cli test --suite plan|api-errors|correctness|all [case options]\n"
        "       flagfft-cli bench [case options] --warmup W --iters K --launches-per-sample K [--print-path]\n"
        "       flagfft-cli tune [case options] --db PATH --warmup W --iters K --static-limit N --finalists N [--retune]\n"
        "\nCase options: --api c2c|z2z|r2c|d2z|c2r|z2d --shape N|NxM|NxMxK (repeatable)\n"
        "              --batch B --direction forward|inverse --placement out-of-place|in-place\n"
        "              --plan-api plan1d|plan2d|plan3d|planmany --stream --json\n";
}

int parse_positive(const std::string &name, const char *value, bool allow_zero = false) {
    try {
        const std::string token = value;
        std::size_t consumed = 0;
        const int result = std::stoi(token, &consumed);
        if (consumed != token.size()) {
            throw AssertionFailure("invalid value for " + name);
        }
        if ((allow_zero && result < 0) || (!allow_zero && result <= 0)) {
            throw AssertionFailure(name + " must be " + (allow_zero ? "non-negative" : "positive"));
        }
        return result;
    } catch (const std::invalid_argument &) {
        throw AssertionFailure("invalid value for " + name);
    } catch (const std::out_of_range &) {
        throw AssertionFailure("invalid value for " + name);
    }
}

Args parse_args(int argc, char **argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage();
        std::exit(0);
    }
    Args args;
    args.command = argv[1];
    if (args.command != "test" && args.command != "bench" && args.command != "tune") {
        throw AssertionFailure("unknown command: " + args.command);
    }
    std::vector<std::vector<int>> shapes;
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        auto need = [&](const std::string &name) -> const char * {
            if (++i >= argc) throw AssertionFailure(name + " requires a value");
            return argv[i];
        };
        if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else if (argument == "--suite") {
            args.suite = need("--suite");
        } else if (argument == "--api") {
            args.prototype.api = parse_fft_api(need("--api"));
        } else if (argument == "--shape") {
            shapes.push_back(parse_shape(need("--shape")));
        } else if (argument == "--batch") {
            args.prototype.batch = parse_positive("--batch", need("--batch"));
        } else if (argument == "--direction") {
            args.prototype.direction = parse_direction(need("--direction"));
        } else if (argument == "--placement") {
            args.prototype.placement = parse_placement(need("--placement"));
        } else if (argument == "--plan-api") {
            args.prototype.plan_api = parse_plan_api(need("--plan-api"));
        } else if (argument == "--stream") {
            args.prototype.stream = true;
        } else if (argument == "--json") {
            args.json_output = true;
        } else if (argument == "--warmup") {
            args.timing.warmup = parse_positive("--warmup", need("--warmup"), true);
        } else if (argument == "--iters") {
            args.timing.iters = parse_positive("--iters", need("--iters"));
        } else if (argument == "--launches-per-sample") {
            args.timing.launches_per_sample = parse_positive("--launches-per-sample", need("--launches-per-sample"));
        } else if (argument == "--print-path") {
            args.print_path = true;
        } else if (argument == "--db") {
            args.db = need("--db");
        } else if (argument == "--static-limit") {
            args.static_limit = parse_positive("--static-limit", need("--static-limit"));
        } else if (argument == "--finalists") {
            args.finalists = parse_positive("--finalists", need("--finalists"));
        } else if (argument == "--retune") {
            args.retune = true;
        } else {
            throw AssertionFailure("unknown argument: " + argument);
        }
    }
    if (args.command == "test" && args.suite != "plan" && args.suite != "api-errors" &&
        args.suite != "correctness" && args.suite != "all") {
        throw AssertionFailure("unknown test suite: " + args.suite);
    }
    if (args.command == "tune" && args.db.empty()) {
        throw AssertionFailure("tune requires --db PATH");
    }
    if (shapes.empty()) shapes.push_back(args.prototype.shape);
    for (auto &shape : shapes) {
        CaseSpec spec = args.prototype;
        spec.shape = std::move(shape);
        args.cases.push_back(std::move(spec));
    }
    return args;
}

flagfft::FFTRequest request_for(int64_t n, std::string dtype = "complex64") {
    flagfft::FFTRequest request;
    request.fft_length = n;
    request.input_shape = {1, n};
    request.input_strides = {n, 1};
    request.requested_n = n;
    request.raw_dim = 1;
    request.normalized_dim = 1;
    request.norm = "backward";
    request.input_dtype = dtype;
    request.output_dtype = dtype;
    request.device_type = "cpu";
    request.device_index = 0;
    request.device_arch = "test";
    request.input_layout = "contiguous";
    request.direction = "forward";
    request.batch = 1;
    return request;
}

json run_plan_suite() {
    flagfft::PlanBuilder builder;
    auto leaf = builder.build(16, request_for(16));
    expect_true(leaf->kind == flagfft::PlanNodeKind::CtLeaf, "length 16 must use a leaf plan");
    auto four_step = builder.build(8192, request_for(8192));
    expect_true(four_step->kind == flagfft::PlanNodeKind::FourStep, "length 8192 must use four-step");
    auto bluestein = builder.build(331, request_for(331));
    expect_true(bluestein->kind == flagfft::PlanNodeKind::Bluestein, "length 331 must use Bluestein");
    auto nested = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(
        builder.build(1 << 23, request_for(1 << 23)));
    expect_true(nested != nullptr, "large length must use four-step");
    expect_true(nested->row_plan->kind == flagfft::PlanNodeKind::FourStep ||
                nested->col_plan->kind == flagfft::PlanNodeKind::FourStep,
                "large four-step plan must contain a nested route");
    const std::string target = flagfft::adaptor::triton_target("sm_80");
    const auto fwd = flagfft::KernelKey::leaf(target, "forward", "complex64", 16, {16}, 1, 1, {}, 0);
    const auto inv = flagfft::KernelKey::leaf(target, "inverse", "complex64", 16, {16}, 1, 1, {}, 0);
    const auto fp64 = flagfft::KernelKey::leaf(target, "forward", "complex128", 16, {16}, 1, 1, {}, 0);
    expect_true(!(fwd == inv) && !(fwd == fp64), "kernel keys must distinguish direction and dtype");
    return {{"name", "plan"}, {"assertions", 6}, {"status", "passed"}};
}

json run_api_errors_suite() {
    flagfftHandle plan = nullptr;
    expect_flagfft(flagfftPlan1d(nullptr, 16, FLAGFFT_C2C, 1), FLAGFFT_INVALID_VALUE,
                   "null plan output");
    expect_flagfft(flagfftPlan2d(&plan, 8, 8, FLAGFFT_C2C), FLAGFFT_NOT_SUPPORTED,
                   "unsupported plan2d");
    expect_flagfft(flagfftPlan3d(&plan, 4, 4, 4, FLAGFFT_C2C), FLAGFFT_NOT_SUPPORTED,
                   "unsupported plan3d");
    int dims[2] = {8, 8};
    expect_flagfft(flagfftPlanMany(&plan, 2, dims, nullptr, 1, 64, nullptr, 1, 64,
                                   FLAGFFT_C2C, 1), FLAGFFT_NOT_SUPPORTED,
                   "unsupported rank-2 planmany");
    check_flagfft(flagfftPlan1d(&plan, 16, FLAGFFT_C2C, 1), "create validation plan");
    expect_flagfft(flagfftExecC2C(plan, nullptr, nullptr, FLAGFFT_FORWARD), FLAGFFT_INVALID_VALUE,
                   "null exec buffers");
    expect_flagfft(flagfftExecC2C(plan, reinterpret_cast<flagfftComplex *>(0x1),
                                  reinterpret_cast<flagfftComplex *>(0x2), 0),
                   FLAGFFT_INVALID_VALUE, "invalid direction");
    expect_flagfft(flagfftExecZ2Z(plan, reinterpret_cast<flagfftDoubleComplex *>(0x1),
                                  reinterpret_cast<flagfftDoubleComplex *>(0x2), FLAGFFT_FORWARD),
                   FLAGFFT_INVALID_TYPE, "mismatched exec API");
    check_flagfft(flagfftDestroy(plan), "destroy validation plan");
    int one[1] = {16};
    int padded[1] = {18};
    check_flagfft(flagfftPlanMany(&plan, 1, one, padded, 1, 18, nullptr, 1, 9,
                                  FLAGFFT_R2C, 1), "supported padded r2c planmany");
    check_flagfft(flagfftDestroy(plan), "destroy padded plan");
    return {{"name", "api-errors"}, {"assertions", 9}, {"status", "passed"}};
}

json unsupported_report(const Args &args, const CaseSpec &spec, const std::string &reason) {
    return {{"command", args.command}, {"status", "unsupported"}, {"reason", reason},
            {"cases", json::array({case_json(spec)})}};
}

int execute(const Args &args) {
    json report = {{"command", args.command}, {"status", "passed"}};
    std::vector<CaseSpec> cases = args.cases;
    if (args.command == "test" && args.suite == "all") {
        CaseSpec c2c;
        CaseSpec blue = c2c; blue.shape = {331};
        CaseSpec four = c2c; four.shape = {8192};
        CaseSpec z2z = c2c; z2z.api = FftApi::Z2Z; z2z.direction = FLAGFFT_INVERSE;
        CaseSpec r2c = c2c; r2c.api = FftApi::R2C;
        CaseSpec d2z = c2c; d2z.api = FftApi::D2Z; d2z.placement = Placement::InPlace;
        d2z.plan_api = PlanApi::PlanMany;
        CaseSpec c2r = c2c; c2r.api = FftApi::C2R; c2r.direction = FLAGFFT_INVERSE;
        c2r.placement = Placement::InPlace; c2r.plan_api = PlanApi::PlanMany;
        CaseSpec z2d = c2c; z2d.api = FftApi::Z2D; z2d.direction = FLAGFFT_INVERSE;
        cases = {c2c, blue, four, z2z, r2c, d2z, c2r, z2d};
    }
    if (args.command == "test" && (args.suite == "plan" || args.suite == "all")) {
        report["suites"] = json::array({run_plan_suite()});
        if (args.suite == "plan") return emit_json_report(report);
    }
    const bool run_cases = args.command != "test" || args.suite == "correctness" || args.suite == "all";
    if (run_cases) {
        const Operation operation = args.command == "bench" ? Operation::Bench
                                    : args.command == "tune" ? Operation::Tune : Operation::Test;
        for (const auto &spec : cases) {
            const std::string reason = unsupported_reason(spec, operation);
            if (!reason.empty()) return emit_json_report(unsupported_report(args, spec, reason));
        }
    }
    if ((args.command == "test" && args.suite == "api-errors") || args.suite == "all" || run_cases) {
        std::string reason;
        if (!has_cuda_device(reason)) {
            report["status"] = "skipped";
            report["reason"] = reason;
            return emit_json_report(report);
        }
    }
    if (args.command == "test" && (args.suite == "api-errors" || args.suite == "all")) {
        if (!report.contains("suites")) report["suites"] = json::array();
        report["suites"].push_back(run_api_errors_suite());
        if (args.suite == "api-errors") return emit_json_report(report);
    }
    if (args.command == "test") {
        json results = json::array();
        for (const auto &spec : cases) results.push_back(run_correctness(spec, args.print_path));
        for (const auto &result : results) {
            if (!result.at("correctness").at("passed").get<bool>()) report["status"] = "failed";
        }
        report["cases"] = std::move(results);
    } else if (args.command == "bench") {
        json results = json::array();
        for (const auto &spec : cases) results.push_back(run_benchmark(spec, args.timing, args.print_path));
        for (const auto &result : results) {
            if (!result.at("correctness").at("passed").get<bool>()) report["status"] = "failed";
        }
        report["cases"] = std::move(results);
    } else {
        report["db"] = args.db;
        report["retune"] = args.retune;
        report["cases"] = run_tune_cases(cases, args.db, args.timing.warmup, args.timing.iters,
                                         args.static_limit, args.finalists, args.retune);
    }
    return emit_json_report(report);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return execute(parse_args(argc, argv));
    } catch (const flagfft::cli::CliException &error) {
        return flagfft::cli::emit_json_report({
            {"status", error.exit_code() == flagfft::cli::kExitRuntimeError ? "error" : "failed"},
            {"error", error.what()}, {"_exit_code", error.exit_code()},
        });
    } catch (const std::exception &error) {
        return flagfft::cli::emit_json_report({
            {"status", "error"}, {"error", error.what()},
            {"_exit_code", flagfft::cli::kExitRuntimeError},
        });
    }
}
