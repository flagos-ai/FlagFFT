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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cli_tools/bench/report.hpp"
#include "cli_tools/bench/runner.hpp"
#include "cli_tools/common/case.hpp"
#include "cli_tools/tune/tune.hpp"

namespace {

using namespace flagfft::cli;
using namespace flagfft::cli::bench;

struct Args {
  std::string command;
  std::vector<CaseSpec> cases;
  CaseSpec prototype;
  flagfft::cli::tune::TuneOptions tune;
  int warmup = 10;
  int iters = 100;
  bool print_path = false;
  bool json_output = false;
};

void print_usage() {
  std::cout << "Usage: flagfft-cli bench --rank 1|2|3 --shape <format> [options]\n"
               "       flagfft-cli tune [options]\n"
               "\n"
               "Bench options:\n"
               "  --rank 1|2|3             Transform rank (required)\n"
               "  --shape A,B,...           Comma-separated shapes, x for dims\n"
               "                           rank 1: --shape 16,32,64\n"
               "                           rank 2: --shape 23x42,128x64\n"
               "                           rank 3: --shape 128x128x128\n"
               "  --api c2c|z2z|r2c|d2z|c2r|z2d   FFT type (default: c2c)\n"
               "  --batch N                 Batch size for rank 1 and complex rank 2 (default: 1)\n"
               "  --direction forward|inverse       (default: forward)\n"
               "  --placement out-of-place|in-place (default: out-of-place)\n"
               "  --warmup N                Warmup iterations (default: 10)\n"
               "  --iters N                 Timed iterations (default: 100)\n"
               "  --json                    Output JSON report\n"
               "  --print-path              Include plan description in output\n"
               "\n"
               "Tune options (decomposition-only v1):\n"
               "  --shape N                 FFT length (currently 1048576)\n"
               "  --batch N                 Batch size (currently 1)\n"
               "  --max-candidates N        Number of FourStep splits to screen (default: 5)\n"
               "  --finalists N             Candidates receiving the long benchmark (default: 2)\n"
               "  --screen-warmup N         Screening warmup iterations (default: 10)\n"
               "  --screen-iters N          Screening timed iterations (default: 50)\n"
               "  --final-warmup N          Final warmup iterations (default: 50)\n"
               "  --final-iters N           Final timed iterations (default: 1000)\n"
               "  --db PATH                 Tuned-plan SQLite database path\n"
               "  --no-save                 Do not persist trials or the winning plan\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
    print_usage();
    std::exit(0);
  }
  Args args;
  args.command = argv[1];
  if (args.command != "bench" && args.command != "tune") {
    throw AssertionFailure("unknown command: " + args.command);
  }
  if (args.command == "tune") {
    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      auto need = [&](const std::string& name) -> const char* {
        if (++i >= argc) throw AssertionFailure(name + " requires a value");
        return argv[i];
      };
      if (argument == "--help" || argument == "-h") {
        print_usage();
        std::exit(0);
      } else if (argument == "--shape") {
        const auto shape = parse_shape(need("--shape"));
        if (shape.size() != 1) {
          throw AssertionFailure("tune --shape must contain one dimension");
        }
        args.tune.length = shape[0];
      } else if (argument == "--batch") {
        args.tune.batch = parse_positive("--batch", need("--batch"));
      } else if (argument == "--max-candidates") {
        args.tune.max_candidates = parse_positive("--max-candidates", need("--max-candidates"));
      } else if (argument == "--finalists") {
        args.tune.finalists = parse_positive("--finalists", need("--finalists"));
      } else if (argument == "--screen-warmup") {
        args.tune.screen_warmup = parse_positive("--screen-warmup", need("--screen-warmup"), true);
      } else if (argument == "--screen-iters") {
        args.tune.screen_iters = parse_positive("--screen-iters", need("--screen-iters"));
      } else if (argument == "--final-warmup") {
        args.tune.final_warmup = parse_positive("--final-warmup", need("--final-warmup"), true);
      } else if (argument == "--final-iters") {
        args.tune.final_iters = parse_positive("--final-iters", need("--final-iters"));
      } else if (argument == "--db") {
        args.tune.db_path = need("--db");
      } else if (argument == "--no-save") {
        args.tune.save = false;
      } else if (argument == "--json") {
        args.json_output = true;
      } else {
        throw AssertionFailure("unknown tune argument: " + argument);
      }
    }
    return args;
  }

  std::vector<std::vector<int>> shapes;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    auto need = [&](const std::string& name) -> const char* {
      if (++i >= argc) throw AssertionFailure(name + " requires a value");
      return argv[i];
    };
    if (argument == "--help" || argument == "-h") {
      print_usage();
      std::exit(0);
    } else if (argument == "--rank") {
      args.prototype.rank = parse_rank(need("--rank"));
    } else if (argument == "--shape") {
      const std::string raw = need("--shape");
      std::size_t start = 0;
      while (start <= raw.size()) {
        std::size_t end = raw.find(',', start);
        const std::string part = raw.substr(start, end == std::string::npos ? end : end - start);
        if (!part.empty()) shapes.push_back(parse_shape(part));
        if (end == std::string::npos) break;
        start = end + 1;
      }
    } else if (argument == "--api") {
      args.prototype.api = parse_fft_api(need("--api"));
    } else if (argument == "--batch") {
      args.prototype.batch = parse_positive("--batch", need("--batch"));
    } else if (argument == "--direction") {
      args.prototype.direction = parse_direction(need("--direction"));
    } else if (argument == "--placement") {
      args.prototype.placement = parse_placement(need("--placement"));
    } else if (argument == "--warmup") {
      args.warmup = parse_positive("--warmup", need("--warmup"), true);
    } else if (argument == "--iters") {
      args.iters = parse_positive("--iters", need("--iters"));
    } else if (argument == "--json") {
      args.json_output = true;
    } else if (argument == "--print-path") {
      args.print_path = true;
    } else {
      throw AssertionFailure("unknown argument: " + argument);
    }
  }

  if (shapes.empty()) shapes.push_back(args.prototype.shape);
  for (auto& shape : shapes) {
    CaseSpec spec = args.prototype;
    spec.shape = std::move(shape);
    args.cases.push_back(std::move(spec));
  }

  for (const auto& c : args.cases) {
    if (c.rank == 2 && !is_complex_api(c.api)) {
      throw AssertionFailure("rank 2 currently supports only c2c and z2z");
    }
    if (c.rank > 2 && c.batch != 1) {
      throw AssertionFailure("--batch is only supported with --rank 1 or complex --rank 2");
    }
    if (static_cast<int>(c.shape.size()) != c.rank) {
      throw AssertionFailure("--shape dimension count does not match --rank");
    }
    if (is_real_forward_api(c.api) && c.direction != FLAGFFT_FORWARD) {
      throw AssertionFailure(fft_api_name(c.api) + " only supports forward direction");
    }
    if (is_real_inverse_api(c.api) && c.direction != FLAGFFT_INVERSE) {
      throw AssertionFailure(fft_api_name(c.api) + " only supports inverse direction");
    }
  }

  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    std::string reason;
    if (!has_cuda_device(reason)) {
      std::cerr << "error: " << reason << "\n";
      return kExitSkipped;
    }

    if (args.command == "tune") {
      std::cout << flagfft::cli::tune::run_decomposition_tune(args.tune).dump(2) << "\n";
      return 0;
    }

    std::vector<BenchResult> results;
    for (const auto& spec : args.cases) {
      results.push_back(run_benchmark(spec, args.warmup, args.iters, args.print_path));
    }

    if (args.json_output) {
      std::cout << format_json(args.cases, results, args.warmup, args.iters).dump(2) << "\n";
    } else {
      std::cout << format_table(args.cases, results, args.warmup, args.iters);
    }

    return 0;
  } catch (const flagfft::cli::CliException& error) {
    std::cerr << "error: " << error.what() << "\n";
    return error.exit_code();
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return flagfft::cli::kExitRuntimeError;
  }
}
