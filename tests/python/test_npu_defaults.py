"""CPU checks for target-local Ascend defaults, overrides and cache identity."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


PROGRAM = r'''
#include "flagfft/npu_defaults.hpp"
#include <cassert>
#include <future>
#include <iostream>
#include <string>

using namespace flagfft;
std::string value(const char *name) {
  const char *setting = npu_performance_env(name);
  return setting ? setting : "unset";
}

int main(int argc, char **argv) {
  const std::string mode = argv[1];
  const char *key = "FLAGFFT_NPU_VECTOR_DIF";
  if (mode == "defaults") {
    assert(value(key) == "unset");
    const auto baseline = npu_performance_identity();
    for (const auto &arch : {"Ascend910B", "Ascend910B4", "Ascend910B4-1"}) {
      NpuPerformanceScope policy("npu", arch, "complex64");
      assert(value(key) == "1");
      assert(value("FLAGFFT_NPU_DIF_TRANSPOSE_TILE") == "64");
      assert(value("FLAGFFT_NPU_STRUCTURED_PACK") == "32");
      assert(value("FLAGFFT_NPU_GRAPH") == "auto");
      assert(value("FLAGFFT_NPU_DIF_DENSE_SHORT_PREFIX") == "0");
      assert(value("FLAGFFT_NPU_GRAPH_TRACE") == "unset");
      assert(npu_performance_identity() != baseline);
      assert(std::getenv(key) == nullptr);
      assert(std::async(std::launch::async, [key] { return value(key); }).get() == "unset");
      {
        NpuPerformanceScope other("cuda", "sm_80", "complex64");
        assert(value(key) == "unset");
      }
      assert(value(key) == "1");
      try {
        NpuPerformanceScope other("npu", "Ascend910C", "complex64");
        assert(value(key) == "unset");
        throw 1;
      } catch (int) {}
      assert(value(key) == "1");
    }
    assert(value(key) == "unset");
    for (const auto &backend : {"cuda", "musa", "maca", "hcu", "ix"}) {
      NpuPerformanceScope other(backend, "Ascend910B4", "complex64");
      assert(value(key) == "unset");
    }
    for (const auto &dtype : {"complex128", "float64", "float16"}) {
      NpuPerformanceScope other("npu", "Ascend910B4", dtype);
      assert(value(key) == "unset");
    }
    { NpuPerformanceScope other("npu:Ascend910B4:1", "complex64"); assert(value(key) == "1"); }
    { NpuPerformanceScope other("npu:Ascend910C:1", "complex64"); assert(value(key) == "unset"); }
  } else if (mode == "override") {
    NpuPerformanceScope policy("npu", "Ascend910B4", "complex64");
    assert(value(key) == "0");
    assert(value("FLAGFFT_NPU_GRAPH") == "0");
    assert(value("FLAGFFT_NPU_DIF_TRANSPOSE_TILE") == "32");
    assert(value("FLAGFFT_NPU_TENSOR_FFT") == "1");
  } else if (mode == "rollback") {
    NpuPerformanceScope policy("npu", "Ascend910B4", "complex64");
    assert(value(key) == (std::getenv(key) ? "1" : "unset"));
    assert(value("FLAGFFT_NPU_GRAPH") == "unset");
  } else if (mode == "invalid") {
    bool rejected = false;
    try { NpuPerformanceScope policy("npu", "Ascend910B4", "complex64"); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    assert(!npu_performance_defaults_active);
  }
  std::cout << "ok\n";
}
'''


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    temp = tmp_path_factory.mktemp("npu-defaults")
    source = temp / "probe.cpp"
    binary = temp / "probe"
    source.write_text(PROGRAM)
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(root / "src/utils/include"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("mode,overrides", [
    ("defaults", {}),
    ("override", {"FLAGFFT_NPU_VECTOR_DIF": "0", "FLAGFFT_NPU_GRAPH": "0",
                  "FLAGFFT_NPU_DIF_TRANSPOSE_TILE": "32"}),
    ("rollback", {"FLAGFFT_NPU_OPTIMIZED_DEFAULTS": "0"}),
    ("rollback", {"FLAGFFT_NPU_OPTIMIZED_DEFAULTS": "0", "FLAGFFT_NPU_VECTOR_DIF": "1"}),
    ("invalid", {"FLAGFFT_NPU_OPTIMIZED_DEFAULTS": "invalid"}),
])
def test_native_defaults(native_probe, mode, overrides):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FLAGFFT_NPU_")}
    environment.update(overrides)
    result = subprocess.run([str(native_probe), mode], env=environment,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


PLANNER_PROGRAM = r'''
#include "flagfft/core.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

namespace flagfft::adaptor {
std::string backend_name() { return "cpu-check"; }
int64_t max_dynamic_smem_bytes(int) { return 192 * 1024; }
}

int main() {
  flagfft::FFTRequest request;
  request.fft_length = request.requested_n = 46189;
  request.device_type = "npu";
  request.device_arch = "Ascend910B4";
  request.input_dtype = request.output_dtype = "complex64";
  request.input_layout = "contiguous";
  request.raw_dim = request.origin_rank = 1;
  request.batch = 64;
  request.input_shape = {64, 46189};
  request.input_strides = {46189, 1};
  request.direction = "forward";
  const auto key = [](const flagfft::PlanNodePtr &node) {
    return flagfft::PlanKey::from_node(node).repr();
  };
  flagfft::PlanBuilder builder, tuner;
  const auto optimized = key(builder.build(46189, request));
  // Tuning must use the same defaults before any ordinary build has run,
  // and its child-node cache must not poison the following public build.
  assert(key(tuner.build_decomposition_tune_candidates(46189, request, 4).front().node) == optimized);
  assert(key(tuner.build(46189, request)) == optimized);
  assert(!flagfft::npu_performance_defaults_active);
  assert(std::getenv("FLAGFFT_NPU_VECTOR_DIF") == nullptr);
  setenv("FLAGFFT_NPU_OPTIMIZED_DEFAULTS", "0", 1);
  const auto original = key(builder.build(46189, request));
  assert(original != optimized);
  // Legacy tuning and heuristic builds can rank alternatives differently.
  // Compare each reused cache with a fresh planner using the same entry point.
  flagfft::PlanBuilder fresh_builder, fresh_tuner;
  assert(original == key(fresh_builder.build(46189, request)));
  const auto original_tune = key(fresh_tuner.build_decomposition_tune_candidates(46189, request, 4).front().node);
  assert(key(tuner.build_decomposition_tune_candidates(46189, request, 4).front().node) == original_tune);
  assert(key(tuner.build(46189, request)) == key(fresh_tuner.build(46189, request)));
  unsetenv("FLAGFFT_NPU_OPTIMIZED_DEFAULTS");
  assert(key(builder.build(46189, request)) == optimized);
  assert(key(tuner.build(46189, request)) == optimized);
  assert(!flagfft::npu_performance_defaults_active);
  std::cout << "planner ok\n";
}
'''

PLANNER_SOURCES = [
    "src/exec/c_api_internal.cpp", "src/utils/base.cpp", "src/utils/plan_key.cpp",
    "src/plan/nodes.cpp", "src/plan/builder_context.cpp", "src/plan/factorization.cpp",
    "src/plan/rader_utils.cpp", "src/plan/cost_model.cpp", "src/plan/auto_candidates.cpp",
    "src/plan/tune_candidates.cpp", "src/codegen/request_policy.cpp",
]


def test_planner_defaults_and_cache(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "planner.cpp"
    binary = tmp_path / "planner"
    source.write_text(PLANNER_PROGRAM)
    command = [compiler, "-std=c++20", "-O0"]
    for include in ["include", "src", "src/utils/include"]:
        command.extend(["-I", str(root/include)])
    command.extend([str(source), *(str(root/p) for p in PLANNER_SOURCES), "-o", str(binary)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FLAGFFT_NPU_")}
    result = subprocess.run([str(binary)], env=environment, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "planner ok"
