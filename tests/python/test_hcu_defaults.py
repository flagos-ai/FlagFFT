"""Native CPU checks for BW1000 default policy, isolation and profile equivalence."""
from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest


PROGRAM = r'''
#include "flagfft/performance_defaults.hpp"
#include <cassert>
#include <future>
#include <iostream>

using namespace flagfft;
std::string value(const char *name) {
  const char *setting = performance_env(name);
  return setting ? setting : "unset";
}

int main(int argc, char **argv) {
  const std::string mode = argv[1];
  const char *key = "FLAGFFT_HCU_DENSE_WAVE";
  if (mode == "defaults") {
    assert(value(key) == "unset");
    for (const char *dtype : {"complex64", "float32", "complex128", "float64"}) {
      BackendPerformanceScope policy("hcu", "gfx936", dtype);
      assert(value(key) == "packed");
      assert(value("FLAGFFT_HCU_PLAN_GRAPH") == "real_2d");
      assert(value("FLAGFFT_TUNE_DISABLE") == "1");
      assert(value("FLAGFFT_NUMPY_CAPTURE_FULL_BATCH") == "unset");
      assert(value("FLAGFFT_HCU_3D_HYBRID") == "unset");
      for (const auto &setting : kHcuPerformanceDefaults) {
        assert(value(setting.name) == setting.value);
        assert(std::getenv(setting.name) == nullptr);
      }
      assert(std::async(std::launch::async, [key] { return value(key); }).get() == "unset");
      {
        BackendPerformanceScope other("npu", "Ascend910B4", "complex64");
        assert(value(key) == "unset");
        assert(value("FLAGFFT_TUNE_DISABLE") == "unset");
        assert(value("FLAGFFT_NPU_VECTOR_DIF") == "1");
      }
      assert(value(key) == "packed");
      assert(value("FLAGFFT_NPU_VECTOR_DIF") == "unset");
      try {
        BackendPerformanceScope other("cuda", "sm_80", "complex64");
        assert(value(key) == "unset");
        throw 1;
      } catch (int) {}
      assert(value(key) == "packed");
    }
    for (const char *backend : {"cuda", "musa", "maca", "npu", "ix"}) {
      BackendPerformanceScope other(backend, "gfx936", "complex64");
      assert(value(key) == "unset");
    }
    for (const char *arch : {"gfx90a", "gfx9360", "gfx936-extra", ""}) {
      BackendPerformanceScope other("hcu", arch, "complex64");
      assert(value(key) == "unset");
    }
    { BackendPerformanceScope other("hcu", "gfx936", "float16"); assert(value(key) == "unset"); }
    for (const char *target : {"hcu:gfx936", "hcu:gfx936:64"}) {
      BackendPerformanceScope other(target, "complex128"); assert(value(key) == "packed");
    }
    for (const char *target : {"hcu:gfx9360:64", "cuda:gfx936:64", "hcu"}) {
      BackendPerformanceScope other(target, "complex64"); assert(value(key) == "unset");
    }
  } else if (mode == "override") {
    BackendPerformanceScope policy("hcu", "gfx936", "complex64");
    assert(value(key) == "0");
    assert(value("FLAGFFT_HCU_PLAN_GRAPH") == "0");
    assert(value("FLAGFFT_HCU_WAVE_SCALAR_BUDGET") == "32");
    assert(value("FLAGFFT_TUNE_DISABLE") == "0");
    assert(value("FLAGFFT_HCU_CUSTOM_EXPERIMENT") == "custom");
    assert(value("FLAGFFT_HCU_VECTOR_IO") == "fp32");
  } else if (mode == "rollback") {
    BackendPerformanceScope policy("hcu", "gfx936", "complex64");
    assert(value(key) == (std::getenv(key) ? "packed" : "unset"));
    assert(value("FLAGFFT_HCU_PLAN_GRAPH") == "unset");
    assert(value("FLAGFFT_TUNE_DISABLE") == "unset");
  } else if (mode == "invalid") {
    { BackendPerformanceScope other("cuda", "sm_80", "complex64"); }
    bool rejected = false;
    try { BackendPerformanceScope policy("hcu", "gfx936", "complex64"); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    assert(!hcu_performance_defaults_active && !npu_performance_defaults_active);
  } else if (mode == "identity") {
    BackendPerformanceScope policy("hcu", "gfx936", "complex64");
    std::cout << hcu_performance_identity() << "\n";
    return 0;
  }
  assert(!hcu_performance_defaults_active && !npu_performance_defaults_active);
  std::cout << "ok\n";
}
'''


def clean_environment():
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("FLAGFFT_HCU_", "FLAGFFT_NPU_"))
            and k not in {"FLAGFFT_TUNE_DISABLE", "FLAGFFT_NUMPY_CAPTURE_FULL_BATCH"}}


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    temp = tmp_path_factory.mktemp("hcu-defaults")
    source, binary = temp / "probe.cpp", temp / "probe"
    source.write_text(PROGRAM)
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(root / "src/utils/include"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("mode,overrides", [
    ("defaults", {}),
    ("override", {"FLAGFFT_HCU_DENSE_WAVE": "0", "FLAGFFT_HCU_PLAN_GRAPH": "0",
                  "FLAGFFT_HCU_WAVE_SCALAR_BUDGET": "32", "FLAGFFT_TUNE_DISABLE": "0",
                  "FLAGFFT_HCU_CUSTOM_EXPERIMENT": "custom"}),
    ("rollback", {"FLAGFFT_HCU_OPTIMIZED_DEFAULTS": "0"}),
    ("rollback", {"FLAGFFT_HCU_OPTIMIZED_DEFAULTS": "0", "FLAGFFT_HCU_DENSE_WAVE": "packed"}),
    ("invalid", {"FLAGFFT_HCU_OPTIMIZED_DEFAULTS": "invalid"}),
])
def test_native_defaults(native_probe, mode, overrides):
    result = subprocess.run([str(native_probe), mode], env=clean_environment() | overrides,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_automatic_defaults_match_explicit_qualified_profile(native_probe):
    root = Path(__file__).resolve().parents[2]
    profile = dict(re.findall(r"^export (FLAGFFT_(?:HCU_\w+|TUNE_DISABLE))=(\S+)$",
                             (root / "conf/hcu_bw1000.sh").read_text(), re.M))

    def identity(overrides):
        return subprocess.run([str(native_probe), "identity"],
                              env=clean_environment() | overrides,
                              check=True, capture_output=True, text=True).stdout.strip()

    automatic = identity({})
    # Disabling automatic defaults makes this comparison independent of them.
    explicit = identity(profile | {"FLAGFFT_HCU_OPTIMIZED_DEFAULTS": "0"})
    assert automatic == explicit
    effective = dict(item.split("=", 1) for item in automatic.split(";")[1:])
    assert effective == profile
    assert identity({"FLAGFFT_HCU_OPTIMIZED_DEFAULTS": "0"}) != automatic
    assert identity({"FLAGFFT_HCU_PLAN_GRAPH": "0"}) != automatic


PLANNER_PROGRAM = r'''
#include "flagfft/core.hpp"
#include <cassert>
#include <iostream>

namespace flagfft::adaptor {
std::string backend_name() { return "cpu-check"; }
int64_t max_dynamic_smem_bytes(int) { return 64 * 1024; }
}

int main() {
  using namespace flagfft;
  PlanBuilder reused;
  int changes = 0;
  for (const char *dtype : {"complex64", "complex128"}) {
    for (int64_t n : {64, 512, 4096, 65536, 1155, 323, 46189}) {
      FFTRequest request;
      request.fft_length = request.requested_n = n;
      request.device_type = "hcu";
      request.device_arch = "gfx936";
      request.input_dtype = request.output_dtype = dtype;
      request.input_layout = "contiguous";
      request.raw_dim = request.origin_rank = 1;
      request.batch = 37;
      request.input_shape = {37, n};
      request.input_strides = {n, 1};
      request.direction = "forward";
      const auto build = [&](PlanBuilder &builder) {
        return PlanKey::from_node(builder.build(n, request)).repr();
      };
      const auto automatic = build(reused);
      for (const auto &setting : kHcuPerformanceDefaults)
        setenv(setting.name, setting.value, 1);
      setenv("FLAGFFT_HCU_OPTIMIZED_DEFAULTS", "0", 1);
      PlanBuilder explicit_profile;
      assert(build(explicit_profile) == automatic);
      assert(build(reused) == automatic);
      for (const auto &setting : kHcuPerformanceDefaults) unsetenv(setting.name);
      PlanBuilder legacy;
      const auto original = build(legacy);
      assert(build(reused) == original);
      changes += original != automatic;
      unsetenv("FLAGFFT_HCU_OPTIMIZED_DEFAULTS");
      assert(build(reused) == automatic);
      assert(!hcu_performance_defaults_active && !npu_performance_defaults_active);
    }
  }
  assert(changes > 0);
  std::cout << "planner ok\n";
}
'''


def test_real_planner_matches_profile_and_invalidates_cached_policy(tmp_path):
    from test_npu_defaults import PLANNER_SOURCES

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source, binary = tmp_path / "planner.cpp", tmp_path / "planner"
    source.write_text(PLANNER_PROGRAM)
    command = [compiler, "-std=c++20", "-O0"]
    for include in ["include", "src", "src/utils/include"]:
        command.extend(["-I", str(root / include)])
    command.extend([str(source), *(str(root / p) for p in PLANNER_SOURCES), "-o", str(binary)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], env=clean_environment(), check=True,
                            capture_output=True, text=True)
    assert result.stdout.strip() == "planner ok"
