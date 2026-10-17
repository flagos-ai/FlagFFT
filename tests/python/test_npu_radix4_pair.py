"""Validate the generated pair body against the unchanged native host tables."""

import ast
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.kernels_npu_radix4 import build_npu_radix4_pair_source
from flagfft_codegen.kernels_npu_grouped import build_npu_grouped_fft64_source
from test_npu_grouped_fft64 import Language, Pointer, insert_slice


@pytest.fixture(scope="module")
def native_tables_executable(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A host C++ compiler is needed to extract the original tables")
    repo = Path(__file__).resolve().parents[2]
    code = (repo / "src/codegen/compiler.cpp").read_text()
    start = code.index("  void build_npu_aiv_fft_radix4_pair_tables(")
    end = code.index("  std::shared_ptr<CompiledRawNode> make_npu_aiv_fft256_real_child(", start)
    prelude = """#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
struct FFTRequest { std::string direction; };
enum class NpuAivFFT64Mode { Complex=0, RealForward=1, RealInverse=2 };
"""
    main = """int main(int argc, char** argv) {
  if (argc != 5) return 2;
  FFTRequest request{std::string(argv[4]) == "1" ? "inverse" : "forward"};
  std::vector<uint32_t> index; std::vector<float> twiddle;
  build_npu_aiv_fft_radix4_pair_tables(request, std::atoll(argv[1]),
      std::atoll(argv[2]), static_cast<NpuAivFFT64Mode>(std::atoi(argv[3])), index, twiddle);
  uint64_t lengths[2] = {index.size(), twiddle.size()};
  std::cout.write(reinterpret_cast<const char*>(lengths), sizeof(lengths));
  std::cout.write(reinterpret_cast<const char*>(index.data()), index.size()*sizeof(uint32_t));
  std::cout.write(reinterpret_cast<const char*>(twiddle.data()), twiddle.size()*sizeof(float));
}
"""
    folder = tmp_path_factory.mktemp("radix4_native_host_tables")
    source, executable = folder / "tables.cpp", folder / "tables"
    source.write_text(prelude + code[start:end] + main)
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    return executable


@pytest.mark.parametrize("length,group", [(64, 1), (64, 4), (64, 8), (64, 16), (64, 32),
    (128, 4), (128, 8), (128, 16), (2048, 1)])
@pytest.mark.parametrize("mode,inverse", [(0, False), (0, True), (1, False), (2, True)])
def test_pair_body_uses_original_tables(native_tables_executable, length, group, mode, inverse):
    if length == 2048 and mode != 0:
        with pytest.raises(ValueError):
            build_npu_radix4_pair_source(length=length, group=group, mode=mode)
        return
    raw = subprocess.check_output([str(native_tables_executable), str(length),
                                   str(group), str(mode), str(int(inverse))])
    ni, nt = struct.unpack("<QQ", raw[:16])
    index = np.frombuffer(raw, np.uint32, ni, 16).copy()
    twiddle = np.frombuffer(raw, np.float32, nt, 16 + ni * 4).copy()
    rng = np.random.default_rng(20261018)
    count = 3 * group
    real = rng.standard_normal((count, length)).astype(np.float32)
    if mode == 2:
        z = np.fft.rfft(real, axis=1).astype(np.complex64)
        data = z.view(np.float32)
        expected = (np.fft.irfft(z, n=length, axis=1) * length).astype(np.float32)
    elif mode == 1:
        data = real
        expected = np.fft.rfft(real, axis=1).astype(np.complex64).view(np.float32)
    else:
        z = (real + 1j * rng.standard_normal(real.shape)).astype(np.complex64)
        data = z.view(np.float32)
        expected = ((np.fft.ifft(z, axis=1) * length) if inverse else
                    np.fft.fft(z, axis=1)).astype(np.complex64).view(np.float32)
    name, body = build_npu_radix4_pair_source(length=length, group=group, mode=mode)
    if length == 64:
        integrated = build_npu_grouped_fft64_source(group=group, stride=1, mode=mode, radix4_pair=True)
        assert integrated.replace("def npu_grouped_fft64_kernel(", f"def {name}(") == body
    tree = ast.parse(body)
    tree.body[0].decorator_list = []
    language = Language()
    namespace = {"tl": language, "tle": SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice))}
    exec(compile(tree, name, "exec"), namespace)
    output = np.full(expected.size + 128, 31415.25, np.float32)
    for pid in range(3):
        language.pid = pid
        namespace[name](Pointer(data.reshape(-1)), Pointer(output, 64), Pointer(index), Pointer(twiddle), 3)
    actual, reference = output[64:-64].reshape(expected.shape).astype(np.float64), expected.astype(np.float64)
    delta = actual - reference
    assert np.isfinite(actual).all()
    assert np.max(np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)) < 1e-6
    assert np.max(np.max(np.abs(delta), axis=1) / np.maximum(np.max(np.abs(reference), axis=1), 1e-30)) < 1e-6
    assert np.all(output[:64] == 31415.25) and np.all(output[-64:] == 31415.25)


@pytest.mark.parametrize("backend,dtype", [("npu", "complex64"), ("cuda", "complex64"),
    ("ix", "complex64"), ("maca", "complex64"), ("npu", "complex128")])
def test_pair_cli_scope(tmp_path, backend, dtype):
    repo = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "flagfft_codegen.jit_source", "--kernel", "npu_grouped_fft64",
        "--npu-fft-radix4-pair", "--npu-fft-group", "32", "--npu-fft-stride", "1",
        "--npu-fft-mode", "2", "--direction", "inverse", "--dtype", dtype,
        "--target", f"{backend}:source-contract:32", "--out-dir", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(repo / "python"), PYTHONDONTWRITEBYTECODE="1"))
    if backend != "npu" or dtype != "complex64":
        assert result.returncode != 0 and "NPU FP32" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    meta = json.loads(result.stdout)
    assert meta["radix4_pair"] and meta["batch_per_block"] == 32
    assert meta["signature"] == "*fp32:16,*fp32:16,*i32:16,*fp32:16,i32"
    assert meta["module_path"].endswith("_r4.py")


def test_pair_rejects_strided_rows():
    with pytest.raises(ValueError, match="contiguous"):
        build_npu_grouped_fft64_source(group=32, stride=64, mode=0, radix4_pair=True)
