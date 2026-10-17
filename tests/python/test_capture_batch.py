"""Full-batch validation must not silently qualify a smaller execution plan."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_capture_chunk_policy_preserves_explicit_full_batch(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('a host C++ compiler is required')
    source = tmp_path / 'batch_capture.cpp'
    source.write_text(r'''
#include "batch_capture.hpp"
#include <cassert>
using flagfft::validation::capture_batch_chunk;
int main() {
  const auto big = 4ULL * 128 * 2048 * 64 * 8;
  assert(capture_batch_chunk(4, big, big, false) == 1);
  assert(capture_batch_chunk(4, big, big, true) == 4);
  const auto small = 4ULL * 16 * 16 * 16 * 8;
  assert(capture_batch_chunk(4, small, small, false) == 4);
  assert(capture_batch_chunk(4, small, small, true) == 4);
  assert(capture_batch_chunk(65536, 65536ULL * 129 * 8, 65536ULL * 256 * 4, false) == 65280);
  assert(capture_batch_chunk(65536, 65536ULL * 129 * 8, 65536ULL * 256 * 4, true) == 65536);
  assert(capture_batch_chunk(4, 0, 0, false) == 4);
  assert(capture_batch_chunk(1, big, big, false) == 1);
}
''')
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / 'batch_capture'
    subprocess.run([compiler,'-std=c++17','-I',str(root/'tools/numpy_fft_validation'),
                    str(source),'-o',str(executable)],check=True,capture_output=True)
    subprocess.run([str(executable)],check=True,capture_output=True)
