"""Compiler/runtime diagnostics must never enter the binary capture pipe."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="native capture targets Linux")
def test_binary_stdout_isolated_from_runtime_logs(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    source = tmp_path / "capture_stdout.cpp"
    source.write_text(r'''
#include "stdout_channel.hpp"
#include <cstdlib>

void late_diagnostic() { std::printf("late runtime log\n"); }

int main() {
  flagfft::validation::BinaryStdoutChannel channel;
  std::atexit(late_diagnostic);
  std::printf("C runtime log\n");
  std::cout << "C++ runtime log\n";
  std::fflush(stdout);
  std::cout.flush();
  // A compiler subprocess inherits fd 1. Its output must also be diagnostic.
  if (std::system("printf 'compiler subprocess log\\n'") != 0) return 1;
  const char payload[] = {'\0', '\xff', '\n', '\x7f'};
  channel.stream().write(payload, sizeof(payload));
  channel.stream().flush();
  return channel.stream() ? 0 : 2;
}
''')
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "capture_stdout"
    subprocess.run([compiler, "-std=c++17", "-I",
        str(root / "tools/numpy_fft_validation"), str(source), "-o", str(executable)],
        check=True, capture_output=True)
    result = subprocess.run([str(executable)], capture_output=True, check=True)
    assert result.stdout == b"\0\xff\n\x7f"
    for message in (b"C runtime log", b"C++ runtime log", b"compiler subprocess log",
                    b"late runtime log"):
        assert message in result.stderr
