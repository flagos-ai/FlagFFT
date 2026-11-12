// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace flagfft::validation {

// Reserve the original stdout pipe for binary results before importing any
// runtime/compiler. Python print(), printf() and C++ diagnostics then share
// stderr instead of corrupting the numerical protocol. This standalone tool
// deliberately keeps fd 1 redirected until process exit, including Python's
// interpreter teardown and its final buffered log flushes.
class BinaryStdoutChannel {
 public:
  BinaryStdoutChannel() {
#if defined(__linux__)
    std::cout.flush();
    std::fflush(stdout);
    const int saved = ::dup(STDOUT_FILENO);
    if (saved < 0) fail("dup(stdout)");
    if (::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      const int error = errno;
      ::close(saved);
      errno = error;
      fail("redirect stdout diagnostics");
    }
    output_.open("/proc/self/fd/" + std::to_string(saved), std::ios::binary);
    ::close(saved);
    if (!output_) throw std::runtime_error("cannot open binary stdout channel");
#else
    throw std::runtime_error("binary stdout capture requires Linux");
#endif
  }

  std::ostream& stream() { return output_; }

 private:
  [[noreturn]] static void fail(const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
  }
  std::ofstream output_;
};

}  // namespace flagfft::validation
