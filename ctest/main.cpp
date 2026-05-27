#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "flagfft_test.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (const char* profile = std::getenv("FLAGFFT_TEST_PROFILE")) {
    if (std::strcmp(profile, "smoke") == 0) {
      ::testing::GTEST_FLAG(filter) = "Plan*:Smoke*:*Smoke*";
    } else if (std::strcmp(profile, "full") != 0) {
      std::cerr << "Unknown FLAGFFT_TEST_PROFILE: " << profile << " (expected smoke or full)\n";
      return 2;
    }
  }
  flagfft::test_adaptor::initialize();
  return RUN_ALL_TESTS();
}
