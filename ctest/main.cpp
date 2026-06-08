#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "flagfft_test.h"

using flagfft_test::g_test_params;

flagfft_test::TestParams flagfft_test::g_test_params {};

int main(int argc, char** argv) {
  // Parse custom parameters (before GTest initialization)
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--nx=", 5) == 0) {
      g_test_params.nx = atoi(argv[i] + 5);
    } else if (strncmp(argv[i], "--ny=", 5) == 0) {
      g_test_params.ny = atoi(argv[i] + 5);
    } else if (strncmp(argv[i], "--batch=", 8) == 0) {
      g_test_params.batch = atoi(argv[i] + 8);
    } else if (strncmp(argv[i], "--direction=", 12) == 0) {
      const char* dir = argv[i] + 12;
      if (strcmp(dir, "forward") == 0 || strcmp(dir, "fwd") == 0)
        g_test_params.direction = 0;
      else if (strcmp(dir, "inverse") == 0 || strcmp(dir, "inv") == 0)
        g_test_params.direction = 1;
    } else if (strncmp(argv[i], "--scale=", 8) == 0) {
      g_test_params.scale = atof(argv[i] + 8);
    } else if (strncmp(argv[i], "--json-file=", 12) == 0) {
      g_test_params.json_file = argv[i] + 12;
      g_test_params.json_output = true;
    }
  }

  ::testing::InitGoogleTest(&argc, argv);

  // Preserve existing FLAGFFT_TEST_PROFILE support
  if (g_test_params.nx == 0 && g_test_params.batch == 0) {
    // Only apply profile when no explicit params
    if (const char* profile = std::getenv("FLAGFFT_TEST_PROFILE")) {
      if (std::strcmp(profile, "smoke") == 0) {
        ::testing::GTEST_FLAG(filter) = "Plan*:Smoke*:*Smoke*";
      } else if (std::strcmp(profile, "full") != 0) {
        std::cerr << "Unknown FLAGFFT_TEST_PROFILE: " << profile << " (expected smoke or full)\n";
        return 2;
      }
    }
  }

  flagfft::test_adaptor::initialize();

  // Register JSON result listener
  if (g_test_params.json_output) {
    auto* listener = new flagfft_test::JsonTestListener(g_test_params.json_file);
    ::testing::UnitTest::GetInstance()->listeners().Append(listener);
  }

  return RUN_ALL_TESTS();
}
