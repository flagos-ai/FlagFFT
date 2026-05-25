#include <gtest/gtest.h>

#include "flagfft_test.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    flagfft_test::adaptor::initialize();
    return RUN_ALL_TESTS();
}
