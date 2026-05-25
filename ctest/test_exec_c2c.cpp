#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

// Tolerance for single-precision complex FFT comparison
constexpr double kRelTol = 1e-4;

// =========================================================================
// Helper
// =========================================================================

static bool has_ref() { return backend_name() != "null"; }

// =========================================================================
// 1D C2C
// =========================================================================

TEST(C2C_1D, ForwardPowerOfTwo) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, 1);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftComplex> h_out(N);
        std::vector<flagfftComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(C2C_1D, InversePowerOfTwo) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, 1);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_INVERSE);

        std::vector<flagfftComplex> h_out(N);
        std::vector<flagfftComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(C2C_1D, RoundtripForwardInverse) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, 1);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_mid = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_mid, FLAGFFT_FORWARD);
    ExecC2C(plan, d_mid, d_out, FLAGFFT_INVERSE);

    std::vector<flagfftComplex> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));

    // Roundtrip: H_out = F^-1(F(H_in)) = N * H_in
    for (int i = 0; i < N; ++i) {
        double expected_x = h_in[i].x * N;
        double expected_y = h_in[i].y * N;
        EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol)
            << "Mismatch at index " << i << " (real)";
        EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol)
            << "Mismatch at index " << i << " (imag)";
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan);
}

TEST(C2C_1D, NonPowerOfTwo) {
    constexpr int N = 243;  // 3^5
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, 1);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftComplex> h_out(N);
        std::vector<flagfftComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(C2C_1D, Batch) {
    constexpr int N = 128;
    constexpr int B = 4;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, B);

    int total = N * B;
    auto h_in = random_complex(total);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(total * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(total * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(total * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, total * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_C2C, B);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftComplex> h_out(total);
        std::vector<flagfftComplex> h_ref_out(total);
        copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), total * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

// =========================================================================
// 2D C2C
// =========================================================================

TEST(C2C_2D, ForwardSmall) {
    constexpr int NX = 32;
    constexpr int NY = 16;
    constexpr int N = NX * NY;
    flagfftHandle plan = nullptr;
    Plan2d(&plan, NX, NY, FLAGFFT_C2C);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_2d(ref, NX, NY, FLAGFFT_C2C);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftComplex> h_out(N);
        std::vector<flagfftComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

// =========================================================================
// 3D C2C
// =========================================================================

TEST(C2C_3D, ForwardSmall) {
    constexpr int NX = 16;
    constexpr int NY = 8;
    constexpr int NZ = 4;
    constexpr int N = NX * NY * NZ;
    flagfftHandle plan = nullptr;
    Plan3d(&plan, NX, NY, NZ, FLAGFFT_C2C);

    auto h_in = random_complex(N);
    auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));

    ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_3d(ref, NX, NY, NZ, FLAGFFT_C2C);
        ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftComplex> h_out(N);
        std::vector<flagfftComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}
