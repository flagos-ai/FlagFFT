#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;  // double precision

static bool has_ref() { return backend_name() != "null"; }

TEST(Z2Z_1D, ForwardPowerOfTwo) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

    auto h_in = random_double_complex(N);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
        ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftDoubleComplex> h_out(N);
        std::vector<flagfftDoubleComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(Z2Z_1D, InversePowerOfTwo) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

    auto h_in = random_double_complex(N);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
        ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_INVERSE);

        std::vector<flagfftDoubleComplex> h_out(N);
        std::vector<flagfftDoubleComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(Z2Z_1D, RoundtripForwardInverse) {
    constexpr int N = 256;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

    auto h_in = random_double_complex(N);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_mid = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_mid, FLAGFFT_FORWARD);
    ExecZ2Z(plan, d_mid, d_out, FLAGFFT_INVERSE);

    std::vector<flagfftDoubleComplex> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleComplex));

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

TEST(Z2Z_1D, NonPowerOfTwo) {
    constexpr int N = 243;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

    auto h_in = random_double_complex(N);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
        ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftDoubleComplex> h_out(N);
        std::vector<flagfftDoubleComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(Z2Z_1D, Batch) {
    constexpr int N = 128;
    constexpr int B = 3;
    int total = N * B;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, B);

    auto h_in = random_double_complex(total);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(total * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(total * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(total * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, total * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_Z2Z, B);
        ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftDoubleComplex> h_out(total);
        std::vector<flagfftDoubleComplex> h_ref_out(total);
        copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), total * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(Z2Z_2D, ForwardSmall) {
    constexpr int NX = 32;
    constexpr int NY = 16;
    constexpr int N = NX * NY;
    flagfftHandle plan = nullptr;
    Plan2d(&plan, NX, NY, FLAGFFT_Z2Z);

    auto h_in = random_double_complex(N);
    auto* d_in = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(N * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleComplex));

    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_2d(ref, NX, NY, FLAGFFT_Z2Z);
        ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);

        std::vector<flagfftDoubleComplex> h_out(N);
        std::vector<flagfftDoubleComplex> h_ref_out(N);
        copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}
