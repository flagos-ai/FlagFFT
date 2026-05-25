#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;  // double precision

static bool has_ref() { return backend_name() != "null"; }

static int d2z_out_n(int nx) { return nx / 2 + 1; }

TEST(D2Z_Z2D, Roundtrip1D) {
    constexpr int N = 256;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_D2Z, 1);
    Plan1d(&plan_inv, N, FLAGFFT_Z2D, 1);

    auto h_in = random_double_real(N);
    auto* d_in =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    auto* d_mid = static_cast<flagfftDoubleComplex*>(
        allocate_device(C_OUT * sizeof(flagfftDoubleComplex)));
    auto* d_out =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleReal));

    ExecD2Z(plan_fwd, d_in, d_mid);
    ExecZ2D(plan_inv, d_mid, d_out);

    std::vector<flagfftDoubleReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleReal));

    for (int i = 0; i < N; ++i) {
        double expected = h_in[i] * N;
        EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}

TEST(D2Z, ForwardVsReference) {
    constexpr int N = 256;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_D2Z, 1);

    auto h_in = random_double_real(N);
    auto* d_in =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    auto* d_out = static_cast<flagfftDoubleComplex*>(
        allocate_device(C_OUT * sizeof(flagfftDoubleComplex)));
    auto* d_ref = static_cast<flagfftDoubleComplex*>(
        allocate_device(C_OUT * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleReal));

    ExecD2Z(plan, d_in, d_out);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_D2Z, 1);
        ref_exec_d2z(ref, d_in, d_ref);

        std::vector<flagfftDoubleComplex> h_out(C_OUT);
        std::vector<flagfftDoubleComplex> h_ref_out(C_OUT);
        copy_device_to_host(d_out, h_out.data(), C_OUT * sizeof(flagfftDoubleComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), C_OUT * sizeof(flagfftDoubleComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), C_OUT);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(D2Z_Z2D, Roundtrip1D_NonPowerOfTwo) {
    constexpr int N = 243;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_D2Z, 1);
    Plan1d(&plan_inv, N, FLAGFFT_Z2D, 1);

    auto h_in = random_double_real(N);
    auto* d_in =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    auto* d_mid = static_cast<flagfftDoubleComplex*>(
        allocate_device(C_OUT * sizeof(flagfftDoubleComplex)));
    auto* d_out =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleReal));

    ExecD2Z(plan_fwd, d_in, d_mid);
    ExecZ2D(plan_inv, d_mid, d_out);

    std::vector<flagfftDoubleReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleReal));

    for (int i = 0; i < N; ++i) {
        double expected = h_in[i] * N;
        EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}

TEST(D2Z_Z2D, Roundtrip2D) {
    constexpr int NX = 32;
    constexpr int NY = 16;
    constexpr int N = NX * NY;
    constexpr int C_OUT = NX * (NY / 2 + 1);
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan2d(&plan_fwd, NX, NY, FLAGFFT_D2Z);
    Plan2d(&plan_inv, NX, NY, FLAGFFT_Z2D);

    auto h_in = random_double_real(N);
    auto* d_in =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    auto* d_mid = static_cast<flagfftDoubleComplex*>(
        allocate_device(C_OUT * sizeof(flagfftDoubleComplex)));
    auto* d_out =
        static_cast<flagfftDoubleReal*>(allocate_device(N * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftDoubleReal));

    ExecD2Z(plan_fwd, d_in, d_mid);
    ExecZ2D(plan_inv, d_mid, d_out);

    std::vector<flagfftDoubleReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftDoubleReal));

    for (int i = 0; i < N; ++i) {
        double expected = h_in[i] * N;
        EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}
