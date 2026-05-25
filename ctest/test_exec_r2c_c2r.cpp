#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;  // single precision

static bool has_ref() { return backend_name() != "null"; }

// R2C output size: N → N/2 + 1 complex elements
static int r2c_out_n(int nx) { return nx / 2 + 1; }

TEST(R2C_C2R, Roundtrip1D) {
    constexpr int N = 256;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_R2C, 1);
    Plan1d(&plan_inv, N, FLAGFFT_C2R, 1);

    auto h_in = random_real(N);
    auto* d_in = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    auto* d_mid = static_cast<flagfftComplex*>(allocate_device(C_OUT * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftReal));

    ExecR2C(plan_fwd, d_in, d_mid);
    ExecC2R(plan_inv, d_mid, d_out);

    std::vector<flagfftReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftReal));

    // Roundtrip: H_out = F^-1(F(H_in)) = N * H_in
    for (int i = 0; i < N; ++i) {
        double expected = static_cast<double>(h_in[i]) * N;
        EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol)
            << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}

TEST(R2C, ForwardVsReference) {
    constexpr int N = 256;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_R2C, 1);

    auto h_in = random_real(N);
    auto* d_in = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    auto* d_out = static_cast<flagfftComplex*>(
        allocate_device(C_OUT * sizeof(flagfftComplex)));
    auto* d_ref = static_cast<flagfftComplex*>(
        allocate_device(C_OUT * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftReal));

    ExecR2C(plan, d_in, d_out);

    if (has_ref()) {
        RefHandle ref;
        ref_plan_1d(ref, N, FLAGFFT_R2C, 1);
        ref_exec_r2c(ref, d_in, d_ref);

        std::vector<flagfftComplex> h_out(C_OUT);
        std::vector<flagfftComplex> h_ref_out(C_OUT);
        copy_device_to_host(d_out, h_out.data(), C_OUT * sizeof(flagfftComplex));
        copy_device_to_host(d_ref, h_ref_out.data(), C_OUT * sizeof(flagfftComplex));

        double max_err = max_relative_error(h_out.data(), h_ref_out.data(), C_OUT);
        EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;
    }

    free_device(d_in);
    free_device(d_out);
    free_device(d_ref);
    flagfftDestroy(plan);
}

TEST(R2C_C2R, Roundtrip1D_NonPowerOfTwo) {
    constexpr int N = 243;
    constexpr int C_OUT = N / 2 + 1;
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_R2C, 1);
    Plan1d(&plan_inv, N, FLAGFFT_C2R, 1);

    auto h_in = random_real(N);
    auto* d_in = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    auto* d_mid = static_cast<flagfftComplex*>(
        allocate_device(C_OUT * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftReal));

    ExecR2C(plan_fwd, d_in, d_mid);
    ExecC2R(plan_inv, d_mid, d_out);

    std::vector<flagfftReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftReal));

    for (int i = 0; i < N; ++i) {
        double expected = static_cast<double>(h_in[i]) * N;
        EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol)
            << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}

TEST(R2C_C2R, Roundtrip2D) {
    constexpr int NX = 32;
    constexpr int NY = 16;
    constexpr int N = NX * NY;
    constexpr int C_OUT = NX * (NY / 2 + 1);
    flagfftHandle plan_fwd = nullptr;
    flagfftHandle plan_inv = nullptr;
    Plan2d(&plan_fwd, NX, NY, FLAGFFT_R2C);
    Plan2d(&plan_inv, NX, NY, FLAGFFT_C2R);

    auto h_in = random_real(N);
    auto* d_in = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    auto* d_mid = static_cast<flagfftComplex*>(
        allocate_device(C_OUT * sizeof(flagfftComplex)));
    auto* d_out = static_cast<flagfftReal*>(allocate_device(N * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);

    copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftReal));

    ExecR2C(plan_fwd, d_in, d_mid);
    ExecC2R(plan_inv, d_mid, d_out);

    std::vector<flagfftReal> h_out(N);
    copy_device_to_host(d_out, h_out.data(), N * sizeof(flagfftReal));

    for (int i = 0; i < N; ++i) {
        double expected = static_cast<double>(h_in[i]) * N;
        EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol)
            << "Mismatch at index " << i;
    }

    free_device(d_in);
    free_device(d_mid);
    free_device(d_out);
    flagfftDestroy(plan_fwd);
    flagfftDestroy(plan_inv);
}
