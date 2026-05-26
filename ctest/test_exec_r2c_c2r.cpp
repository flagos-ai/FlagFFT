#include "flagfft_test.h"

using namespace flagfft::test_adaptor;

constexpr double kRelTol = 1e-4;  // single precision

// R2C output size: N -> N/2 + 1 complex elements
static int r2c_out_n(int nx) {
  return nx / 2 + 1;
}

TEST(R2C_C2R, Roundtrip1D) {
  constexpr int N = 256;
  constexpr int C_OUT = N / 2 + 1;
  flagfftHandle plan_fwd = nullptr;
  flagfftHandle plan_inv = nullptr;
  Plan1d(&plan_fwd, N, FLAGFFT_R2C, 1);
  Plan1d(&plan_inv, N, FLAGFFT_C2R, 1);

  auto h_in = random_real(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftReal));

  ExecR2C(plan_fwd, static_cast<flagfftReal*>(d_in.data()), static_cast<flagfftComplex*>(d_mid.data()));
  ExecC2R(plan_inv, static_cast<flagfftComplex*>(d_mid.data()), static_cast<flagfftReal*>(d_out.data()));

  std::vector<flagfftReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftReal));

  // Roundtrip: H_out = F^-1(F(H_in)) = N * H_in
  for (int i = 0; i < N; ++i) {
    double expected = static_cast<double>(h_in[i]) * N;
    EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol) << "Mismatch at index " << i;
  }

  flagfftDestroy(plan_fwd);
  flagfftDestroy(plan_inv);
}

TEST(R2C, ForwardVsReference) {
  constexpr int N = 256;
  constexpr int C_OUT = N / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_R2C, 1);

  auto h_in = random_real(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftReal));
  flagfft::adaptor::Memory d_out(C_OUT * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(C_OUT * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftReal));

  ExecR2C(plan, static_cast<flagfftReal*>(d_in.data()), static_cast<flagfftComplex*>(d_out.data()));

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_R2C, 1);
  ref_exec_r2c(ref, static_cast<flagfftReal*>(d_in.data()), static_cast<flagfftComplex*>(d_ref.data()));

  std::vector<flagfftComplex> h_out(C_OUT);
  std::vector<flagfftComplex> h_ref_out(C_OUT);
  d_out.copy_to_host(h_out.data(), C_OUT * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), C_OUT * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), C_OUT);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftReal));

  ExecR2C(plan_fwd, static_cast<flagfftReal*>(d_in.data()), static_cast<flagfftComplex*>(d_mid.data()));
  ExecC2R(plan_inv, static_cast<flagfftComplex*>(d_mid.data()), static_cast<flagfftReal*>(d_out.data()));

  std::vector<flagfftReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftReal));

  for (int i = 0; i < N; ++i) {
    double expected = static_cast<double>(h_in[i]) * N;
    EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol) << "Mismatch at index " << i;
  }

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftReal));

  ExecR2C(plan_fwd, static_cast<flagfftReal*>(d_in.data()), static_cast<flagfftComplex*>(d_mid.data()));
  ExecC2R(plan_inv, static_cast<flagfftComplex*>(d_mid.data()), static_cast<flagfftReal*>(d_out.data()));

  std::vector<flagfftReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftReal));

  for (int i = 0; i < N; ++i) {
    double expected = static_cast<double>(h_in[i]) * N;
    EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol) << "Mismatch at index " << i;
  }

  flagfftDestroy(plan_fwd);
  flagfftDestroy(plan_inv);
}
