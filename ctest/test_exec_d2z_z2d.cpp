#include "flagfft_test.h"

using namespace flagfft::test_adaptor;

constexpr double kRelTol = 1e-10;  // double precision

static int d2z_out_n(int nx) {
  return nx / 2 + 1;
}

TEST(D2Z_Z2D, Roundtrip1D) {
  constexpr int N = 256;
  constexpr int C_OUT = N / 2 + 1;
  flagfftHandle plan_fwd = nullptr;
  flagfftHandle plan_inv = nullptr;
  Plan1d(&plan_fwd, N, FLAGFFT_D2Z, 1);
  Plan1d(&plan_inv, N, FLAGFFT_Z2D, 1);

  auto h_in = random_double_real(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleReal));

  ExecD2Z(plan_fwd,
          static_cast<flagfftDoubleReal*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_mid.data()));
  ExecZ2D(plan_inv,
          static_cast<flagfftDoubleComplex*>(d_mid.data()),
          static_cast<flagfftDoubleReal*>(d_out.data()));

  std::vector<flagfftDoubleReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleReal));

  for (int i = 0; i < N; ++i) {
    double expected = h_in[i] * N;
    EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
  }

  flagfftDestroy(plan_fwd);
  flagfftDestroy(plan_inv);
}

TEST(D2Z, ForwardVsReference) {
  constexpr int N = 256;
  constexpr int C_OUT = N / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_D2Z, 1);

  auto h_in = random_double_real(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory d_out(C_OUT * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(C_OUT * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleReal));

  ExecD2Z(plan,
          static_cast<flagfftDoubleReal*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()));

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_D2Z, 1);
  ref_exec_d2z(ref,
               static_cast<flagfftDoubleReal*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()));

  std::vector<flagfftDoubleComplex> h_out(C_OUT);
  std::vector<flagfftDoubleComplex> h_ref_out(C_OUT);
  d_out.copy_to_host(h_out.data(), C_OUT * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), C_OUT * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), C_OUT);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleReal));

  ExecD2Z(plan_fwd,
          static_cast<flagfftDoubleReal*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_mid.data()));
  ExecZ2D(plan_inv,
          static_cast<flagfftDoubleComplex*>(d_mid.data()),
          static_cast<flagfftDoubleReal*>(d_out.data()));

  std::vector<flagfftDoubleReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleReal));

  for (int i = 0; i < N; ++i) {
    double expected = h_in[i] * N;
    EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
  }

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory d_mid(C_OUT * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleReal));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleReal));

  ExecD2Z(plan_fwd,
          static_cast<flagfftDoubleReal*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_mid.data()));
  ExecZ2D(plan_inv,
          static_cast<flagfftDoubleComplex*>(d_mid.data()),
          static_cast<flagfftDoubleReal*>(d_out.data()));

  std::vector<flagfftDoubleReal> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleReal));

  for (int i = 0; i < N; ++i) {
    double expected = h_in[i] * N;
    EXPECT_NEAR(h_out[i], expected, N * kRelTol) << "Mismatch at index " << i;
  }

  flagfftDestroy(plan_fwd);
  flagfftDestroy(plan_inv);
}
