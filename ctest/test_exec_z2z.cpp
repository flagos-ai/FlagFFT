#include "flagfft_test.h"

using namespace flagfft::test_adaptor;

constexpr double kRelTol = 1e-10;  // double precision

TEST(Z2Z_1D, ForwardPowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

  auto h_in = random_double_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
  ref_exec_z2z(ref,
               static_cast<flagfftDoubleComplex*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(N);
  std::vector<flagfftDoubleComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(Z2Z_1D, InversePowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

  auto h_in = random_double_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_INVERSE);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
  ref_exec_z2z(ref,
               static_cast<flagfftDoubleComplex*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()),
               FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(N);
  std::vector<flagfftDoubleComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(Z2Z_1D, RoundtripForwardInverse) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

  auto h_in = random_double_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_mid(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_mid.data()),
          FLAGFFT_FORWARD);
  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_mid.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleComplex));

  for (int i = 0; i < N; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol) << "Mismatch at index " << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol) << "Mismatch at index " << i << " (imag)";
  }

  flagfftDestroy(plan);
}

TEST(Z2Z_1D, NonPowerOfTwo) {
  constexpr int N = 243;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_Z2Z, 1);

  auto h_in = random_double_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, 1);
  ref_exec_z2z(ref,
               static_cast<flagfftDoubleComplex*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(N);
  std::vector<flagfftDoubleComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(Z2Z_1D, Batch) {
  constexpr int N = 128;
  constexpr int B = 3;
  int total = N * B;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_Z2Z, B);

  auto h_in = random_double_complex(total);
  flagfft::adaptor::Memory d_in(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(total * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), total * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, B);
  ref_exec_z2z(ref,
               static_cast<flagfftDoubleComplex*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref_out(total);
  d_out.copy_to_host(h_out.data(), total * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), total * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(Z2Z_2D, ForwardSmall) {
  constexpr int NX = 32;
  constexpr int NY = 16;
  constexpr int N = NX * NY;
  flagfftHandle plan = nullptr;
  Plan2d(&plan, NX, NY, FLAGFFT_Z2Z);

  auto h_in = random_double_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftDoubleComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftDoubleComplex));

  ExecZ2Z(plan,
          static_cast<flagfftDoubleComplex*>(d_in.data()),
          static_cast<flagfftDoubleComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_2d(ref, NX, NY, FLAGFFT_Z2Z);
  ref_exec_z2z(ref,
               static_cast<flagfftDoubleComplex*>(d_in.data()),
               static_cast<flagfftDoubleComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(N);
  std::vector<flagfftDoubleComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftDoubleComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}
