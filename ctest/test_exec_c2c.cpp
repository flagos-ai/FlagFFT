#include "flagfft_test.h"

using namespace flagfft::test_adaptor;

constexpr double kRelTol = 1e-4;

// =========================================================================
// 1D C2C
// =========================================================================

TEST(C2C_1D, ForwardPowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, InversePowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_INVERSE);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, RoundtripForwardInverse) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_mid(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_mid.data()),
          FLAGFFT_FORWARD);
  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_mid.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));

  for (int i = 0; i < N; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol) << "Mismatch at index " << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol) << "Mismatch at index " << i << " (imag)";
  }

  flagfftDestroy(plan);
}

TEST(C2C_1D, NonPowerOfTwo) {
  constexpr int N = 243;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, Batch) {
  constexpr int N = 128;
  constexpr int B = 4;
  int total = N * B;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, B);

  auto h_in = random_complex(total);
  flagfft::adaptor::Memory d_in(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(total * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), total * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, B);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref_out(total);
  d_out.copy_to_host(h_out.data(), total * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_2d(ref, NX, NY, FLAGFFT_C2C);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

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
  flagfft::adaptor::Memory d_in(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_out(N * sizeof(flagfftComplex));
  flagfft::adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan,
          static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()),
          FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_3d(ref, NX, NY, NZ, FLAGFFT_C2C);
  ref_exec_c2c(ref,
               static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()),
               FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}
