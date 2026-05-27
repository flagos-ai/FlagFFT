#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cstdint>

#include "flagfft.h"

namespace flagfft_test::adaptor {

// =========================================================================
// RefHandle — RAII wrapper for the platform-specific reference FFT plan
// =========================================================================

class RefHandle {
 public:
  RefHandle();
  ~RefHandle();
  RefHandle(RefHandle&&) noexcept;
  RefHandle& operator=(RefHandle&&) noexcept;
  RefHandle(const RefHandle&) = delete;

  uintptr_t get() const;
  uintptr_t* ptr();

 private:
  uintptr_t impl_;  // cufftHandle / rocfft_plan / 0
};

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize();
std::string backend_name();

// =========================================================================
// Device memory
// =========================================================================

void* allocate_device(std::size_t bytes);
void free_device(void* ptr);
void copy_host_to_device(const void* src, void* dst, std::size_t bytes);
void copy_device_to_host(const void* src, void* dst, std::size_t bytes);

// =========================================================================
// Convenience Plan wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void Plan1d(flagfftHandle* plan, int nx, flagfftType type, int batch = 1) {
  flagfftResult r = flagfftPlan1d(plan, nx, type, batch);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan1d(nx=" << nx << ", type=" << type << ", batch=" << batch << ") failed with code "
           << r;
  }
}

inline void Plan2d(flagfftHandle* plan, int nx, int ny, flagfftType type) {
  flagfftResult r = flagfftPlan2d(plan, nx, ny, type);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan2d(nx=" << nx << ", ny=" << ny << ", type=" << type << ") failed with code " << r;
  }
}

inline void Plan3d(flagfftHandle* plan, int nx, int ny, int nz, flagfftType type) {
  flagfftResult r = flagfftPlan3d(plan, nx, ny, nz, type);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan3d(nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", type=" << type
           << ") failed with code " << r;
  }
}

// =========================================================================
// Convenience Exec wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void ExecC2C(flagfftHandle plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  flagfftResult r = flagfftExecC2C(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecC2C failed with code " << r;
  }
}

inline void ExecZ2Z(flagfftHandle plan,
                    flagfftDoubleComplex* idata,
                    flagfftDoubleComplex* odata,
                    int direction) {
  flagfftResult r = flagfftExecZ2Z(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecZ2Z failed with code " << r;
  }
}

inline void ExecR2C(flagfftHandle plan, flagfftReal* idata, flagfftComplex* odata) {
  flagfftResult r = flagfftExecR2C(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecR2C failed with code " << r;
  }
}

inline void ExecD2Z(flagfftHandle plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  flagfftResult r = flagfftExecD2Z(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecD2Z failed with code " << r;
  }
}

inline void ExecC2R(flagfftHandle plan, flagfftComplex* idata, flagfftReal* odata) {
  flagfftResult r = flagfftExecC2R(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecC2R failed with code " << r;
  }
}

inline void ExecZ2D(flagfftHandle plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  flagfftResult r = flagfftExecZ2D(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecZ2D failed with code " << r;
  }
}

// =========================================================================
// Reference FFT interface (implemented per backend)
// =========================================================================

void ref_plan_1d(RefHandle& plan, int nx, flagfftType type, int batch);
void ref_plan_2d(RefHandle& plan, int nx, int ny, flagfftType type);
void ref_plan_3d(RefHandle& plan, int nx, int ny, int nz, flagfftType type);
void ref_exec_c2c(RefHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction);
void ref_exec_z2z(RefHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleComplex* odata, int direction);
void ref_exec_r2c(RefHandle& plan, flagfftReal* idata, flagfftComplex* odata);
void ref_exec_d2z(RefHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata);
void ref_exec_c2r(RefHandle& plan, flagfftComplex* idata, flagfftReal* odata);
void ref_exec_z2d(RefHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata);

// =========================================================================
// Accuracy comparison helpers
// =========================================================================

enum class TransformClass { Complex, RealForward, RealInverse };

inline TransformClass transform_class(flagfftType type) {
  switch (type) {
    case FLAGFFT_C2C:
    case FLAGFFT_Z2Z:
      return TransformClass::Complex;
    case FLAGFFT_R2C:
    case FLAGFFT_D2Z:
      return TransformClass::RealForward;
    case FLAGFFT_C2R:
    case FLAGFFT_Z2D:
      return TransformClass::RealInverse;
  }
  return TransformClass::Complex;
}

inline const char* transform_class_name(TransformClass kind) {
  switch (kind) {
    case TransformClass::Complex:
      return "complex";
    case TransformClass::RealForward:
      return "real_forward";
    case TransformClass::RealInverse:
      return "real_inverse";
  }
  return "unknown";
}

inline const char* type_name(flagfftType type) {
  switch (type) {
    case FLAGFFT_C2C:
      return "C2C";
    case FLAGFFT_Z2Z:
      return "Z2Z";
    case FLAGFFT_R2C:
      return "R2C";
    case FLAGFFT_D2Z:
      return "D2Z";
    case FLAGFFT_C2R:
      return "C2R";
    case FLAGFFT_Z2D:
      return "Z2D";
  }
  return "unknown";
}

inline bool is_double_precision(flagfftType type) {
  return type == FLAGFFT_Z2Z || type == FLAGFFT_D2Z || type == FLAGFFT_Z2D;
}

inline long double sample_abs(flagfftReal value) {
  return std::abs(static_cast<long double>(value));
}

inline long double sample_abs(flagfftDoubleReal value) {
  return std::abs(static_cast<long double>(value));
}

inline long double sample_abs(const flagfftComplex& value) {
  return std::hypot(static_cast<long double>(value.x), static_cast<long double>(value.y));
}

inline long double sample_abs(const flagfftDoubleComplex& value) {
  return std::hypot(static_cast<long double>(value.x), static_cast<long double>(value.y));
}

inline long double sample_diff_abs(flagfftReal value, flagfftReal reference) {
  return std::abs(static_cast<long double>(value) - static_cast<long double>(reference));
}

inline long double sample_diff_abs(flagfftDoubleReal value, flagfftDoubleReal reference) {
  return std::abs(static_cast<long double>(value) - static_cast<long double>(reference));
}

inline long double sample_diff_abs(const flagfftComplex& value, const flagfftComplex& reference) {
  return std::hypot(static_cast<long double>(value.x) - static_cast<long double>(reference.x),
                    static_cast<long double>(value.y) - static_cast<long double>(reference.y));
}

inline long double sample_diff_abs(const flagfftDoubleComplex& value, const flagfftDoubleComplex& reference) {
  return std::hypot(static_cast<long double>(value.x) - static_cast<long double>(reference.x),
                    static_cast<long double>(value.y) - static_cast<long double>(reference.y));
}

struct ErrorStats {
  double rel_l2 = 0.0;
  double rel_linf = 0.0;
  double max_abs = 0.0;
  double mixed_pointwise = 0.0;
  int worst_l2_batch = 0;
  int worst_linf_batch = 0;
  bool finite = true;
};

template <typename T>
inline ErrorStats error_stats(const T* value, const T* reference, int elements_per_batch, int batch) {
  ErrorStats result;
  for (int b = 0; b < batch; ++b) {
    long double err_sq = 0.0L;
    long double ref_sq = 0.0L;
    long double err_max = 0.0L;
    long double ref_max = 0.0L;
    long double mixed_max = 0.0L;
    for (int i = 0; i < elements_per_batch; ++i) {
      int index = b * elements_per_batch + i;
      long double diff = sample_diff_abs(value[index], reference[index]);
      long double ref = sample_abs(reference[index]);
      if (!std::isfinite(diff) || !std::isfinite(ref)) {
        result.finite = false;
      }
      err_sq += diff * diff;
      ref_sq += ref * ref;
      err_max = std::max(err_max, diff);
      ref_max = std::max(ref_max, ref);
      mixed_max = std::max(mixed_max, diff / std::max(ref, 1.0L));
    }
    long double rel_l2 = ref_sq == 0.0L
                             ? (err_sq == 0.0L ? 0.0L : std::numeric_limits<long double>::infinity())
                             : std::sqrt(err_sq / ref_sq);
    long double rel_linf = ref_max == 0.0L
                               ? (err_max == 0.0L ? 0.0L : std::numeric_limits<long double>::infinity())
                               : err_max / ref_max;
    if (rel_l2 > result.rel_l2) {
      result.rel_l2 = static_cast<double>(rel_l2);
      result.worst_l2_batch = b;
    }
    if (rel_linf > result.rel_linf) {
      result.rel_linf = static_cast<double>(rel_linf);
      result.worst_linf_batch = b;
    }
    result.max_abs = std::max(result.max_abs, static_cast<double>(err_max));
    result.mixed_pointwise = std::max(result.mixed_pointwise, static_cast<double>(mixed_max));
  }
  return result;
}

inline int ceil_log2_covering(std::uint64_t value) {
  int power = 0;
  std::uint64_t covered = 1;
  while (covered < value) {
    covered <<= 1;
    ++power;
  }
  return power;
}

inline double work_factor(int n) {
  if (n <= 64) {
    return static_cast<double>(n);
  }
  // Worst-case length-only envelope: a Bluestein path performs three child
  // FFTs at a convolution length bounded by ceil_pow2(2*n-1), plus three
  // pointwise chirp/product stages.
  return 3.0 * ceil_log2_covering(static_cast<std::uint64_t>(2) * n - 1) + 3.0;
}

struct AccuracyLimits {
  double rel_l2;
  double rel_linf;
};

struct NormalizedAccuracyConstants {
  double rel_l2;
  double rel_linf;
};

inline NormalizedAccuracyConstants accuracy_constants(TransformClass kind) {
  // Frozen from the same-precision cuFFT characterization matrix on
  // A100-SXM4-40GB / CUDA 13.2. Each limit is K=2 times the observed
  // normalized maximum; see the test design document for the measurements.
  switch (kind) {
    case TransformClass::Complex:
      return {1.2419386546059821, 1.9343969087678796};
    case TransformClass::RealForward:
      return {1.234681000407627, 1.8260558195934091};
    case TransformClass::RealInverse:
      return {0.97722970418819066, 1.372182697342486};
  }
  return {0.0, 0.0};
}

inline double unit_roundoff(flagfftType type) {
  if (is_double_precision(type)) {
    return std::numeric_limits<double>::epsilon() / 2.0;
  }
  return std::numeric_limits<float>::epsilon() / 2.0;
}

inline AccuracyLimits accuracy_limits(flagfftType type, int n) {
  NormalizedAccuracyConstants constants = accuracy_constants(transform_class(type));
  double scale = unit_roundoff(type) * work_factor(n);
  return {constants.rel_l2 * scale, constants.rel_linf * scale};
}

inline AccuracyLimits add_limits(AccuracyLimits lhs, AccuracyLimits rhs) {
  return {lhs.rel_l2 + rhs.rel_l2, lhs.rel_linf + rhs.rel_linf};
}

inline std::string accuracy_message(const ErrorStats& stats,
                                    AccuracyLimits limits,
                                    flagfftType type,
                                    int n,
                                    int batch,
                                    const char* input_family = "random") {
  const double scale = unit_roundoff(type) * work_factor(n);
  std::ostringstream message;
  message << std::setprecision(17) << "type=" << type_name(type) << " N=" << n << " batch=" << batch
          << " input=" << input_family << " rel_l2=" << stats.rel_l2 << " limit_l2=" << limits.rel_l2
          << " normalized_l2=" << stats.rel_l2 / scale << " worst_l2_batch=" << stats.worst_l2_batch
          << " rel_linf=" << stats.rel_linf << " limit_linf=" << limits.rel_linf
          << " normalized_linf=" << stats.rel_linf / scale << " worst_linf_batch=" << stats.worst_linf_batch
          << " max_abs=" << stats.max_abs << " mixed_pointwise=" << stats.mixed_pointwise;
  return message.str();
}

inline void maybe_report_accuracy(
    const ErrorStats& stats, flagfftType type, int n, int batch, const char* input_family = "random") {
  if (std::getenv("FLAGFFT_TEST_REPORT_ACCURACY") == nullptr) {
    return;
  }
  AccuracyLimits limits = accuracy_limits(type, n);
  std::cout << "ACCURACY class=" << transform_class_name(transform_class(type)) << " "
            << accuracy_message(stats, limits, type, n, batch, input_family) << '\n';
}

inline void expect_reference_accuracy(
    const ErrorStats& stats, flagfftType type, int n, int batch, const char* input_family = "random") {
  AccuracyLimits limits = accuracy_limits(type, n);
  maybe_report_accuracy(stats, type, n, batch, input_family);
  EXPECT_TRUE(stats.finite && stats.rel_l2 <= limits.rel_l2 && stats.rel_linf <= limits.rel_linf)
      << accuracy_message(stats, limits, type, n, batch, input_family);
}

inline void expect_roundtrip_accuracy(
    const ErrorStats& stats, flagfftType first_type, flagfftType second_type, int n, int batch) {
  AccuracyLimits limits = add_limits(accuracy_limits(first_type, n), accuracy_limits(second_type, n));
  EXPECT_TRUE(stats.finite && stats.rel_l2 <= limits.rel_l2 && stats.rel_linf <= limits.rel_linf)
      << accuracy_message(stats, limits, second_type, n, batch);
}

// Retained as a failure diagnostic for near-zero reference elements. It is not
// used as the numerical acceptance criterion.
inline double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n) {
  return error_stats(a, b, n, 1).mixed_pointwise;
}

inline double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n) {
  return error_stats(a, b, n, 1).mixed_pointwise;
}

inline double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n) {
  return error_stats(a, b, n, 1).mixed_pointwise;
}

inline double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n) {
  return error_stats(a, b, n, 1).mixed_pointwise;
}

// =========================================================================
// Test data generators
// =========================================================================

class StableRng {
 public:
  explicit StableRng(std::uint64_t seed) : state_(seed) {
  }

  double signed_unit() {
    std::uint64_t bits = next() >> 11;
    constexpr double kScale = 1.0 / 9007199254740992.0;
    return (static_cast<double>(bits) * kScale) * 2.0 - 1.0;
  }

 private:
  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  std::uint64_t state_;
};

inline std::uint64_t accuracy_seed(flagfftType type, int n, int batch, std::uint64_t variant = 0) {
  return 0x4654464654455354ULL ^ (static_cast<std::uint64_t>(type) << 48) ^
         (static_cast<std::uint64_t>(n) << 16) ^ static_cast<std::uint64_t>(batch) ^
         (variant * 0x9e3779b97f4a7c15ULL);
}

inline std::vector<flagfftComplex> random_complex(int n, std::uint64_t seed = 1) {
  StableRng rng(seed);
  std::vector<flagfftComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = static_cast<float>(rng.signed_unit());
    v[i].y = static_cast<float>(rng.signed_unit());
  }
  return v;
}

inline std::vector<flagfftDoubleComplex> random_double_complex(int n, std::uint64_t seed = 1) {
  StableRng rng(seed);
  std::vector<flagfftDoubleComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = rng.signed_unit();
    v[i].y = rng.signed_unit();
  }
  return v;
}

inline std::vector<flagfftReal> random_real(int n, std::uint64_t seed = 1) {
  StableRng rng(seed);
  std::vector<flagfftReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<float>(rng.signed_unit());
  }
  return v;
}

inline std::vector<flagfftDoubleReal> random_double_real(int n, std::uint64_t seed = 1) {
  StableRng rng(seed);
  std::vector<flagfftDoubleReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = rng.signed_unit();
  }
  return v;
}

inline constexpr double kAccuracyInputScales[] = {0x1p-20, 1.0, 0x1p20};

inline const char* input_scale_name(double scale) {
  if (scale == 0x1p-20) {
    return "random_scale_2^-20";
  }
  if (scale == 0x1p20) {
    return "random_scale_2^20";
  }
  return "random_scale_1";
}

inline void scale_input(std::vector<flagfftComplex>& values, double scale) {
  for (auto& value : values) {
    value.x = static_cast<float>(value.x * scale);
    value.y = static_cast<float>(value.y * scale);
  }
}

inline void scale_input(std::vector<flagfftDoubleComplex>& values, double scale) {
  for (auto& value : values) {
    value.x *= scale;
    value.y *= scale;
  }
}

inline void scale_input(std::vector<flagfftReal>& values, double scale) {
  for (auto& value : values) {
    value = static_cast<float>(value * scale);
  }
}

inline void scale_input(std::vector<flagfftDoubleReal>& values, double scale) {
  for (auto& value : values) {
    value *= scale;
  }
}

// =========================================================================
// 1D Test parameterization
// =========================================================================

struct Test1DParam {
  int N;
  int batch;
};

// Registry — add/remove sizes and batch values here
constexpr int k1DSizesSmall[] = {16, 23, 64, 81};
constexpr int k1DSizesMedium[] = {243, 256, 361, 512, 997};
constexpr int k1DSizesLarge[] = {2048, 4096, 8192, 16384};
constexpr int k1DBatchValues[] = {1, 4, 256};

inline std::vector<Test1DParam> Generate1DParams(const int* sizes, int numSizes) {
  std::vector<Test1DParam> params;
  for (int i = 0; i < numSizes; ++i)
    for (int b : k1DBatchValues) params.push_back({sizes[i], b});
  return params;
}

inline bool IsSmoke1DParam(const Test1DParam& param) {
  return param.batch == 1 && (param.N == 16 || param.N == 997 || param.N == 16384);
}

inline std::vector<Test1DParam> Generate1DParamsSmoke() {
  return {
      {   16, 1},
      {  997, 1},
      {16384, 1}
  };
}

inline std::vector<Test1DParam> Generate1DParamsExtended(const int* sizes, int numSizes) {
  std::vector<Test1DParam> params;
  for (const auto& param : Generate1DParams(sizes, numSizes)) {
    if (!IsSmoke1DParam(param)) {
      params.push_back(param);
    }
  }
  return params;
}

inline std::vector<Test1DParam> Generate1DParamsExtendedSmall() {
  return Generate1DParamsExtended(k1DSizesSmall, sizeof(k1DSizesSmall) / sizeof(k1DSizesSmall[0]));
}
inline std::vector<Test1DParam> Generate1DParamsExtendedMedium() {
  return Generate1DParamsExtended(k1DSizesMedium, sizeof(k1DSizesMedium) / sizeof(k1DSizesMedium[0]));
}
inline std::vector<Test1DParam> Generate1DParamsExtendedLarge() {
  return Generate1DParamsExtended(k1DSizesLarge, sizeof(k1DSizesLarge) / sizeof(k1DSizesLarge[0]));
}

}  // namespace flagfft_test::adaptor
