#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flagfft.h"

namespace flagfft::test_adaptor {

// =========================================================================
// RefPlanHandle — RAII wrapper for reference FFT plan (cuFFT / rocFFT / ...)
// =========================================================================

class RefPlanHandle {
 public:
  RefPlanHandle();
  ~RefPlanHandle();
  RefPlanHandle(RefPlanHandle&&) noexcept;
  RefPlanHandle& operator=(RefPlanHandle&&) noexcept;
  RefPlanHandle(const RefPlanHandle&) = delete;
  RefPlanHandle& operator=(const RefPlanHandle&) = delete;

  std::uintptr_t get() const;
  void replace(std::uintptr_t new_handle);

 private:
  std::uintptr_t impl_;
};

// =========================================================================
// Plan creation
// =========================================================================

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch);
void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type);
void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type);
void ref_set_stream(RefPlanHandle& plan, flagfftStream_t stream);

// =========================================================================
// Plan execution
// =========================================================================

void ref_exec_c2c(RefPlanHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction);
void ref_exec_z2z(RefPlanHandle& plan,
                  flagfftDoubleComplex* idata,
                  flagfftDoubleComplex* odata,
                  int direction);
void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata);
void ref_exec_d2z(RefPlanHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata);
void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata);
void ref_exec_z2d(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata);

// =========================================================================
// Backend metadata
// =========================================================================

void initialize();
std::string backend_name();

// =========================================================================
// Data generation
// =========================================================================

std::vector<flagfftComplex> random_complex(int n);
std::vector<flagfftDoubleComplex> random_double_complex(int n);
std::vector<flagfftReal> random_real(int n);
std::vector<flagfftDoubleReal> random_double_real(int n);

// =========================================================================
// Correctness comparison
// =========================================================================

struct ErrorMetric {
  double max_abs = 0.0;
  double rms = 0.0;
};

double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n);
double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n);
double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n);
double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n);

ErrorMetric compute_error(const float* a, const float* b, std::size_t n);
ErrorMetric compute_error(const double* a, const double* b, std::size_t n);
ErrorMetric compute_error(const flagfftComplex* a, const flagfftComplex* b, std::size_t n);
ErrorMetric compute_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, std::size_t n);

}  // namespace flagfft::test_adaptor
