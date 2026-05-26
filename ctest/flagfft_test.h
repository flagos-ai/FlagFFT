#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "flagfft.h"

// =========================================================================
// FlagFFT plan convenience wrappers (assert success, otherwise GTEST_FAIL)
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
// FlagFFT exec convenience wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void ExecC2C(flagfftHandle plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  flagfftResult r = flagfftExecC2C(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecC2C failed with code " << r;
}

inline void ExecZ2Z(flagfftHandle plan,
                    flagfftDoubleComplex* idata,
                    flagfftDoubleComplex* odata,
                    int direction) {
  flagfftResult r = flagfftExecZ2Z(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecZ2Z failed with code " << r;
}

inline void ExecR2C(flagfftHandle plan, flagfftReal* idata, flagfftComplex* odata) {
  flagfftResult r = flagfftExecR2C(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecR2C failed with code " << r;
}

inline void ExecD2Z(flagfftHandle plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  flagfftResult r = flagfftExecD2Z(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecD2Z failed with code " << r;
}

inline void ExecC2R(flagfftHandle plan, flagfftComplex* idata, flagfftReal* odata) {
  flagfftResult r = flagfftExecC2R(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecC2R failed with code " << r;
}

inline void ExecZ2D(flagfftHandle plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  flagfftResult r = flagfftExecZ2D(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecZ2D failed with code " << r;
}
