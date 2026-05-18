#pragma once

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum flagfftResult_t {
    FLAGFFT_SUCCESS = 0,
    FLAGFFT_INVALID_PLAN = 1,
    FLAGFFT_ALLOC_FAILED = 2,
    FLAGFFT_INVALID_TYPE = 3,
    FLAGFFT_INVALID_VALUE = 4,
    FLAGFFT_INTERNAL_ERROR = 5,
    FLAGFFT_EXEC_FAILED = 6,
    FLAGFFT_SETUP_FAILED = 7,
    FLAGFFT_INVALID_SIZE = 8,
    FLAGFFT_UNALIGNED_DATA = 9,
    FLAGFFT_INCOMPLETE_PARAMETER_LIST = 10,
    FLAGFFT_INVALID_DEVICE = 11,
    FLAGFFT_PARSE_ERROR = 12,
    FLAGFFT_NO_WORKSPACE = 13,
    FLAGFFT_NOT_SUPPORTED = 14,
} flagfftResult;

typedef enum flagfftType_t {
    FLAGFFT_R2C = 0x2a,
    FLAGFFT_C2R = 0x2c,
    FLAGFFT_C2C = 0x29,
    FLAGFFT_D2Z = 0x6a,
    FLAGFFT_Z2D = 0x6c,
    FLAGFFT_Z2Z = 0x69,
} flagfftType;

#define FLAGFFT_FORWARD (-1)
#define FLAGFFT_INVERSE 1

typedef struct flagfftComplex_t {
    float x;
    float y;
} flagfftComplex;

typedef struct flagfftDoubleComplex_t {
    double x;
    double y;
} flagfftDoubleComplex;

typedef float flagfftReal;
typedef double flagfftDoubleReal;

typedef struct flagfftPlan_t *flagfftHandle;

flagfftResult flagfftPlan1d(flagfftHandle *plan, int nx, flagfftType type, int batch);
flagfftResult flagfftPlan2d(flagfftHandle *plan, int nx, int ny, flagfftType type);
flagfftResult flagfftPlan3d(flagfftHandle *plan, int nx, int ny, int nz, flagfftType type);
flagfftResult flagfftPlanMany(flagfftHandle *plan,
                              int rank,
                              int *n,
                              int *inembed,
                              int istride,
                              int idist,
                              int *onembed,
                              int ostride,
                              int odist,
                              flagfftType type,
                              int batch);

flagfftResult flagfftExecC2C(flagfftHandle plan,
                             flagfftComplex *idata,
                             flagfftComplex *odata,
                             int direction);
flagfftResult flagfftExecZ2Z(flagfftHandle plan,
                             flagfftDoubleComplex *idata,
                             flagfftDoubleComplex *odata,
                             int direction);
flagfftResult flagfftExecR2C(flagfftHandle plan, flagfftReal *idata, flagfftComplex *odata);
flagfftResult flagfftExecD2Z(flagfftHandle plan,
                             flagfftDoubleReal *idata,
                             flagfftDoubleComplex *odata);
flagfftResult flagfftExecC2R(flagfftHandle plan, flagfftComplex *idata, flagfftReal *odata);
flagfftResult flagfftExecZ2D(flagfftHandle plan,
                             flagfftDoubleComplex *idata,
                             flagfftDoubleReal *odata);

flagfftResult flagfftSetStream(flagfftHandle plan, cudaStream_t stream);
flagfftResult flagfftDestroy(flagfftHandle plan);

#ifdef __cplusplus
}
#endif
