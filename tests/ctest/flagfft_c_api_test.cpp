#include <cuda_runtime_api.h>
#include <cufft.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <vector>

#include "flagfft/flagfft.h"

namespace {

void require_cuda() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        GTEST_SKIP() << "CUDA device is required";
    }
}

std::vector<flagfftComplex> sample_input(int n, int batch) {
    std::mt19937 rng(1234 + n * 17 + batch);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<flagfftComplex> data(static_cast<std::size_t>(n * batch));
    for (auto &value : data) {
        value.x = dist(rng);
        value.y = dist(rng);
    }
    return data;
}

void assert_close(const std::vector<flagfftComplex> &actual,
                  const std::vector<flagfftComplex> &expected,
                  float atol,
                  float rtol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        float dr = std::abs(actual[i].x - expected[i].x);
        float di = std::abs(actual[i].y - expected[i].y);
        float er = std::abs(expected[i].x);
        float ei = std::abs(expected[i].y);
        ASSERT_LE(dr, atol + rtol * er) << "real mismatch at " << i;
        ASSERT_LE(di, atol + rtol * ei) << "imag mismatch at " << i;
    }
}

std::vector<flagfftComplex> run_cufft(const std::vector<flagfftComplex> &input,
                                      int n,
                                      int batch,
                                      int direction) {
    flagfftComplex *d_in = nullptr;
    flagfftComplex *d_out = nullptr;
    const std::size_t bytes = input.size() * sizeof(flagfftComplex);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    cufftHandle plan = 0;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        ADD_FAILURE() << "cufftPlan1d failed";
        return {};
    }
    if (cufftExecC2C(plan,
                     reinterpret_cast<cufftComplex *>(d_in),
                     reinterpret_cast<cufftComplex *>(d_out),
                     direction) != CUFFT_SUCCESS) {
        ADD_FAILURE() << "cufftExecC2C failed";
        cufftDestroy(plan);
        return {};
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "cudaDeviceSynchronize failed";
        cufftDestroy(plan);
        return {};
    }

    std::vector<flagfftComplex> output(input.size());
    EXPECT_EQ(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    cufftDestroy(plan);
    cudaFree(d_out);
    cudaFree(d_in);
    return output;
}

std::vector<flagfftComplex> run_flagfft(const std::vector<flagfftComplex> &input,
                                        int n,
                                        int batch,
                                        int direction,
                                        bool custom_stream) {
    flagfftComplex *d_in = nullptr;
    flagfftComplex *d_out = nullptr;
    const std::size_t bytes = input.size() * sizeof(flagfftComplex);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    flagfftHandle plan = nullptr;
    if (flagfftPlan1d(&plan, n, FLAGFFT_C2C, batch) != FLAGFFT_SUCCESS) {
        ADD_FAILURE() << "flagfftPlan1d failed";
        return {};
    }

    cudaStream_t stream = nullptr;
    if (custom_stream) {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamCreate failed";
            flagfftDestroy(plan);
            return {};
        }
        if (flagfftSetStream(plan, stream) != FLAGFFT_SUCCESS) {
            ADD_FAILURE() << "flagfftSetStream failed";
            flagfftDestroy(plan);
            cudaStreamDestroy(stream);
            return {};
        }
    }

    if (flagfftExecC2C(plan, d_in, d_out, direction) != FLAGFFT_SUCCESS) {
        ADD_FAILURE() << "flagfftExecC2C failed";
        flagfftDestroy(plan);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        return {};
    }
    if (custom_stream) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize failed";
        }
        cudaStreamDestroy(stream);
    } else {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed";
        }
    }

    std::vector<flagfftComplex> output(input.size());
    EXPECT_EQ(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
    cudaFree(d_out);
    cudaFree(d_in);
    return output;
}

void compare_case(int n, int batch, int direction, bool custom_stream = false) {
    require_cuda();
    auto input = sample_input(n, batch);
    auto expected = run_cufft(input, n, batch, direction);
    auto actual = run_flagfft(input, n, batch, direction, custom_stream);
    float scale = direction == FLAGFFT_INVERSE ? static_cast<float>(n) : 1.0f;
    // fp32 nested four-step accumulates ~sqrt(N) * eps_fp32 error; bump the
    // base tolerance for very large transforms (n >= 2^20) where the cost
    // model picks nested fused four-step over a single layer.
    float base_tol = n >= (1 << 20) ? 3.0e-3f : 1.5e-3f;
    assert_close(actual, expected, base_tol * scale, base_tol);
}

std::vector<flagfftDoubleComplex> sample_input_z(int n, int batch) {
    std::mt19937 rng(9876 + n * 31 + batch);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<flagfftDoubleComplex> data(static_cast<std::size_t>(n * batch));
    for (auto &value : data) {
        value.x = dist(rng);
        value.y = dist(rng);
    }
    return data;
}

void assert_close_z(const std::vector<flagfftDoubleComplex> &actual,
                    const std::vector<flagfftDoubleComplex> &expected,
                    double atol,
                    double rtol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double dr = std::abs(actual[i].x - expected[i].x);
        double di = std::abs(actual[i].y - expected[i].y);
        double er = std::abs(expected[i].x);
        double ei = std::abs(expected[i].y);
        ASSERT_LE(dr, atol + rtol * er) << "real mismatch at " << i;
        ASSERT_LE(di, atol + rtol * ei) << "imag mismatch at " << i;
    }
}

std::vector<flagfftDoubleComplex> run_cufft_z(const std::vector<flagfftDoubleComplex> &input,
                                              int n,
                                              int batch,
                                              int direction) {
    flagfftDoubleComplex *d_in = nullptr;
    flagfftDoubleComplex *d_out = nullptr;
    const std::size_t bytes = input.size() * sizeof(flagfftDoubleComplex);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    cufftHandle plan = 0;
    if (cufftPlan1d(&plan, n, CUFFT_Z2Z, batch) != CUFFT_SUCCESS) {
        ADD_FAILURE() << "cufftPlan1d Z2Z failed";
        return {};
    }
    if (cufftExecZ2Z(plan,
                     reinterpret_cast<cufftDoubleComplex *>(d_in),
                     reinterpret_cast<cufftDoubleComplex *>(d_out),
                     direction) != CUFFT_SUCCESS) {
        ADD_FAILURE() << "cufftExecZ2Z failed";
        cufftDestroy(plan);
        return {};
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ADD_FAILURE() << "cudaDeviceSynchronize failed";
        cufftDestroy(plan);
        return {};
    }

    std::vector<flagfftDoubleComplex> output(input.size());
    EXPECT_EQ(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    cufftDestroy(plan);
    cudaFree(d_out);
    cudaFree(d_in);
    return output;
}

std::vector<flagfftDoubleComplex> run_flagfft_z(const std::vector<flagfftDoubleComplex> &input,
                                                int n,
                                                int batch,
                                                int direction,
                                                bool custom_stream) {
    flagfftDoubleComplex *d_in = nullptr;
    flagfftDoubleComplex *d_out = nullptr;
    const std::size_t bytes = input.size() * sizeof(flagfftDoubleComplex);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    flagfftHandle plan = nullptr;
    if (flagfftPlan1d(&plan, n, FLAGFFT_Z2Z, batch) != FLAGFFT_SUCCESS) {
        ADD_FAILURE() << "flagfftPlan1d Z2Z failed";
        return {};
    }

    cudaStream_t stream = nullptr;
    if (custom_stream) {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamCreate failed";
            flagfftDestroy(plan);
            return {};
        }
        if (flagfftSetStream(plan, stream) != FLAGFFT_SUCCESS) {
            ADD_FAILURE() << "flagfftSetStream failed";
            flagfftDestroy(plan);
            cudaStreamDestroy(stream);
            return {};
        }
    }

    if (flagfftExecZ2Z(plan, d_in, d_out, direction) != FLAGFFT_SUCCESS) {
        ADD_FAILURE() << "flagfftExecZ2Z failed";
        flagfftDestroy(plan);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        return {};
    }
    if (custom_stream) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize failed";
        }
        cudaStreamDestroy(stream);
    } else {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed";
        }
    }

    std::vector<flagfftDoubleComplex> output(input.size());
    EXPECT_EQ(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
    cudaFree(d_out);
    cudaFree(d_in);
    return output;
}

void compare_case_z(int n, int batch, int direction, bool custom_stream = false) {
    require_cuda();
    auto input = sample_input_z(n, batch);
    auto expected = run_cufft_z(input, n, batch, direction);
    auto actual = run_flagfft_z(input, n, batch, direction, custom_stream);
    double scale = direction == FLAGFFT_INVERSE ? static_cast<double>(n) : 1.0;
    assert_close_z(actual, expected, 2e-10 * scale, 2e-10);
}

}  // namespace

TEST(FlagFFTCApi, C2CLeafMatchesCuFFTForwardBatches) {
    compare_case(16, 1, FLAGFFT_FORWARD);
    compare_case(19, 4, FLAGFFT_FORWARD);
    compare_case(105, 7, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, C2CLeafMatchesCuFFTInverseBatches) {
    compare_case(16, 2, FLAGFFT_INVERSE);
    compare_case(19, 3, FLAGFFT_INVERSE);
    compare_case(105, 5, FLAGFFT_INVERSE);
}

TEST(FlagFFTCApi, C2CFourStepMatchesCuFFT) {
    compare_case(8192, 1, FLAGFFT_FORWARD);
    compare_case(8192, 3, FLAGFFT_FORWARD);
    compare_case(16384, 1, FLAGFFT_FORWARD);
    compare_case(16384, 64, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, C2CBluesteinMatchesCuFFT) {
    compare_case(331, 1, FLAGFFT_FORWARD);
    compare_case(331, 2, FLAGFFT_INVERSE);
    compare_case(997, 1, FLAGFFT_FORWARD);
    compare_case(1009, 2, FLAGFFT_INVERSE);
}

TEST(FlagFFTCApi, C2CBluesteinLargeAndMixedLengthsMatchCuFFT) {
    compare_case(5 * 331, 2, FLAGFFT_FORWARD);
    compare_case(5 * 331, 1, FLAGFFT_INVERSE);
    compare_case(16385, 1, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, C2CNestedFourStepMatchesCuFFT) {
    compare_case(1 << 23, 1, FLAGFFT_FORWARD);
    compare_case(1 << 23, 2, FLAGFFT_INVERSE);
}

TEST(FlagFFTCApi, Z2ZLeafMatchesCuFFTForwardBatches) {
    compare_case_z(16, 1, FLAGFFT_FORWARD);
    compare_case_z(19, 4, FLAGFFT_FORWARD);
    compare_case_z(105, 7, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, Z2ZLeafMatchesCuFFTInverseBatches) {
    compare_case_z(16, 2, FLAGFFT_INVERSE);
    compare_case_z(19, 3, FLAGFFT_INVERSE);
    compare_case_z(105, 5, FLAGFFT_INVERSE);
}

TEST(FlagFFTCApi, Z2ZFourStepMatchesCuFFT) {
    compare_case_z(8192, 1, FLAGFFT_FORWARD);
    compare_case_z(8192, 3, FLAGFFT_INVERSE);
    compare_case_z(16384, 1, FLAGFFT_FORWARD);
    compare_case_z(16384, 4, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, Z2ZBluesteinMatchesCuFFT) {
    compare_case_z(331, 1, FLAGFFT_FORWARD);
    compare_case_z(997, 2, FLAGFFT_INVERSE);
    compare_case_z(1009, 2, FLAGFFT_FORWARD);
    compare_case_z(16385, 1, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, Z2ZNestedFourStepMatchesCuFFT) {
    compare_case_z(1 << 23, 1, FLAGFFT_FORWARD);
}

TEST(FlagFFTCApi, Z2ZSetStreamUsesProvidedStream) {
    compare_case_z(16, 3, FLAGFFT_FORWARD, true);
    compare_case_z(8192, 2, FLAGFFT_INVERSE, true);
}

TEST(FlagFFTCApi, SetStreamUsesProvidedStream) {
    compare_case(16, 3, FLAGFFT_FORWARD, true);
    compare_case(8192, 3, FLAGFFT_FORWARD, true);
}

TEST(FlagFFTCApi, RejectsInvalidAndUnsupportedCalls) {
    require_cuda();

    EXPECT_EQ(flagfftPlan1d(nullptr, 16, FLAGFFT_C2C, 1), FLAGFFT_INVALID_VALUE);

    flagfftHandle plan = nullptr;
    int dims2[2] = {8, 8};
    EXPECT_EQ(flagfftPlanMany(&plan, 2, dims2, nullptr, 1, 64, nullptr, 1, 64,
                              FLAGFFT_C2C, 1),
              FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(plan, nullptr);

    ASSERT_EQ(flagfftPlan1d(&plan, 16, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
    EXPECT_EQ(flagfftExecC2C(plan, nullptr, nullptr, FLAGFFT_FORWARD), FLAGFFT_INVALID_VALUE);
    EXPECT_EQ(flagfftExecC2C(plan,
                             reinterpret_cast<flagfftComplex *>(0x1),
                             reinterpret_cast<flagfftComplex *>(0x2),
                             0),
              FLAGFFT_INVALID_VALUE);
    // Type mismatch: Z2Z exec on a C2C plan must be rejected with INVALID_TYPE.
    EXPECT_EQ(flagfftExecZ2Z(plan,
                             reinterpret_cast<flagfftDoubleComplex *>(0x1),
                             reinterpret_cast<flagfftDoubleComplex *>(0x2),
                             FLAGFFT_FORWARD),
              FLAGFFT_INVALID_TYPE);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);

    flagfftHandle zplan = nullptr;
    ASSERT_EQ(flagfftPlan1d(&zplan, 16, FLAGFFT_Z2Z, 1), FLAGFFT_SUCCESS);
    EXPECT_EQ(flagfftExecZ2Z(zplan, nullptr, nullptr, FLAGFFT_FORWARD), FLAGFFT_INVALID_VALUE);
    EXPECT_EQ(flagfftExecZ2Z(zplan,
                             reinterpret_cast<flagfftDoubleComplex *>(0x1),
                             reinterpret_cast<flagfftDoubleComplex *>(0x2),
                             0),
              FLAGFFT_INVALID_VALUE);
    EXPECT_EQ(flagfftExecC2C(zplan,
                             reinterpret_cast<flagfftComplex *>(0x1),
                             reinterpret_cast<flagfftComplex *>(0x2),
                             FLAGFFT_FORWARD),
              FLAGFFT_INVALID_TYPE);
    // R2C / C2R remain unimplemented.
    EXPECT_EQ(flagfftExecR2C(zplan, nullptr, nullptr), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(flagfftExecD2Z(zplan, nullptr, nullptr), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(flagfftExecC2R(zplan, nullptr, nullptr), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(flagfftExecZ2D(zplan, nullptr, nullptr), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(flagfftDestroy(zplan), FLAGFFT_SUCCESS);
}
