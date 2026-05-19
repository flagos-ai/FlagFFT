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
    assert_close(actual, expected, 1.5e-3f * scale, 1.5e-3f);
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

TEST(FlagFFTCApi, SetStreamUsesProvidedStream) {
    compare_case(16, 3, FLAGFFT_FORWARD, true);
    compare_case(8192, 3, FLAGFFT_FORWARD, true);
}

TEST(FlagFFTCApi, RejectsInvalidAndUnsupportedCalls) {
    require_cuda();

    EXPECT_EQ(flagfftPlan1d(nullptr, 16, FLAGFFT_C2C, 1), FLAGFFT_INVALID_VALUE);

    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan1d(&plan, 16, FLAGFFT_Z2Z, 1), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(plan, nullptr);

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
    EXPECT_EQ(flagfftExecZ2Z(plan, nullptr, nullptr, FLAGFFT_FORWARD), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}
