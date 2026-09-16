// Isolated native-SDK FP64 arithmetic probe; no Python GPU array dependency.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

__global__ void arithmetic(const double *input, double *output) {
  int i = threadIdx.x;
  if (i < 3) output[i] = (input[i] - 1.0) * 3.0;
}

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main() {
  try {
    double input[] = {1.0 + 0x1p-40, 1.0 - 0x1p-40, -1.0 + 0x1p-40};
    double output[3] = {};
    double *x = nullptr, *y = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&x), sizeof(input)));
    check(cudaMalloc(reinterpret_cast<void **>(&y), sizeof(output)));
    check(cudaMemcpy(x, input, sizeof(input), cudaMemcpyHostToDevice));
    arithmetic<<<1, 64>>>(x, y);
    check(cudaGetLastError());
    check(cudaMemcpy(output, y, sizeof(output), cudaMemcpyDeviceToHost));
    for (double value : output) std::printf("%.17g\n", value);
    check(cudaFree(x));
    check(cudaFree(y));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
