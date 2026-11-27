// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <mc_runtime.h>

namespace {

constexpr unsigned extent = 256;
constexpr unsigned plane_size = extent * extent;
constexpr unsigned volume = extent * plane_size;
constexpr unsigned threads = 4 * 64;

template<unsigned Order, unsigned Tile>
__global__ void transpose_cube(const double2* input, double2* output) {
  constexpr unsigned columns = Order == 120 ? plane_size : extent;
  constexpr unsigned rows = Order == 201 ? plane_size : extent;
  constexpr unsigned col_blocks = columns / Tile;
  constexpr unsigned tiles_per_plane = rows / Tile * col_blocks;
  __shared__ double2 tile[Tile][Tile + 1];

  const unsigned plane = blockIdx.x / tiles_per_plane;
  const unsigned block = blockIdx.x % tiles_per_plane;
  const unsigned row0 = block / col_blocks * Tile;
  const unsigned col0 = block % col_blocks * Tile;
  const auto base = static_cast<std::size_t>(blockIdx.y) * volume;

  for (unsigned index = threadIdx.x; index < Tile * Tile; index += threads) {
    const unsigned row = row0 + index / Tile;
    const unsigned col = col0 + index % Tile;
    std::size_t offset;
    if constexpr (Order == 21) {
      offset = plane * plane_size + row * extent + col;
    } else if constexpr (Order == 210) {
      offset = (row * extent + plane) * extent + col;
    } else {
      offset = row * columns + col;
    }
    tile[index / Tile][index % Tile] = input[base + offset];
  }
  __syncthreads();

  for (unsigned index = threadIdx.x; index < Tile * Tile; index += threads) {
    const unsigned row = row0 + index % Tile;
    const unsigned col = col0 + index / Tile;
    std::size_t offset;
    if constexpr (Order == 21) {
      offset = plane * plane_size + col * extent + row;
    } else if constexpr (Order == 210) {
      offset = (col * extent + plane) * extent + row;
    } else {
      offset = col * rows + row;
    }
    output[base + offset] = tile[index % Tile][index / Tile];
  }
}

template<unsigned Order, unsigned Tile>
int launch(const void* input, void* output, unsigned batch, void* stream) {
  const void* kernel = reinterpret_cast<const void*>(transpose_cube<Order, Tile>);
  void* args[] = {&input, &output};
  return mcLaunchKernel(kernel, dim3(volume / (Tile * Tile), batch), dim3(threads), args, 0,
                        reinterpret_cast<mcStream_t>(stream));
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp64_cube_transpose(
    const void* input, void* output, unsigned batch, unsigned order, void* stream) {
  switch (order) {
    case 21: return launch<21, 16>(input, output, batch, stream);
    case 210: return launch<210, 16>(input, output, batch, stream);
    case 201: return launch<201, 16>(input, output, batch, stream);
    case 120: return launch<120, 32>(input, output, batch, stream);
    default: return static_cast<int>(mcErrorInvalidValue);
  }
}
