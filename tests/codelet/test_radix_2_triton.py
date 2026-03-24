"""
Unit tests for Triton radix-2 FFT implementation.

Tests correctness and performance against torch.fft.
"""

import pytest
import torch
import time
import numpy as np
from src.codelet.triton.radix_2 import (
    triton_fwd_rad2_b1,
    triton_inv_rad2_b1,
    batch_fwd_rad2_b1,
    batch_inv_rad2_b1,
)


def torch_fft_rad2(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 2-point FFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 2)

    Returns:
        Output tensor with 2-point FFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.fft(x, n=2, dim=-1)
    else:
        # For real inputs, convert to complex, compute FFT, return result
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.fft(x_complex, n=2, dim=-1)


def torch_ifft_rad2(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 2-point IFFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 2)

    Returns:
        Output tensor with 2-point IFFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.ifft(x, n=2, dim=-1)
    else:
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.ifft(x_complex, n=2, dim=-1)


class TestRadix2Correctness:
    """Correctness tests for radix-2 FFT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.tolerance = 1e-5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            pytest.skip("Triton requires CUDA device")

    def test_single_pair_float32(self):
        """Test single pair of float32 values."""
        x = torch.tensor([1.0, 2.0], device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result (complex output for real input)
        result_torch = torch_fft_rad2(x)

        # For real input [a, b], FFT gives [a+b, a-b] (both real)
        assert torch.allclose(result_triton, result_torch.real, atol=self.tolerance), \
            f"Triton: {result_triton}, Torch: {result_torch}"

    def test_single_pair_complex64(self):
        """Test single pair of complex64 values."""
        x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad2(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Triton: {result_triton}, Torch: {result_torch}"

    def test_batch_float32(self):
        """Test batch of float32 pairs."""
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad2(x)

        assert torch.allclose(result_triton, result_torch.real, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch.real).abs().max()}"

    def test_batch_complex64(self):
        """Test batch of complex64 pairs."""
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad2(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_large_batch(self):
        """Test large batch of pairs."""
        torch.manual_seed(42)
        n_pairs = 65536
        x = torch.randn(n_pairs, 2, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad2(x)

        assert torch.allclose(result_triton, result_torch.real, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch.real).abs().max()}"

    def test_large_batch_complex64(self):
        """Test large batch of complex64 pairs."""
        torch.manual_seed(42)
        n_pairs = 65536
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad2_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad2(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_inverse_consistency_real(self):
        """Test that inverse FFT is consistent with forward FFT for real data.

        Note: For radix-2, FFT(FFT(x)) = 2*x, so IFFT(x) = FFT(x) / 2
        """
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, device=self.device)

        # Forward then inverse (with scaling) should recover original
        y = batch_fwd_rad2_b1(x)
        # IFFT = FFT / 2
        x_recovered = batch_inv_rad2_b1(y) / 2.0

        assert torch.allclose(x, x_recovered, atol=self.tolerance), \
            f"Max diff: {(x - x_recovered).abs().max()}"

    def test_inverse_consistency_complex(self):
        """Test that inverse FFT is consistent with forward FFT for complex data.

        Note: For radix-2, FFT(FFT(x)) = 2*x, so IFFT(x) = FFT(x) / 2
        """
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        # Forward then inverse (with scaling) should recover original
        y = batch_fwd_rad2_b1(x)
        # IFFT = FFT / 2
        x_recovered = batch_inv_rad2_b1(y) / 2.0

        assert torch.allclose(x, x_recovered, atol=self.tolerance), \
            f"Max diff: {(x - x_recovered).abs().max()}"

    def test_specific_values_real(self):
        """Test with specific known values for real inputs."""
        # [a, b] -> [a+b, a-b]
        test_cases = [
            ([0.0, 0.0], [0.0, 0.0]),
            ([1.0, 0.0], [1.0, 1.0]),
            ([0.0, 1.0], [1.0, -1.0]),
            ([1.0, 1.0], [2.0, 0.0]),
            ([2.0, 3.0], [5.0, -1.0]),
            ([-1.0, 2.0], [1.0, -3.0]),
        ]

        for input_vals, expected_vals in test_cases:
            x = torch.tensor(input_vals, device=self.device)
            expected = torch.tensor(expected_vals, device=self.device)

            result = batch_fwd_rad2_b1(x)

            assert torch.allclose(result, expected, atol=self.tolerance), \
                f"Input: {input_vals}, Expected: {expected_vals}, Got: {result}"

    def test_specific_values_complex(self):
        """Test with specific known values for complex inputs."""
        # For complex: (a+bi), (c+di) -> (a+c + (b+d)i), (a-c + (b-d)i)
        test_cases = [
            # Input: [(a+bi), (c+di)]
            # Expected: [(a+c + (b+d)i), (a-c + (b-d)i)]
            (
                [1.0 + 0.0j, 1.0 + 0.0j],  # [1, 1]
                [2.0 + 0.0j, 0.0 + 0.0j],  # [2, 0]
            ),
            (
                [1.0 + 1.0j, 2.0 + 3.0j],  # [1+i, 2+3i]
                [3.0 + 4.0j, -1.0 - 2.0j],  # [3+4i, -1-2i]
            ),
            (
                [0.0 + 0.0j, 0.0 + 0.0j],  # [0, 0]
                [0.0 + 0.0j, 0.0 + 0.0j],  # [0, 0]
            ),
            (
                [1.0 + 2.0j, 3.0 + 4.0j],  # [1+2i, 3+4i]
                [4.0 + 6.0j, -2.0 - 2.0j],  # [4+6i, -2-2i]
            ),
        ]

        for input_vals, expected_vals in test_cases:
            x = torch.tensor(input_vals, device=self.device)
            expected = torch.tensor(expected_vals, device=self.device)

            result = batch_fwd_rad2_b1(x)

            assert torch.allclose(result, expected, atol=self.tolerance), \
                f"Input: {input_vals}, Expected: {expected_vals}, Got: {result}"


class TestRadix2Performance:
    """Performance benchmarks for radix-2 FFT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            pytest.skip("Triton requires CUDA device")
        self.warmup_iterations = 10
        self.benchmark_iterations = 100

    def benchmark_triton(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark Triton implementation."""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = batch_fwd_rad2_b1(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = batch_fwd_rad2_b1(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / iterations

    def benchmark_torch(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark torch.fft implementation."""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = torch_fft_rad2(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch_fft_rad2(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / iterations

    def test_performance_1k_pairs_real(self):
        """Benchmark performance for 1K pairs of real values."""
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1K pairs (real):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_64k_pairs_real(self):
        """Benchmark performance for 64K pairs of real values."""
        torch.manual_seed(42)
        n_pairs = 65536
        x = torch.randn(n_pairs, 2, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n64K pairs (real):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_1m_pairs_real(self):
        """Benchmark performance for 1M pairs of real values."""
        torch.manual_seed(42)
        n_pairs = 1048576  # 2^20
        x = torch.randn(n_pairs, 2, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1M pairs (real):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_1k_pairs_complex(self):
        """Benchmark performance for 1K pairs of complex values."""
        torch.manual_seed(42)
        n_pairs = 1024
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1K pairs (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_64k_pairs_complex(self):
        """Benchmark performance for 64K pairs of complex values."""
        torch.manual_seed(42)
        n_pairs = 65536
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n64K pairs (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_1m_pairs_complex(self):
        """Benchmark performance for 1M pairs of complex values."""
        torch.manual_seed(42)
        n_pairs = 1048576
        x = torch.randn(n_pairs, 2, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1M pairs (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")


def run_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
