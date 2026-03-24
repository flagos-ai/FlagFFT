"""
Unit tests for Triton radix-3 FFT implementation.

Tests correctness and performance against torch.fft.
"""

import pytest
import torch
import time
import numpy as np
from src.codelet.triton.radix_3 import (
    triton_fwd_rad3_b1,
    triton_inv_rad3_b1,
    batch_fwd_rad3_b1,
    batch_inv_rad3_b1,
)


def torch_fft_rad3(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 3-point FFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 3)

    Returns:
        Output tensor with 3-point FFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.fft(x, n=3, dim=-1)
    else:
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.fft(x_complex, n=3, dim=-1)


def torch_ifft_rad3(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 3-point IFFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 3)

    Returns:
        Output tensor with 3-point IFFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.ifft(x, n=3, dim=-1)
    else:
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.ifft(x_complex, n=3, dim=-1)


class TestRadix3Correctness:
    """Correctness tests for radix-3 FFT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.tolerance = 1e-5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            pytest.skip("Triton requires CUDA device")

    def test_single_group_complex64(self):
        """Test single group of complex64 values."""
        x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], device=self.device)

        # Triton result
        result_triton = batch_fwd_rad3_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad3(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Triton: {result_triton}, Torch: {result_torch}"

    def test_batch_complex64(self):
        """Test batch of complex64 groups."""
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad3_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad3(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_large_batch_complex64(self):
        """Test large batch of complex64 groups."""
        torch.manual_seed(42)
        n_groups = 65536
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad3_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad3(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_inverse_consistency_complex(self):
        """Test that inverse FFT is consistent with forward FFT for complex data.

        Note: For radix-3, IFFT(FFT(x)) = x / 3
        """
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        # Forward then inverse (with scaling) should recover original
        y = batch_fwd_rad3_b1(x)
        x_recovered = batch_inv_rad3_b1(y) / 3.0

        assert torch.allclose(x, x_recovered, atol=self.tolerance), \
            f"Max diff: {(x - x_recovered).abs().max()}"

    def test_specific_values_complex(self):
        """Test with specific known values for complex inputs."""
        test_cases = [
            # Input: [a+bi, c+di, e+fi]
            (
                [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],  # [1, 1, 1]
                [3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],  # sum = 3
            ),
            (
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],  # [1, 0, 0]
                [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],  # all ones
            ),
            (
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],  # [0, 0, 0]
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],  # all zeros
            ),
        ]

        for input_vals, expected_vals in test_cases:
            x = torch.tensor(input_vals, device=self.device)
            expected = torch.tensor(expected_vals, device=self.device)

            result = batch_fwd_rad3_b1(x)

            assert torch.allclose(result, expected, atol=self.tolerance), \
                f"Input: {input_vals}, Expected: {expected_vals}, Got: {result}"


class TestRadix3Performance:
    """Performance benchmarks for radix-3 FFT."""

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
            _ = batch_fwd_rad3_b1(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = batch_fwd_rad3_b1(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / iterations

    def benchmark_torch(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark torch.fft implementation."""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = torch_fft_rad3(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch_fft_rad3(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / iterations

    def test_performance_1k_groups_complex(self):
        """Benchmark performance for 1K groups of complex values."""
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1K groups (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_64k_groups_complex(self):
        """Benchmark performance for 64K groups of complex values."""
        torch.manual_seed(42)
        n_groups = 65536
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n64K groups (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    def test_performance_1m_groups_complex(self):
        """Benchmark performance for 1M groups of complex values."""
        torch.manual_seed(42)
        n_groups = 1048576  # 2^20
        x = torch.randn(n_groups, 3, dtype=torch.complex64, device=self.device)

        triton_time = self.benchmark_triton(x, self.benchmark_iterations)
        torch_time = self.benchmark_torch(x, self.benchmark_iterations)

        speedup = torch_time / triton_time

        print(f"\n1M groups (complex):")
        print(f"  Triton: {triton_time*1000:.4f} ms")
        print(f"  Torch:  {torch_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")


def run_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
