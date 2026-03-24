"""
Unit tests for Triton radix-7 FFT implementation.

Tests correctness and performance against torch.fft.
"""

import pytest
import torch
from src.codelet.triton.radix_7 import (
    triton_fwd_rad7_b1,
    triton_inv_rad7_b1,
    batch_fwd_rad7_b1,
    batch_inv_rad7_b1,
    batch_fwd_rad7_soa,
    batch_inv_rad7_soa,
    batch_fwd_rad7_matmul,
    batch_inv_rad7_matmul,
)


def torch_fft_rad7(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 7-point FFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 7)

    Returns:
        Output tensor with 7-point FFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.fft(x, n=7, dim=-1)
    else:
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.fft(x_complex, n=7, dim=-1)


def torch_ifft_rad7(x: torch.Tensor) -> torch.Tensor:
    """
    Reference 7-point IFFT using torch.fft.

    Args:
        x: Input tensor of shape (..., 7)

    Returns:
        Output tensor with 7-point IFFT applied.
    """
    if torch.is_complex(x):
        return torch.fft.ifft(x, n=7, dim=-1)
    else:
        x_complex = torch.complex(x, torch.zeros_like(x))
        return torch.fft.ifft(x_complex, n=7, dim=-1)


class TestRadix7Correctness:
    """Correctness tests for radix-7 FFT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.tolerance = 1e-5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            pytest.skip("Triton requires CUDA device")

    def test_single_group_complex64(self):
        """Test single group of complex64 values."""
        x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j, 7.0 + 8.0j,
                          9.0 + 10.0j, 11.0 + 12.0j, 13.0 + 14.0j], device=self.device)

        # Triton result
        result_triton = batch_fwd_rad7_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad7(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Triton: {result_triton}, Torch: {result_torch}"

    def test_batch_complex64(self):
        """Test batch of complex64 groups."""
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad7_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad7(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_large_batch_complex64(self):
        """Test large batch of complex64 groups."""
        torch.manual_seed(42)
        n_groups = 65536
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)

        # Triton result
        result_triton = batch_fwd_rad7_b1(x)

        # torch.fft result
        result_torch = torch_fft_rad7(x)

        assert torch.allclose(result_triton, result_torch, atol=self.tolerance), \
            f"Max diff: {(result_triton - result_torch).abs().max()}"

    def test_soa_matches_aos(self):
        """SoA kernel must produce identical results to AoS kernel."""
        torch.manual_seed(42)
        x = torch.randn(65536, 7, dtype=torch.complex64, device=self.device)
        assert torch.allclose(batch_fwd_rad7_soa(x), batch_fwd_rad7_b1(x), atol=self.tolerance), \
            "SoA forward differs from AoS"
        assert torch.allclose(batch_inv_rad7_soa(x), batch_inv_rad7_b1(x), atol=self.tolerance), \
            "SoA inverse differs from AoS"

    def test_matmul_matches_aos(self):
        """DFT matrix (tl.dot) kernel must match AoS butterfly kernel."""
        torch.manual_seed(42)
        x = torch.randn(65536, 7, dtype=torch.complex64, device=self.device)
        assert torch.allclose(batch_fwd_rad7_matmul(x), batch_fwd_rad7_b1(x), atol=self.tolerance), \
            f"matmul fwd diff: {(batch_fwd_rad7_matmul(x) - batch_fwd_rad7_b1(x)).abs().max()}"
        assert torch.allclose(batch_inv_rad7_matmul(x), batch_inv_rad7_b1(x), atol=self.tolerance), \
            f"matmul inv diff: {(batch_inv_rad7_matmul(x) - batch_inv_rad7_b1(x)).abs().max()}"

    def test_inverse_consistency_complex(self):
        """Test that inverse FFT is consistent with forward FFT for complex data.

        Note: For radix-7, IFFT(FFT(x)) = x / 7
        """
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)

        # Forward then inverse (with scaling) should recover original
        y = batch_fwd_rad7_b1(x)
        x_recovered = batch_inv_rad7_b1(y) / 7.0

        assert torch.allclose(x, x_recovered, atol=self.tolerance), \
            f"Max diff: {(x - x_recovered).abs().max()}"

    def test_specific_values_complex(self):
        """Test with specific known values for complex inputs."""
        test_cases = [
            # All ones -> [7, 0, 0, 0, 0, 0, 0]
            (
                [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                [7.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ),
            # Single impulse -> all ones
            (
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            ),
            # All zeros
            (
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ),
        ]

        for input_vals, expected_vals in test_cases:
            x = torch.tensor(input_vals, device=self.device)
            expected = torch.tensor(expected_vals, device=self.device)

            result = batch_fwd_rad7_b1(x)

            assert torch.allclose(result, expected, atol=self.tolerance), \
                f"Input: {input_vals}, Expected: {expected_vals}, Got: {result}"


class TestRadix7Performance:
    """Performance benchmarks for radix-7 FFT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            pytest.skip("Triton requires CUDA device")
        self.warmup_iterations = 10
        self.benchmark_iterations = 100

    def benchmark_triton(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark Triton AoS implementation using CUDA Events."""
        for _ in range(self.warmup_iterations):
            _ = batch_fwd_rad7_b1(x)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = batch_fwd_rad7_b1(x)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / 1000 / iterations  # seconds

    def benchmark_soa(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark Triton SoA implementation using CUDA Events."""
        for _ in range(self.warmup_iterations):
            _ = batch_fwd_rad7_soa(x)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = batch_fwd_rad7_soa(x)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / 1000 / iterations  # seconds

    def benchmark_matmul(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark Triton DFT-matrix (tl.dot) implementation."""
        for _ in range(self.warmup_iterations):
            _ = batch_fwd_rad7_matmul(x)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = batch_fwd_rad7_matmul(x)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / 1000 / iterations  # seconds

    def benchmark_torch(self, x: torch.Tensor, iterations: int) -> float:
        """Benchmark torch.fft implementation using CUDA Events."""
        for _ in range(self.warmup_iterations):
            _ = torch_fft_rad7(x)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = torch_fft_rad7(x)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / 1000 / iterations  # seconds

    def _print_bench(self, label: str, n_groups: int, aos_t: float, soa_t: float, matmul_t: float, torch_t: float):
        print(f"\n{label} ({n_groups:,} groups):")
        print(f"  Triton AoS:    {aos_t*1000:.4f} ms")
        print(f"  Triton SoA:    {soa_t*1000:.4f} ms  ({aos_t/soa_t:.2f}x vs AoS)")
        print(f"  Triton matmul: {matmul_t*1000:.4f} ms  ({aos_t/matmul_t:.2f}x vs AoS)")
        print(f"  torch.fft:     {torch_t*1000:.4f} ms")
        print(f"  matmul vs torch: {torch_t/matmul_t:.2f}x")

    def test_performance_16_groups_complex(self):
        torch.manual_seed(42)
        n_groups = 16
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)
        self._print_bench("16 groups", n_groups,
                          self.benchmark_triton(x, self.benchmark_iterations),
                          self.benchmark_soa(x, self.benchmark_iterations),
                          self.benchmark_matmul(x, self.benchmark_iterations),
                          self.benchmark_torch(x, self.benchmark_iterations))

    def test_performance_1k_groups_complex(self):
        torch.manual_seed(42)
        n_groups = 1024
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)
        self._print_bench("1K groups", n_groups,
                          self.benchmark_triton(x, self.benchmark_iterations),
                          self.benchmark_soa(x, self.benchmark_iterations),
                          self.benchmark_matmul(x, self.benchmark_iterations),
                          self.benchmark_torch(x, self.benchmark_iterations))

    def test_performance_64k_groups_complex(self):
        torch.manual_seed(42)
        n_groups = 65536
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)
        self._print_bench("64K groups", n_groups,
                          self.benchmark_triton(x, self.benchmark_iterations),
                          self.benchmark_soa(x, self.benchmark_iterations),
                          self.benchmark_matmul(x, self.benchmark_iterations),
                          self.benchmark_torch(x, self.benchmark_iterations))

    def test_performance_1m_groups_complex(self):
        torch.manual_seed(42)
        n_groups = 1048576
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)
        self._print_bench("1M groups", n_groups,
                          self.benchmark_triton(x, self.benchmark_iterations),
                          self.benchmark_soa(x, self.benchmark_iterations),
                          self.benchmark_matmul(x, self.benchmark_iterations),
                          self.benchmark_torch(x, self.benchmark_iterations))

    def test_performance_10m_groups_complex(self):
        torch.manual_seed(42)
        n_groups = 10_000_000
        x = torch.randn(n_groups, 7, dtype=torch.complex64, device=self.device)
        self._print_bench("10M groups", n_groups,
                          self.benchmark_triton(x, self.benchmark_iterations),
                          self.benchmark_soa(x, self.benchmark_iterations),
                          self.benchmark_matmul(x, self.benchmark_iterations),
                          self.benchmark_torch(x, self.benchmark_iterations))


def run_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
