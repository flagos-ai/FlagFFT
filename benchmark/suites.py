"""Benchmark size suites for FlagFFT performance testing."""

# Full benchmark suite: representative powers-of-two, primes, and composites
FULL_SIZES = [16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384]

# Smoke suite: quick validation subset
SMOKE_SIZES = [256, 512]

# Default FFT types to benchmark
DEFAULT_APIS = ["c2c", "z2z"]

# Default directions to benchmark
DEFAULT_DIRECTIONS = ["forward", "inverse"]
