def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-exe",
        default=None,
        help="Path to the bench_vs_cufft executable. Defaults to FLAGFFT_BENCH_EXE or build/cpp/bench_vs_cufft.",
    )
    parser.addoption(
        "--benchmark-lengths",
        default="16",
        help="Comma-separated FFT lengths for the C++ benchmark pytest smoke.",
    )
    parser.addoption("--benchmark-batch", type=int, default=1)
    parser.addoption("--benchmark-warmup", type=int, default=0)
    parser.addoption("--benchmark-iters", type=int, default=1)
    parser.addoption("--benchmark-launches-per-sample", type=int, default=1)
    parser.addoption("--benchmark-tune-static-limit", type=int, default=1)
    parser.addoption("--benchmark-tune-finalists", type=int, default=1)
    parser.addoption(
        "--benchmark-direction",
        choices=("forward", "inverse"),
        default="forward",
    )
    parser.addoption(
        "--benchmark-tune",
        action="store_true",
        default=False,
        help="Run flagfft tune through bench_vs_cufft before benchmarking.",
    )
    parser.addoption(
        "--benchmark-retune",
        action="store_true",
        default=False,
        help="Run flagfft retune through bench_vs_cufft before benchmarking.",
    )
