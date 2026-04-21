from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft_mixed_radix_tle import (
    FFTDecompositionSpec,
    build_fft_plan,
    describe_fft_plan,
    fft_mixed_radix_triton,
    fft_mixed_radix_triton_manual,
)


def _parse_split_spec(raw: str | None) -> FFTDecompositionSpec | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --split-spec JSON: {exc}") from exc


def check_case(
    n: int,
    batch: int = 2,
    split_spec: FFTDecompositionSpec | None = None,
) -> tuple[str, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(n)
    xr = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    xi = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    x = torch.complex(xr, xi)

    if split_spec is None:
        plan = build_fft_plan(n)
        y = fft_mixed_radix_triton(x, plan=plan)
    else:
        plan = build_fft_plan(n, split_spec=split_spec)
        y = fft_mixed_radix_triton_manual(x, split_spec=split_spec)

    ref = torch.fft.fft(x, dim=-1)
    err = (y - ref).abs()
    return describe_fft_plan(plan), float(err.max()), float(err.square().mean().sqrt())


def run_cases(
    lengths: list[int],
    batch: int = 2,
    split_spec: FFTDecompositionSpec | None = None,
) -> None:
    for n in lengths:
        current_batch = 1 if n >= 16384 else batch
        plan_desc, max_err, rms_err = check_case(n, batch=current_batch, split_spec=split_spec)
        mode = "manual" if split_spec is not None else "auto"
        print(
            f"mode={mode} n={n} batch={current_batch} plan={plan_desc} "
            f"max_abs_err={max_err:.6e} rms_err={rms_err:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FFT correctness check. "
            "--split-spec accepts JSON such as '[32, 3200, null, [32, 100]]' "
            "or '{\"split\": [32, 3200], \"col\": [32, 100]}'."
        )
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 34, 105, 936, 4096, 8192, 16384],
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--split-spec", type=str, default=None)
    args = parser.parse_args()

    split_spec = _parse_split_spec(args.split_spec)
    run_cases(args.lengths, batch=args.batch, split_spec=split_spec)


if __name__ == "__main__":
    main()
