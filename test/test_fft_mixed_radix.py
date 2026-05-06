from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flagfft


def _format_plan_node(node: dict, indent: int = 0) -> str:
    prefix = "  " * indent
    kind = node.get("kind", "unknown")
    length = node.get("length", "?")
    if kind == "ct_leaf":
        label = f"{kind}(n={length}, factors={node.get('factors')}, lanes={node.get('lanes')})"
    elif kind == "four_step":
        label = f"{kind}(n={length}, n1={node.get('n1')}, n2={node.get('n2')})"
    else:
        label = f"{kind}(n={length})"
    lines = [prefix + label]
    if kind == "four_step":
        lines.append(_format_plan_node(node["row"], indent + 1))
        lines.append(_format_plan_node(node["col"], indent + 1))
    return "\n".join(lines)


def check_case(n: int, batch: int = 2) -> tuple[dict, str, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    try:
        import _flagfft_core
    except ImportError as exc:
        raise RuntimeError("_flagfft_core is not built; run `python -m pip install -e .` first") from exc

    torch.manual_seed(n)
    xr = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    xi = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    x = torch.complex(xr, xi)

    plan = _flagfft_core.debug_plan(x)
    y = flagfft.fft(x)
    ref = torch.fft.fft(x, dim=-1)
    err = (y - ref).abs()
    return (
        plan,
        _format_plan_node(plan["root"]),
        float(err.max()),
        float(err.square().mean().sqrt()),
    )


def run_cases(lengths: list[int], batch: int = 2) -> None:
    for n in lengths:
        current_batch = 1 if n >= 16384 else batch
        _plan, plan_desc, max_err, rms_err = check_case(n, batch=current_batch)
        print(
            f"mode=cpp n={n} batch={current_batch} plan={plan_desc} "
            f"max_abs_err={max_err:.6e} rms_err={rms_err:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT correctness check using the C++ FlagFFT backend.")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            19,
            34,
            38,
            60,
            105,
            120,
            190,
            255,
            936,
            1020,
            4096,
            8192,
            16384,
        ],
    )
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    run_cases(args.lengths, batch=args.batch)


if __name__ == "__main__":
    main()
