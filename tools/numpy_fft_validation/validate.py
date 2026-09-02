#!/usr/bin/env python3
"""Run and analyze an out-of-tree FlagFFT/platform-vs-NumPy validation.

The native capture executable runs FlagFFT and the platform FFT library on the
same device input and stores their host outputs as raw bytes.  This script
then computes a NumPy reference and applies the accuracy definition currently
used by ctest/flagfft_test.h.  ``--analyze-only`` reuses those raw files, so a
different NumPy version or a metric-only change does not require rerunning the
device kernels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


MASK64 = (1 << 64) - 1
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
SEED_TAG = 0x4654464654455354

TYPE_CODES = {
    "r2c": 0x2A,
    "c2r": 0x2C,
    "c2c": 0x29,
    "d2z": 0x6A,
    "z2d": 0x6C,
    "z2z": 0x69,
}

ACCURACY_CONSTANTS = {
    "complex": (1.2419386546059821, 1.9343969087678796),
    "real_forward": (1.234681000407627, 1.8260558195934091),
    "real_inverse": (0.97722970418819066, 1.372182697342486),
}

DEFAULT_SCALES = (2.0**-20, 1.0, 2.0**20)


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_root() -> Path:
    return source_root().parent / "results"


def product(shape: Iterable[int]) -> int:
    value = 1
    for dimension in shape:
        value *= int(dimension)
    return value


def api_class(api: str) -> str:
    if api in ("c2c", "z2z"):
        return "complex"
    if api in ("r2c", "d2z"):
        return "real_forward"
    if api in ("c2r", "z2d"):
        return "real_inverse"
    raise ValueError(f"unknown API: {api}")


def is_double(api: str) -> bool:
    return api in ("z2z", "d2z", "z2d")


def is_complex(api: str) -> bool:
    return api in ("c2c", "z2z")


def is_real_forward(api: str) -> bool:
    return api in ("r2c", "d2z")


def is_real_inverse(api: str) -> bool:
    return api in ("c2r", "z2d")


def real_dtype(api: str) -> np.dtype:
    return np.dtype(np.float64 if is_double(api) else np.float32)


def complex_dtype(api: str) -> np.dtype:
    return np.dtype(np.complex128 if is_double(api) else np.complex64)


def half_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] // 2 + 1)


def input_shape(api: str, shape: tuple[int, ...], batch: int) -> tuple[int, ...]:
    if is_real_inverse(api):
        return (batch, *half_shape(shape))
    return (batch, *shape)


def output_shape(api: str, shape: tuple[int, ...], batch: int) -> tuple[int, ...]:
    if is_real_forward(api):
        return (batch, *half_shape(shape))
    if is_real_inverse(api):
        return (batch, *shape)
    return (batch, *shape)


def splitmix_signed_unit(count: int, seed: int) -> np.ndarray:
    """Match ctest/flagfft_test.h's StableRng::signed_unit sequence."""

    if count == 0:
        return np.empty(0, dtype=np.float64)
    indices = np.arange(count, dtype=np.uint64)
    state = np.uint64(seed & MASK64) + indices * np.uint64(SPLITMIX_INCREMENT)
    value = state
    value = (value ^ (value >> np.uint64(30))) * np.uint64(SPLITMIX_MULTIPLIER_1)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(SPLITMIX_MULTIPLIER_2)
    value = value ^ (value >> np.uint64(31))
    bits = (value >> np.uint64(11)).astype(np.float64)
    return bits * (2.0 / 9007199254740992.0) - 1.0


def accuracy_seed(api: str, transform_elements: int, batch: int, variant: int = 0) -> int:
    return (
        SEED_TAG
        ^ ((TYPE_CODES[api] << 48) & MASK64)
        ^ ((int(transform_elements) << 16) & MASK64)
        ^ int(batch)
        ^ ((int(variant) * SPLITMIX_INCREMENT) & MASK64)
    ) & MASK64


def as_complex_from_interleaved(values: np.ndarray, api: str, shape: tuple[int, ...]) -> np.ndarray:
    dtype = complex_dtype(api)
    scalar = values.astype(real_dtype(api), copy=False)
    return scalar.view(dtype).reshape(shape)


def make_input(api: str, shape: tuple[int, ...], batch: int, scale: float) -> tuple[np.ndarray, int]:
    transform_elements = product(shape)
    seed = accuracy_seed(api, transform_elements, batch)
    target_shape = input_shape(api, shape, batch)
    count = product(target_shape)

    if is_complex(api) or is_real_inverse(api):
        values = splitmix_signed_unit(count * 2, seed)
        result = as_complex_from_interleaved(values, api, target_shape)
        if is_real_inverse(api):
            # cuFFT/muFFT/ACFFT C2R-compatible input: the DC and Nyquist
            # coefficients are real.  NumPy's irfft follows the same rule.
            result[..., 0].imag = 0
            if shape[-1] % 2 == 0:
                result[..., -1].imag = 0
    else:
        result = splitmix_signed_unit(count, seed).astype(real_dtype(api)).reshape(target_shape)

    dtype = result.real.dtype if np.iscomplexobj(result) else result.dtype
    scale_value = np.asarray(scale, dtype=dtype).item()
    if np.iscomplexobj(result):
        # Scale the two scalar components separately, matching the C++ test
        # helper's per-component cast for float32.
        scaled = np.empty_like(result)
        scaled.real = result.real * scale_value
        scaled.imag = result.imag * scale_value
        result = scaled
    else:
        result = (result * scale_value).astype(result.dtype, copy=False)
    return np.ascontiguousarray(result), seed


def numpy_reference(value: np.ndarray, api: str, shape: tuple[int, ...], direction: str) -> np.ndarray:
    axes = tuple(range(1, len(shape) + 1))
    transform_size = product(shape)
    if is_complex(api):
        if direction == "forward":
            return np.fft.fftn(value, s=shape, axes=axes)
        return np.fft.ifftn(value, s=shape, axes=axes) * transform_size
    if is_real_forward(api):
        return np.fft.rfftn(value, s=shape, axes=axes)
    if is_real_inverse(api):
        # The device APIs intentionally use the unnormalized inverse.
        return np.fft.irfftn(value, s=shape, axes=axes) * transform_size
    raise ValueError(f"unknown API: {api}")


def _component_arrays(value: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    value = np.asarray(value)
    reference = np.asarray(reference)
    if np.iscomplexobj(value) or np.iscomplexobj(reference):
        value_real = np.asarray(value.real, dtype=np.longdouble)
        value_imag = np.asarray(value.imag, dtype=np.longdouble)
        ref_real = np.asarray(reference.real, dtype=np.longdouble)
        ref_imag = np.asarray(reference.imag, dtype=np.longdouble)
        diff = np.hypot(value_real - ref_real, value_imag - ref_imag)
        ref_abs = np.hypot(ref_real, ref_imag)
        return diff, ref_abs, True
    value_real = np.asarray(value, dtype=np.longdouble)
    ref_real = np.asarray(reference, dtype=np.longdouble)
    return np.abs(value_real - ref_real), np.abs(ref_real), False


def error_stats(value: np.ndarray, reference: np.ndarray, elements_per_batch: int, batch: int) -> dict[str, Any]:
    diff, ref_abs, _ = _component_arrays(value.reshape(batch, elements_per_batch), reference.reshape(batch, elements_per_batch))
    finite = bool(np.all(np.isfinite(diff)) and np.all(np.isfinite(ref_abs)))

    rel_l2 = np.longdouble(0.0)
    rel_linf = np.longdouble(0.0)
    max_abs = np.longdouble(0.0)
    mixed_pointwise = np.longdouble(0.0)
    worst_l2_batch = 0
    worst_linf_batch = 0

    for batch_index in range(batch):
        batch_diff = diff[batch_index]
        batch_ref = ref_abs[batch_index]
        err_sq = np.sum(batch_diff * batch_diff, dtype=np.longdouble)
        ref_sq = np.sum(batch_ref * batch_ref, dtype=np.longdouble)
        err_max = np.max(batch_diff, initial=np.longdouble(0.0))
        ref_max = np.max(batch_ref, initial=np.longdouble(0.0))
        mixed_max = np.max(batch_diff / np.maximum(batch_ref, np.longdouble(1.0)), initial=np.longdouble(0.0))

        batch_rel_l2 = np.sqrt(err_sq / ref_sq) if ref_sq != 0 else (
            np.longdouble(0.0) if err_sq == 0 else np.longdouble(np.inf)
        )
        batch_rel_linf = err_max / ref_max if ref_max != 0 else (
            np.longdouble(0.0) if err_max == 0 else np.longdouble(np.inf)
        )
        if batch_rel_l2 > rel_l2:
            rel_l2 = batch_rel_l2
            worst_l2_batch = batch_index
        if batch_rel_linf > rel_linf:
            rel_linf = batch_rel_linf
            worst_linf_batch = batch_index
        max_abs = max(max_abs, err_max)
        mixed_pointwise = max(mixed_pointwise, mixed_max)

    return {
        "rel_l2": float(rel_l2),
        "rel_linf": float(rel_linf),
        "max_abs": float(max_abs),
        "mixed_pointwise": float(mixed_pointwise),
        "worst_l2_batch": int(worst_l2_batch),
        "worst_linf_batch": int(worst_linf_batch),
        "finite": finite,
    }


def ceil_log2_covering(value: int) -> int:
    return max(0, (int(value) - 1).bit_length())


def work_factor(n: int) -> float:
    if n <= 64:
        return float(n)
    return float(3 * ceil_log2_covering(2 * n - 1) + 3)


def accuracy_limit(api: str, n: int) -> dict[str, float]:
    constants = ACCURACY_CONSTANTS[api_class(api)]
    unit_roundoff = float(np.finfo(real_dtype(api)).eps) / 2.0
    scale = unit_roundoff * work_factor(n)
    return {
        "rel_l2": constants[0] * scale,
        "rel_linf": constants[1] * scale,
        "normalized_scale": scale,
    }


def judged_stats(stats: dict[str, Any], limits: dict[str, float]) -> dict[str, Any]:
    result = dict(stats)
    result["normalized_l2"] = stats["rel_l2"] / limits["normalized_scale"]
    result["normalized_linf"] = stats["rel_linf"] / limits["normalized_scale"]
    result["passed"] = bool(
        stats["finite"]
        and stats["rel_l2"] <= limits["rel_l2"]
        and stats["rel_linf"] <= limits["rel_linf"]
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(path: Path, api: str, shape: tuple[int, ...], batch: int) -> np.ndarray:
    is_input = path.name == "input.bin"
    complex_values = (is_complex(api) or is_real_inverse(api)) if is_input else (
        is_complex(api) or is_real_forward(api)
    )
    dtype = complex_dtype(api) if complex_values else real_dtype(api)
    expected_shape = input_shape(api, shape, batch) if is_input else output_shape(api, shape, batch)
    data = np.fromfile(path, dtype=dtype)
    expected = product(expected_shape)
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} {dtype} values, got {data.size}")
    return data.reshape(expected_shape)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=True) + "\n")


def parse_scales(raw: str | None, matrix_scales: list[float]) -> list[float]:
    if raw is None:
        return [float(value) for value in matrix_scales]
    if raw.strip().lower() == "all":
        return list(DEFAULT_SCALES)
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--scales must not be empty")
    return values


def parse_shape_filter(raw: str | None) -> set[tuple[int, ...]] | None:
    if raw is None:
        return None
    result: set[tuple[int, ...]] = set()
    for item in raw.split(","):
        dimensions = tuple(
            int(part)
            for part in item.strip().replace("X", "x").replace("x", ",").split(",")
            if part.strip()
        )
        if not dimensions or any(dimension <= 0 for dimension in dimensions) or len(dimensions) > 3:
            raise ValueError(f"invalid shape filter: {item}")
        result.add(dimensions)
    return result


def load_cases(
    source: Path,
    combination: str,
    ops_filter: set[str] | None,
    scale_override: str | None,
    shape_filter: set[tuple[int, ...]] | None,
) -> list[dict[str, Any]]:
    with (source / "conf/operators.yaml").open() as stream:
        operators = yaml.safe_load(stream).get("ops", [])
    with (source / "conf/test_matrix.yaml").open() as stream:
        matrix = yaml.safe_load(stream)

    if combination in ("full", "all"):
        combinations = list(matrix.get("combinations", {}).keys())
    else:
        combinations = [part.strip() for part in combination.split(",") if part.strip()]
        unknown = [name for name in combinations if name not in matrix.get("combinations", {})]
        if unknown:
            raise ValueError(f"unknown combination(s): {', '.join(unknown)}")

    cases: list[dict[str, Any]] = []
    for combo_name in combinations:
        combo = matrix["combinations"][combo_name]
        sizes = matrix[combo["sizes"]] if isinstance(combo["sizes"], str) else combo["sizes"]
        batches = matrix[combo.get("batches", [1])] if isinstance(combo.get("batches", [1]), str) else combo.get("batches", [1])
        scales = matrix[combo.get("scales", [1.0])] if isinstance(combo.get("scales", [1.0]), str) else combo.get("scales", [1.0])
        scales = parse_scales(scale_override, scales)
        rank = combo.get("rank")
        if rank is None:
            rank = len(sizes[0]) if isinstance(sizes[0], list) else 1
        algorithms = combo.get("algorithms")
        if algorithms is None:
            if rank == 1:
                name_parts = set(combo_name.split("_"))
                algorithms = [algorithm for algorithm in ("ct", "bs") if algorithm in name_parts]
                if not algorithms:
                    algorithms = ["ct", "bs"]
            else:
                algorithms = [f"{rank}d"]

        for operator in operators:
            op_id = operator["id"]
            if ops_filter is not None and op_id not in ops_filter:
                continue
            if operator.get("rank") != rank:
                continue
            for algorithm in operator["algorithms"]:
                if algorithm not in algorithms:
                    continue
                for size in sizes:
                    dimensions = tuple(int(item) for item in (size if isinstance(size, list) else [size]))
                    if shape_filter is not None and dimensions not in shape_filter:
                        continue
                    for batch in batches:
                        for scale in scales:
                            for direction in operator["directions"]:
                                cases.append({
                                    "combination": combo_name,
                                    "op_id": op_id,
                                    "algorithm": algorithm,
                                    "api": operator["cli_type"],
                                    "rank": rank,
                                    "shape": list(dimensions),
                                    "batch": int(batch),
                                    "scale": float(scale),
                                    "direction": direction,
                                })
    return cases


def case_name(index: int, case: dict[str, Any]) -> str:
    shape = "x".join(str(value) for value in case["shape"])
    scale = f"{case['scale']:.17g}".replace("-", "m").replace(".", "p")
    return f"{index:05d}_{case['op_id']}_{case['algorithm']}_{case['direction']}_n{shape}_b{case['batch']}_s{scale}"


def git_commit(source: Path) -> str:
    commands = [["git", "-C", str(source), "rev-parse", "HEAD"]]
    git_pointer = source / ".git"
    if git_pointer.is_file():
        pointer = git_pointer.read_text().strip()
        if pointer.startswith("gitdir:"):
            gitdir = Path(pointer.split(":", 1)[1].strip())
            commands.append(["git", "--git-dir", str(gitdir), "rev-parse", "HEAD"])
            # A worktree created on the host may retain the host-side absolute
            # admin path in .git.  The standard development container mounts
            # the workspace at /workspace, so translate that one known mount
            # point when the first command cannot see the host path.
            host_workspace = Path("/rjs/llb/fft-dev")
            container_workspace = Path("/workspace")
            try:
                relative = gitdir.relative_to(host_workspace)
            except ValueError:
                pass
            else:
                commands.append(["git", "--git-dir", str(container_workspace / relative), "rev-parse", "HEAD"])
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return completed.stdout.strip()
    return "unknown"


def environment_snapshot(backend: str | None, source: Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "system": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "backend": backend or "unknown",
        "visible_devices": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "MUSA_VISIBLE_DEVICES": os.environ.get("MUSA_VISIBLE_DEVICES", ""),
            "PPU_VISIBLE_DEVICES": os.environ.get("PPU_VISIBLE_DEVICES", ""),
        },
    }
    snapshot["git_commit"] = git_commit(source or source_root())
    return snapshot


def analyze_case(case_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    api = case["api"]
    shape = tuple(int(value) for value in case["shape"])
    batch = int(case["batch"])
    transform_elements = product(shape)
    input_file = case_dir / "input.bin"
    flagfft_file = case_dir / "flagfft.bin"
    platform_file = case_dir / "platform.bin"
    input_value = load_raw(input_file, api, shape, batch)
    flagfft_value = load_raw(flagfft_file, api, shape, batch)
    platform_value = load_raw(platform_file, api, shape, batch)
    numpy_value = np.ascontiguousarray(numpy_reference(input_value, api, shape, case["direction"]))
    np.save(case_dir / "numpy.npy", numpy_value, allow_pickle=False)

    output_elements = product(output_shape(api, shape, batch)[1:])
    limits = accuracy_limit(api, transform_elements)
    flagfft_stats = judged_stats(
        error_stats(flagfft_value, numpy_value, output_elements, batch), limits
    )
    platform_stats = judged_stats(
        error_stats(platform_value, numpy_value, output_elements, batch), limits
    )
    pairwise_stats = error_stats(flagfft_value, platform_value, output_elements, batch)

    result = dict(case)
    prior_capture_status = case.get("status", {}).get("capture", "passed")
    if prior_capture_status == "running":
        prior_capture_status = "passed"
    result.update({
        "input_file": str(input_file.name),
        "flagfft_file": str(flagfft_file.name),
        "platform_file": str(platform_file.name),
        "numpy_file": "numpy.npy",
        "input_sha256": sha256(input_file),
        "flagfft_sha256": sha256(flagfft_file),
        "platform_sha256": sha256(platform_file),
        "numpy_sha256": sha256(case_dir / "numpy.npy"),
        "numpy_dtype": str(numpy_value.dtype),
        "metric": {
            "definition": "ctest/flagfft_test.h::error_stats",
            "transform_class": api_class(api),
            "transform_elements": transform_elements,
            "output_elements_per_transform": output_elements,
            "limits": limits,
            "flagfft_vs_numpy": flagfft_stats,
            "platform_vs_numpy": platform_stats,
            "flagfft_vs_platform": pairwise_stats,
        },
        "status": {
            "capture": prior_capture_status,
            "flagfft": "passed" if flagfft_stats["passed"] else "failed",
            "platform": "passed" if platform_stats["passed"] else "failed",
            "both": bool(flagfft_stats["passed"] and platform_stats["passed"]),
        },
    })
    write_json(case_dir / "case.json", result)
    return result


def write_summary(output_dir: Path, records: list[dict[str, Any]], config: dict[str, Any], environment: dict[str, Any], duration: float) -> dict[str, Any]:
    summary = {
        "format_version": 1,
        "reference": {
            "implementation": "numpy.fft",
            "normalization": "backward API semantics; device inverse is multiplied by product(shape)",
            "dtype_policy": "NumPy computes its native reference dtype; device outputs retain their native dtype",
        },
        "environment": environment,
        "config": config,
        "cases": records,
        "summary": {
            "total": len(records),
            "flagfft_passed": sum(record.get("status", {}).get("flagfft") == "passed" for record in records),
            "flagfft_failed": sum(record.get("status", {}).get("flagfft") == "failed" for record in records),
            "platform_passed": sum(record.get("status", {}).get("platform") == "passed" for record in records),
            "platform_failed": sum(record.get("status", {}).get("platform") == "failed" for record in records),
            "both_passed": sum(record.get("status", {}).get("both", False) for record in records),
            "capture_errors": sum(record.get("status", {}).get("capture") == "error" for record in records),
            "duration_s": round(duration, 3),
        },
    }
    write_json(output_dir / "summary.json", summary)

    columns = [
        "index", "op_id", "algorithm", "api", "rank", "shape", "batch", "scale", "direction",
        "capture", "flagfft_status", "platform_status", "flagfft_rel_l2", "flagfft_rel_linf",
        "platform_rel_l2", "platform_rel_linf", "limit_rel_l2", "limit_rel_linf",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            metric = record.get("metric", {})
            flagfft = metric.get("flagfft_vs_numpy", {})
            platform_result = metric.get("platform_vs_numpy", {})
            limits = metric.get("limits", {})
            writer.writerow({
                "index": index,
                "op_id": record.get("op_id", ""),
                "algorithm": record.get("algorithm", ""),
                "api": record.get("api", ""),
                "rank": record.get("rank", ""),
                "shape": "x".join(str(value) for value in record.get("shape", [])),
                "batch": record.get("batch", ""),
                "scale": record.get("scale", ""),
                "direction": record.get("direction", ""),
                "capture": record.get("status", {}).get("capture", ""),
                "flagfft_status": record.get("status", {}).get("flagfft", ""),
                "platform_status": record.get("status", {}).get("platform", ""),
                "flagfft_rel_l2": flagfft.get("rel_l2", ""),
                "flagfft_rel_linf": flagfft.get("rel_linf", ""),
                "platform_rel_l2": platform_result.get("rel_l2", ""),
                "platform_rel_linf": platform_result.get("rel_linf", ""),
                "limit_rel_l2": limits.get("rel_l2", ""),
                "limit_rel_linf": limits.get("rel_linf", ""),
            })
    return summary


def analyze_only(output_dir: Path) -> int:
    case_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir() and (path / "case.json").is_file())
    if not case_dirs:
        raise ValueError(f"no case directories found under {output_dir}")
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for case_dir in case_dirs:
        case = json.loads((case_dir / "case.json").read_text())
        records.append(analyze_case(case_dir, case))
    old_summary = {}
    summary_file = output_dir / "summary.json"
    if summary_file.is_file():
        old_summary = json.loads(summary_file.read_text())
    config = old_summary.get("config", {"mode": "analyze-only"})
    environment = old_summary.get("environment", environment_snapshot(None, source_root()))
    write_summary(output_dir, records, config, environment, time.monotonic() - started)
    return 0 if all(record.get("status", {}).get("both") for record in records) else 1


def run_experiment(args: argparse.Namespace) -> int:
    source = Path(args.source_dir).resolve()
    capture_bin = Path(args.capture_bin).resolve()
    if not capture_bin.is_file():
        raise ValueError(f"capture executable not found: {capture_bin}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        results_root() / (datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_numpy_fft_correctness")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ops_filter = {item.strip() for item in args.ops.split(",") if item.strip()} if args.ops else None
    cases = load_cases(
        source,
        args.combination,
        ops_filter,
        args.scales,
        parse_shape_filter(args.shapes),
    )
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("no test cases selected")

    environment = environment_snapshot(args.backend, source)
    config = {
        "source_dir": str(source),
        "capture_bin": str(capture_bin),
        "combination": args.combination,
        "ops": sorted(ops_filter) if ops_filter is not None else None,
        "scales": args.scales,
        "shapes": args.shapes,
        "gpu": str(args.gpu),
        "timeout_s": args.timeout,
        "case_count": len(cases),
    }
    write_json(output_dir / "run_config.json", {"config": config, "environment": environment})

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, base_case in enumerate(cases, 1):
        case_dir = output_dir / case_name(index, base_case)
        case_dir.mkdir(parents=True, exist_ok=True)
        value, seed = make_input(base_case["api"], tuple(base_case["shape"]), base_case["batch"], base_case["scale"])
        np.save(case_dir / "input.npy", value, allow_pickle=False)
        value.tofile(case_dir / "input.bin")
        case = dict(base_case)
        case.update({
            "seed": seed,
            "input_dtype": str(value.dtype),
            "input_shape": list(value.shape),
            "output_shape": list(output_shape(base_case["api"], tuple(base_case["shape"]), base_case["batch"])),
            "status": {"capture": "running"},
        })
        write_json(case_dir / "case.json", case)

        command = [
            str(capture_bin),
            f"--api={base_case['api']}",
            f"--shape={'x'.join(str(value) for value in base_case['shape'])}",
            f"--batch={base_case['batch']}",
            f"--direction={base_case['direction']}",
            f"--input={case_dir / 'input.bin'}",
            f"--output-dir={case_dir}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["MUSA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.backend and args.backend.upper() == "PPU":
            env.setdefault("PPU_VISIBLE_DEVICES", str(args.gpu))
        capture_started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            (case_dir / "capture.stdout").write_text(completed.stdout)
            (case_dir / "capture.stderr").write_text(completed.stderr)
            case["capture_duration_s"] = round(time.monotonic() - capture_started, 3)
            case["capture_command"] = command
            if completed.returncode != 0:
                case["status"] = {"capture": "error"}
                case["capture_returncode"] = completed.returncode
                case["error"] = completed.stderr[-4000:]
                write_json(case_dir / "case.json", case)
                records.append(case)
                if args.stop_on_error:
                    break
                continue
            case = analyze_case(case_dir, case)
        except subprocess.TimeoutExpired as error:
            case["status"] = {"capture": "timeout"}
            case["capture_duration_s"] = round(time.monotonic() - capture_started, 3)
            case["capture_command"] = command
            case["error"] = str(error)
            write_json(case_dir / "case.json", case)
            records.append(case)
            if args.stop_on_error:
                break
            continue
        except Exception as error:
            case["status"] = {"capture": "error"}
            case["capture_duration_s"] = round(time.monotonic() - capture_started, 3)
            case["capture_command"] = command
            case["error"] = repr(error)
            write_json(case_dir / "case.json", case)
            records.append(case)
            if args.stop_on_error:
                break
            continue

        records.append(case)
        flagfft_status = case["status"]["flagfft"]
        platform_status = case["status"]["platform"]
        print(f"[{index}/{len(cases)}] {case_dir.name}: FlagFFT={flagfft_status}, platform={platform_status}")

    summary = write_summary(output_dir, records, config, environment, time.monotonic() - started)
    print(json.dumps(summary["summary"], indent=2))
    print(f"Results: {output_dir}")
    return 0 if summary["summary"]["both_passed"] == summary["summary"]["total"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-bin", help="path to the native validation capture executable")
    parser.add_argument("--source-dir", default=str(source_root()))
    parser.add_argument("--backend", default=None, help="recorded backend name, e.g. CUDA, MUSA, or PPU")
    parser.add_argument("--combination", default="full", help="matrix combination, or full/all")
    parser.add_argument("--ops", default=None, help="comma-separated operator IDs")
    parser.add_argument("--shapes", default=None, help="comma-separated exact shapes, e.g. 256 or 64x64")
    parser.add_argument("--scales", default=None, help="comma-separated scales, or 'all'")
    parser.add_argument("--gpu", default="0", help="single device ID for this run")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--analyze-only", metavar="RESULT_DIR", default=None,
                        help="recompute NumPy metrics from an existing result directory")
    args = parser.parse_args()
    if args.analyze_only is None and not args.capture_bin:
        parser.error("--capture-bin is required unless --analyze-only is used")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.analyze_only is not None:
            return analyze_only(Path(args.analyze_only).resolve())
        return run_experiment(args)
    except Exception as error:
        print(f"numpy_fft_validation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
