#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified 36-operator acceptance runner: FlagFFT/platform FFT vs NumPy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import queue
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 3
ENV_INFO: dict[str, Any] = {}
WORKER_PROCESSES: list[multiprocessing.Process] = []
INTERRUPTED = False
GROUPS = (
    "1d_ct_single",
    "1d_prime_single",
    "1d_ct_batch",
    "1d_prime_batch",
    "2d",
    "3d",
)
# The acceptance manifest always keeps all 36 operator definitions.  Backend
# limitations are represented as policy skips in the manifest/summary instead
# of silently removing operators from the acceptance surface.
UNSUPPORTED_APIS_BY_BACKEND: dict[str, frozenset[str]] = {
    "ix": frozenset({"z2z", "z2d", "d2z"}),
}
BACKEND_SKIP_REASONS = {
    "ix": "IX/CoreX does not support FP64; Z2Z, Z2D and D2Z are excluded from execution.",
}
DIRECTIONS = {
    "c2c": ("forward", "inverse"),
    "c2r": ("inverse",),
    "r2c": ("forward",),
    "z2z": ("forward", "inverse"),
    "z2d": ("inverse",),
    "d2z": ("forward",),
}
RED = GREEN = YELLOW = NC = ""


def init_colors(mode: str) -> None:
    global RED, GREEN, YELLOW, NC
    if mode == "always" or (mode == "auto" and sys.stderr.isatty()):
        RED, GREEN, YELLOW, NC = "\033[31m", "\033[32m", "\033[33m", "\033[0m"
    else:
        RED = GREEN = YELLOW = NC = ""


def pinfo(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True)


def pwarn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}", file=sys.stderr, flush=True)


def perror(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr, flush=True)


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
ARTIFACT_POLICIES = ("none", "failed", "all")
# These files are the native capture boundary. The .npy files are included
# only so old/interrupted result directories can be cleaned consistently;
# new runs never create them.
RAW_ARTIFACT_FILENAMES = (
    "input.bin",
    "flagfft.bin",
    "platform.bin",
    "input.npy",
    "numpy.npy",
)

# Accuracy runs can contain hundreds of millions of values.  Keep the NumPy
# oracle and the error reduction bounded even when a single batch is very
# large; the native capture processes already have their own peak memory.
REFERENCE_BATCH_CHUNK = 4
ERROR_STATS_CHUNK_ELEMENTS = 1 << 18


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


def accuracy_seed(
    api: str, transform_elements: int, batch: int, variant: int = 0
) -> int:
    return (
        SEED_TAG
        ^ ((TYPE_CODES[api] << 48) & MASK64)
        ^ ((int(transform_elements) << 16) & MASK64)
        ^ int(batch)
        ^ ((int(variant) * SPLITMIX_INCREMENT) & MASK64)
    ) & MASK64


def as_complex_from_interleaved(
    values: np.ndarray, api: str, shape: tuple[int, ...]
) -> np.ndarray:
    dtype = complex_dtype(api)
    scalar = values.astype(real_dtype(api), copy=False)
    return scalar.view(dtype).reshape(shape)


def make_input(
    api: str, shape: tuple[int, ...], batch: int, scale: float
) -> tuple[np.ndarray, int]:
    """Generate deterministic native-dtype input, including valid real half spectra."""
    seed = accuracy_seed(api, product(shape), batch)
    target_shape = input_shape(api, shape, batch)
    count = product(target_shape)
    if is_complex(api) or is_real_inverse(api):
        result = as_complex_from_interleaved(
            splitmix_signed_unit(count * 2, seed), api, target_shape
        )
        if is_real_inverse(api):
            # DC/Nyquist planes must be Hermitian across every preceding FFT
            # axis. Merely zeroing their imaginary parts is only valid in 1D.
            boundaries = [0] + ([shape[-1] // 2] if shape[-1] % 2 == 0 else [])
            for boundary in boundaries:
                plane = result[..., boundary]
                mirrored = plane
                for axis, length in enumerate(shape[:-1], 1):
                    mirrored = np.take(
                        mirrored, (-np.arange(length)) % length, axis=axis
                    )
                result[..., boundary] = (plane + mirrored.conj()) * 0.5
    else:
        result = (
            splitmix_signed_unit(count, seed)
            .astype(real_dtype(api))
            .reshape(target_shape)
        )
    scale_value = np.asarray(scale, dtype=real_dtype(api)).item()
    result = result * scale_value
    return np.ascontiguousarray(result), seed


def numpy_reference(
    value: np.ndarray, api: str, shape: tuple[int, ...], direction: str
) -> np.ndarray:
    # NumPy 2.x can keep float32 FFTs in single precision. Explicitly use
    # float64/complex128 so the oracle does not inherit device rounding.
    value = np.asarray(
        value, dtype=np.complex128 if np.iscomplexobj(value) else np.float64
    )
    axes = tuple(range(1, len(shape) + 1))
    transform_size = product(shape)
    output_dtype = np.complex128 if is_complex(api) or is_real_forward(api) else np.float64
    result = np.empty(output_shape(api, shape, value.shape[0]), dtype=output_dtype)
    # Compute each small group of batches independently.  Calling NumPy FFT
    # over all 256 batches at once can allocate a large internal workspace in
    # addition to the input and output arrays under the 8 GiB MUSA cgroup.
    for start in range(0, value.shape[0], REFERENCE_BATCH_CHUNK):
        stop = min(value.shape[0], start + REFERENCE_BATCH_CHUNK)
        chunk = value[start:stop]
        if is_complex(api):
            if direction == "forward":
                transformed = np.fft.fftn(chunk, s=shape, axes=axes)
            else:
                transformed = np.fft.ifftn(chunk, s=shape, axes=axes)
                transformed *= transform_size
        elif is_real_forward(api):
            transformed = np.fft.rfftn(chunk, s=shape, axes=axes)
        elif is_real_inverse(api):
            # The device APIs intentionally use the unnormalized inverse.
            transformed = np.fft.irfftn(chunk, s=shape, axes=axes)
            transformed *= transform_size
        else:
            raise ValueError(f"unknown API: {api}")
        result[start:stop] = transformed
    return result


def _component_arrays(
    value: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
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


def error_stats(
    value: np.ndarray, reference: np.ndarray, elements_per_batch: int, batch: int
) -> dict[str, Any]:
    rel_l2 = np.longdouble(0.0)
    rel_linf = np.longdouble(0.0)
    max_abs = np.longdouble(0.0)
    mixed_pointwise = np.longdouble(0.0)
    worst_l2_batch = 0
    worst_linf_batch = 0

    value = np.asarray(value).reshape(batch, elements_per_batch)
    reference = np.asarray(reference).reshape(batch, elements_per_batch)
    for batch_index in range(batch):
        err_sq = np.longdouble(0.0)
        ref_sq = np.longdouble(0.0)
        err_max = np.longdouble(0.0)
        ref_max = np.longdouble(0.0)
        mixed_max = np.longdouble(0.0)
        for start in range(0, elements_per_batch, ERROR_STATS_CHUNK_ELEMENTS):
            stop = min(elements_per_batch, start + ERROR_STATS_CHUNK_ELEMENTS)
            batch_diff, batch_ref, _ = _component_arrays(
                value[batch_index, start:stop],
                reference[batch_index, start:stop],
            )
            finite = bool(
                np.all(np.isfinite(batch_diff))
                and np.all(np.isfinite(batch_ref))
            )
            if not finite:
                return {
                    "rel_l2": float("inf"),
                    "rel_linf": float("inf"),
                    "max_abs": float("inf"),
                    "mixed_pointwise": float("inf"),
                    "worst_l2_batch": batch_index,
                    "worst_linf_batch": batch_index,
                    "finite": False,
                }
            err_sq += np.sum(batch_diff * batch_diff, dtype=np.longdouble)
            ref_sq += np.sum(batch_ref * batch_ref, dtype=np.longdouble)
            err_max = max(err_max, np.max(batch_diff, initial=np.longdouble(0.0)))
            ref_max = max(ref_max, np.max(batch_ref, initial=np.longdouble(0.0)))
            mixed_max = max(
                mixed_max,
                np.max(
                    batch_diff / np.maximum(batch_ref, np.longdouble(1.0)),
                    initial=np.longdouble(0.0),
                ),
            )

        batch_rel_l2 = (
            np.sqrt(err_sq / ref_sq)
            if ref_sq != 0
            else (np.longdouble(0.0) if err_sq == 0 else np.longdouble(np.inf))
        )
        batch_rel_linf = (
            err_max / ref_max
            if ref_max != 0
            else (np.longdouble(0.0) if err_max == 0 else np.longdouble(np.inf))
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


def cleanup_raw_artifacts(case_dir: Path) -> list[str]:
    """Remove only known numerical artifacts, leaving reports and logs intact."""
    errors = []
    for filename in RAW_ARTIFACT_FILENAMES:
        path = case_dir / filename
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(f"{filename}: {error}")
    return errors


def retain_raw_artifacts(policy: str, record: dict[str, Any]) -> bool:
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "failed":
        return any(
            record.get(field, {}).get("status") != "Passed"
            for field in ("accuracy", "platform_accuracy")
        )
    raise ValueError(
        f"unknown artifact policy: {policy}; expected one of {', '.join(ARTIFACT_POLICIES)}"
    )


def load_raw(path: Path, api: str, shape: tuple[int, ...], batch: int) -> np.ndarray:
    is_input = path.name == "input.bin"
    complex_values = (
        (is_complex(api) or is_real_inverse(api))
        if is_input
        else (is_complex(api) or is_real_forward(api))
    )
    dtype = complex_dtype(api) if complex_values else real_dtype(api)
    expected_shape = (
        input_shape(api, shape, batch) if is_input else output_shape(api, shape, batch)
    )
    data = np.fromfile(path, dtype=dtype)
    expected = product(expected_shape)
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} {dtype} values, got {data.size}")
    return data.reshape(expected_shape)


def json_safe(value: Any) -> Any:
    """Emit strict JSON even when failed numeric comparisons contain NaN/Inf."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_operators(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    ops = data.get("ops", [])
    seen = set()
    for op in ops:
        op_id = op["id"]
        if op_id in seen:
            raise ValueError(f"duplicate operator ID: {op_id}")
        seen.add(op_id)
        if (
            op.get("api") not in DIRECTIONS
            or type(op.get("rank")) is not int
            or op["rank"] not in (1, 2, 3)
        ):
            raise ValueError(f"{op_id}: invalid api or rank")
        if op["rank"] == 1:
            if op.get("algorithm") not in ("ct", "prime") or op.get("batch") not in (
                "single",
                "batch",
            ):
                raise ValueError(
                    f"{op_id}: 1D requires algorithm ct/prime and batch single/batch"
                )
            expected = f"1d_{op['algorithm']}_{op['batch']}_{op['api']}"
        else:
            expected = f"{op['rank']}d_{op['api']}"
        if op_id != expected or not isinstance(op.get("sizes"), str):
            raise ValueError(
                f"{op_id}: expected ID {expected} and a size-set reference"
            )
    expected_ids = {f"{group}_{api}" for group in GROUPS for api in DIRECTIONS}
    if seen != expected_ids:
        missing = ", ".join(sorted(expected_ids - seen))
        raise ValueError(
            f"operators.yaml must define all 36 acceptance operators; missing: {missing}"
        )
    return ops


def parse_scales(raw: str | None, matrix_scales: list[float]) -> list[float]:
    if raw is None:
        values = matrix_scales
    elif raw.strip().lower() == "all":
        values = DEFAULT_SCALES
    else:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("scales must be finite positive numbers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate scales")
    return values


def load_test_matrix(path: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(path.read_text())
    for mode, values in matrix.get("batches", {}).items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"batches.{mode} must be a nonempty list")
        if any(type(value) is not int or value <= 0 for value in values) or len(
            set(values)
        ) != len(values):
            raise ValueError(f"batches.{mode} must contain distinct positive integers")
        if mode in ("single", "3d") and values != [1]:
            raise ValueError(f"batches.{mode} must be [1]")
        if mode == "batch" and any(value <= 1 for value in values):
            raise ValueError("batches.batch values must be greater than 1")
    parse_scales(None, matrix.get("scales", [1.0]))
    return matrix


def operator_group(op: dict) -> str:
    if op["rank"] == 1:
        return f"1d_{op['algorithm']}_{op['batch']}"
    return f"{op['rank']}d"


def resolve_combination_names(value: str, matrix: dict | None = None) -> list[str]:
    """Group filters are derived from operators, not a second operator matrix."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("no combination specified")
    if any(name in ("full", "all") for name in names):
        if len(names) != 1:
            raise ValueError("'full'/'all' must be used alone")
        return list(GROUPS)
    aliases = {"1d_bs_single": "1d_prime_single", "1d_bs_batch": "1d_prime_batch"}
    names = [aliases.get(name, name) for name in names]
    unknown = set(names) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown combinations: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(names))


def parse_shape_filter(raw: str | None) -> set[tuple[int, ...]] | None:
    if raw is None:
        return None
    shapes = set()
    for part in raw.split(","):
        dimensions = tuple(int(value) for value in part.strip().lower().split("x"))
        if len(dimensions) not in (1, 2, 3) or any(value <= 0 for value in dimensions):
            raise ValueError("shapes must be positive dimensions, e.g. 256,64x64")
        shapes.add(dimensions)
    return shapes


def case_name(case: dict, performance: bool = False) -> str:
    shape = "x".join(str(value) for value in case["shape"])
    name = f"{case['op_id']}__n{shape}__b{case['batch']}__{case['direction']}"
    if not performance:
        scale = (
            f"{case['scale']:.17g}".replace("-", "m")
            .replace(".", "p")
            .replace("+", "p")
        )
        name += f"__s{scale}"
    return name


def expand_test_cases(
    ops: list[dict],
    matrix: dict,
    combination: str = "full",
    scales: str | None = None,
    shapes: set[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    groups = set(resolve_combination_names(combination))
    scale_values = parse_scales(scales, matrix.get("scales", [1.0]))
    cases = []
    seen = set()
    for op in ops:
        group = operator_group(op)
        if group not in groups:
            continue
        sizes = matrix.get(op["sizes"])
        if not isinstance(sizes, list) or not sizes:
            raise ValueError(f"{op['id']}: missing or empty size set {op['sizes']}")
        batch_key = op["batch"] if op["rank"] == 1 else f"{op['rank']}d"
        batches = matrix.get("batches", {}).get(batch_key)
        if not batches:
            raise ValueError(f"{op['id']}: missing batches.{batch_key}")
        for size in sizes:
            shape = size if isinstance(size, list) else [size]
            if len(shape) != op["rank"] or any(
                type(n) is not int or n <= 0 for n in shape
            ):
                raise ValueError(f"{op['id']}: invalid rank-{op['rank']} shape {shape}")
            if shapes is not None and tuple(shape) not in shapes:
                continue
            for batch in batches:
                for scale in scale_values:
                    for direction in DIRECTIONS[op["api"]]:
                        case = {
                            "op_id": op["id"],
                            "api": op["api"],
                            "rank": op["rank"],
                            "algorithm": op.get("algorithm", f"{op['rank']}d"),
                            "batch_mode": op.get("batch"),
                            "shape": list(shape),
                            "batch": batch,
                            "scale": scale,
                            "direction": direction,
                        }
                        case["case_id"] = case_name(case)
                        if case["case_id"] in seen:
                            raise ValueError(f"duplicate case: {case['case_id']}")
                        seen.add(case["case_id"])
                        cases.append(case)
    return cases


def expand_all_test_cases(ops: list[dict], matrix: dict) -> list[dict[str, Any]]:
    return expand_test_cases(ops, matrix)


def performance_cases(cases: list[dict]) -> list[dict]:
    unique = {}
    for case in cases:
        key = case_name(case, performance=True)
        if key not in unique:
            perf_case = {key: value for key, value in case.items() if key != "scale"}
            perf_case["case_id"] = key
            unique[key] = perf_case
    return list(unique.values())


def build_accuracy_cmd(
    case: dict, capture_bin: Path, case_dir: Path, implementation: str
) -> list[str]:
    return [
        str(capture_bin),
        f"--api={case['api']}",
        f"--shape={'x'.join(str(value) for value in case['shape'])}",
        f"--batch={case['batch']}",
        f"--direction={case['direction']}",
        f"--input={case_dir / 'input.bin'}",
        f"--output-dir={case_dir}",
        f"--implementation={implementation}",
    ]


def build_perf_cmd(case: dict, build_dir: Path, warmup: int, iters: int) -> list[str]:
    return [
        str(build_dir / "flagfft-cli"),
        "bench",
        "--api",
        case["api"],
        "--rank",
        str(case["rank"]),
        "--shape",
        "x".join(str(n) for n in case["shape"]),
        "--batch",
        str(case["batch"]),
        "--direction",
        case["direction"],
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--json",
        "--print-path",
    ]


def detect_backend(build_dir: Path) -> str:
    """Detect the backend from CMake, then from the installed Triton plugin."""
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        try:
            for line in cache.read_text(errors="replace").splitlines():
                if line.startswith("BACKEND:"):
                    value = line.split("=", 1)[1].strip().strip('"')
                    if value:
                        return value.lower()
        except OSError:
            pass

    try:
        from triton._C import libtriton

        if hasattr(libtriton, "ppu"):
            return "ppu"
        if hasattr(libtriton, "mthreads"):
            return "musa"
        if hasattr(libtriton, "iluvatar"):
            return "ix"
        if hasattr(libtriton, "cuda"):
            return "cuda"
    except ImportError:
        pass
    return "unknown"


def operator_skip_reason(op: dict, backend: str) -> str | None:
    if op.get("api") not in UNSUPPORTED_APIS_BY_BACKEND.get(backend, frozenset()):
        return None
    return BACKEND_SKIP_REASONS.get(
        backend, f"backend {backend.upper()} does not support API {op['api'].upper()}."
    )


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
                commands.append(
                    [
                        "git",
                        "--git-dir",
                        str(container_workspace / relative),
                        "rev-parse",
                        "HEAD",
                    ]
                )
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


def probe_env(build_dir: Path) -> None:
    ENV_INFO.update(
        {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": git_commit(ROOT),
            "backend": detect_backend(build_dir),
        }
    )
    ENV_INFO["reference_library"] = {
        "cuda": "cuFFT",
        "musa": "muFFT",
        "ppu": "PPU cuFFT-compatible FFT",
        "ix": "ixfft (CoreX cuFFT-compatible FFT)",
    }.get(ENV_INFO["backend"], "unknown")
    try:
        import torch

        ENV_INFO["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
            ),
        }
    except ImportError:
        ENV_INFO["torch"] = {"version": "N/A", "device_count": 0}
    try:
        import triton

        ENV_INFO["triton"] = {"version": triton.__version__}
    except ImportError:
        ENV_INFO["triton"] = {"version": "N/A"}


def run_subprocess(
    cmd: list[str], timeout: int, gpu_id: int, case_dir: Path, stage: str
) -> dict:
    """Stream logs to disk and kill the complete subprocess group on timeout."""
    env = os.environ.copy()
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "MUSA_VISIBLE_DEVICES",
        "PPU_VISIBLE_DEVICES",
        "IX_VISIBLE_DEVICES",
    ):
        env[variable] = str(gpu_id)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.monotonic()
    result = {
        "command": cmd,
        "stdout_file": f"{stage}.stdout",
        "stderr_file": f"{stage}.stderr",
    }
    process = None
    with (
        (case_dir / result["stdout_file"]).open("w") as stdout,
        (case_dir / result["stderr_file"]).open("w") as stderr,
    ):
        try:
            process = subprocess.Popen(
                cmd,
                cwd=case_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            returncode = process.wait(timeout=timeout)
            result.update(
                {
                    "status": (
                        "Completed"
                        if returncode == 0
                        else ("Skipped" if returncode == 77 else "Error")
                    ),
                    "returncode": returncode,
                }
            )
            if returncode != 0:
                result["error"] = f"process exited with code {returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result.update(
                {"status": "Timeout", "error": f"exceeded {timeout}s timeout"}
            )
        except OSError as error:
            result.update({"status": "Error", "error": str(error)})
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    result["duration"] = time.monotonic() - started
    if result["status"] == "Error":
        with (case_dir / result["stderr_file"]).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4000))
            detail = stream.read().decode("utf-8", errors="replace").strip()
        if detail:
            result["error"] += f": {detail}"
    return result


def pending_accuracy() -> dict:
    return {"status": "NotFound", "reference": "numpy.fft", "plan": None}


def compare_output(
    case: dict,
    case_dir: Path,
    implementation: str,
    reference: np.ndarray | None,
    stage: dict,
) -> dict:
    limits = accuracy_limit(case["api"], product(case["shape"]))
    result = {
        "status": stage.get("status", "Error"),
        "reference": "numpy.fft",
        "limits": limits,
        "capture": stage,
        "duration": stage.get("duration", 0),
        "plan": None,
    }
    if implementation == "flagfft":
        plan_file = case_dir / "flagfft_plan.txt"
        if plan_file.is_file():
            result["plan"] = plan_file.read_text(errors="replace")
    if result["status"] != "Completed":
        result["error"] = stage.get("error", result["status"])
        return result
    try:
        output = load_raw(
            case_dir / f"{implementation}.bin",
            case["api"],
            tuple(case["shape"]),
            case["batch"],
        )
        elements = product(
            output_shape(case["api"], tuple(case["shape"]), case["batch"])[1:]
        )
        result["metric"] = judged_stats(
            error_stats(output, reference, elements, case["batch"]), limits
        )
        result["status"] = "Passed" if result["metric"]["passed"] else "Failed"
        result["output_sha256"] = sha256(case_dir / f"{implementation}.bin")
    except (OSError, ValueError) as error:
        result.update({"status": "Error", "error": str(error)})
    return result


def run_accuracy_case(
    case: dict,
    capture_bin: Path,
    output_dir: Path,
    gpu_id: int,
    timeout: int,
    artifact_policy: str = "failed",
) -> dict:
    if artifact_policy not in ARTIFACT_POLICIES:
        raise ValueError(
            f"unknown artifact policy: {artifact_policy}; expected one of {', '.join(ARTIFACT_POLICIES)}"
        )
    case_dir = output_dir / case["op_id"] / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": FORMAT_VERSION,
        **case,
        "artifact_policy": artifact_policy,
        # Keep partial data available until the case reaches its final state,
        # unless the user explicitly selected the strict no-artifact mode.
        "raw_artifacts_retained": artifact_policy != "none",
        "accuracy": pending_accuracy(),
        "platform_accuracy": pending_accuracy(),
    }
    data_file = (case_dir / "case.json").relative_to(output_dir).as_posix()
    record["data_file"] = data_file
    write_json(case_dir / "case.json", record)
    started = time.monotonic()
    try:
        value, seed = make_input(
            case["api"], tuple(case["shape"]), case["batch"], case["scale"]
        )
        value.tofile(case_dir / "input.bin")
        record.update(
            {
                "seed": seed,
                "input_dtype": str(value.dtype),
                "input_sha256": sha256(case_dir / "input.bin"),
            }
        )
        # The native capture is a separate process.  Drop the generated input
        # before starting it so the worker does not retain a full-size array
        # alongside the capture process and its output buffers.
        del value

        capture_stages = {}
        for implementation, field in (
            ("flagfft", "accuracy"),
            ("platform", "platform_accuracy"),
        ):
            command = build_accuracy_cmd(case, capture_bin, case_dir, implementation)
            stage = run_subprocess(command, timeout, gpu_id, case_dir, implementation)
            capture_stages[implementation] = stage
            record["capture_stages"] = capture_stages
            # Persist each native stage before starting the next one.  This is
            # also useful when a long-running worker is interrupted.
            write_json(case_dir / "case.json", record)

        # A failed or timed-out native capture does not need a NumPy reference.
        # Record those stage results first, then only compare completed output.
        for implementation, field in (
            ("flagfft", "accuracy"),
            ("platform", "platform_accuracy"),
        ):
            stage = capture_stages[implementation]
            if stage.get("status") != "Completed":
                record[field] = compare_output(
                    case, case_dir, implementation, None, stage
                )
                record[field]["data_file"] = data_file

        if any(
            stage.get("status") == "Completed" for stage in capture_stages.values()
        ):
            # Both native processes have exited before NumPy allocates its
            # double-precision reference.  Reloading from disk avoids keeping
            # the original input alive through the capture phase.
            reference_input = load_raw(
                case_dir / "input.bin",
                case["api"],
                tuple(case["shape"]),
                case["batch"],
            )
            reference_value = np.asarray(
                reference_input,
                dtype=np.complex128 if np.iscomplexobj(reference_input) else np.float64,
            )
            del reference_input
            try:
                reference = numpy_reference(
                    reference_value,
                    case["api"],
                    tuple(case["shape"]),
                    case["direction"],
                )
            finally:
                del reference_value
            record["numpy_dtype"] = str(reference.dtype)

            for implementation, field in (
                ("flagfft", "accuracy"),
                ("platform", "platform_accuracy"),
            ):
                stage = capture_stages[implementation]
                if stage.get("status") == "Completed":
                    record[field] = compare_output(
                        case, case_dir, implementation, reference, stage
                    )
                    record[field]["data_file"] = data_file

            # Release the large reference before artifact cleanup and before
            # the worker starts the next case.
            reference = None
    except Exception as error:
        for field in ("accuracy", "platform_accuracy"):
            if record[field]["status"] == "NotFound":
                record[field] = {"status": "Error", "error": repr(error), "plan": None}
    finally:
        record["duration"] = time.monotonic() - started
        keep = retain_raw_artifacts(artifact_policy, record)
        cleanup_errors = [] if keep else cleanup_raw_artifacts(case_dir)
        record["raw_artifacts_retained"] = keep or bool(cleanup_errors)
        if cleanup_errors:
            record["artifact_cleanup_errors"] = cleanup_errors
        write_json(case_dir / "case.json", record)
    return record


def parse_perf_result(output: str, case: dict | None = None) -> dict:
    decoder = json.JSONDecoder()
    data = None
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "cases" in candidate:
            data = candidate
            break
    if not data or not data.get("cases"):
        return {
            "status": "Error",
            "error": "no benchmark cases in output",
            "plan": None,
        }
    raw_case = data["cases"][0]
    timing = raw_case.get("timing", {})
    ff = timing.get("flagfft_median_ms", 0)
    ref = timing.get("ref_median_ms", 0)
    speedup = timing.get("speedup", 0)
    valid = all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in (ff, ref, speedup)
    )
    return {
        "status": "Passed" if valid else "Failed",
        "flagfft_median_ms": ff,
        "ref_median_ms": ref,
        "speedup": speedup,
        "plan": raw_case.get("plan_description"),
    }


def run_performance_case(
    case: dict,
    build_dir: Path,
    output_dir: Path,
    gpu_id: int,
    timeout: int,
    warmup: int,
    iters: int,
    baseline_valid: bool | None,
) -> dict:
    case_dir = output_dir / case["op_id"] / "performance" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    command = build_perf_cmd(case, build_dir, warmup, iters)
    stage = run_subprocess(command, timeout, gpu_id, case_dir, "bench")
    if stage["status"] == "Completed":
        result = parse_perf_result(
            (case_dir / "bench.stdout").read_text(errors="replace"), case
        )
    else:
        result = {"status": stage["status"], "error": stage.get("error"), "plan": None}
    result.update(
        {
            "duration": stage["duration"],
            "capture": stage,
            "baseline_valid": baseline_valid,
            "data_file": (case_dir / "result.json").relative_to(output_dir).as_posix(),
        }
    )
    record = {"format_version": FORMAT_VERSION, **case, "performance": result}
    write_json(case_dir / "result.json", record)
    return record


def worker_proc(
    gpu_id: int,
    work_queue,
    display_queue,
    capture_bin: Path,
    build_dir: Path,
    output_dir: Path,
    args,
) -> None:
    def stop_worker(signum, frame):
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    while True:
        job = work_queue.get()
        if job is None:
            break
        baseline_valid = None if args.performance_only else True
        for case in job["accuracy_cases"] if not args.performance_only else []:
            record = run_accuracy_case(
                case,
                capture_bin,
                output_dir,
                gpu_id,
                args.timeout,
                args.artifact_policy,
            )
            baseline_valid = (
                baseline_valid and record["platform_accuracy"]["status"] == "Passed"
            )
            display_queue.put(
                {
                    **case,
                    "gpu": gpu_id,
                    "phase": "accuracy",
                    "status": record["accuracy"]["status"],
                    "duration": record["duration"],
                    "result": record["accuracy"],
                    "platform_result": record["platform_accuracy"],
                }
            )
        if not args.accuracy_only:
            case = job["performance_case"]
            try:
                record = run_performance_case(
                    case,
                    build_dir,
                    output_dir,
                    gpu_id,
                    args.timeout,
                    args.warmup,
                    args.iters,
                    baseline_valid,
                )
                result = record["performance"]
            except Exception as error:
                result = {"status": "Error", "error": repr(error), "plan": None}
            display_queue.put(
                {
                    **case,
                    "gpu": gpu_id,
                    "phase": "performance",
                    "status": result["status"],
                    "duration": result.get("duration", 0),
                    "result": result,
                }
            )


def policy_skip_message(case: dict, phase: str) -> dict:
    """Create a report/CSV message for a case excluded by backend policy."""
    reason = case.get("skip_reason", "unsupported by backend policy")
    result = {
        "status": "Skipped",
        "reference": (
            ENV_INFO.get("reference_library")
            if phase == "performance"
            else "numpy.fft"
        ),
        "plan": None,
        "skip_reason": reason,
        "error": reason,
    }
    message = {
        **case,
        "gpu": None,
        "phase": phase,
        "status": "Skipped",
        "duration": 0.0,
        "result": result,
    }
    if phase == "accuracy":
        message["platform_result"] = {
            "status": "Skipped",
            "reference": "numpy.fft",
            "plan": None,
            "skip_reason": reason,
            "error": reason,
        }
    return message


def aggregate_results(
    raw_results: list[dict],
    ops: list[dict],
    cases: list[dict] | None = None,
    run_accuracy: bool = True,
    run_performance: bool = True,
    skipped_op_ids: set[str] | None = None,
    skip_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Judge all expected cases; a missing worker result cannot become Passed."""
    op_results = {}
    skipped_op_ids = skipped_op_ids or set()
    skip_reasons = skip_reasons or {}
    phases = ("accuracy", "platform_accuracy", "performance")
    for op in ops:
        op_results[op["id"]] = {}
        for phase in phases:
            enabled = run_performance if phase == "performance" else run_accuracy
            policy_skipped = op["id"] in skipped_op_ids and enabled
            skip_reason = skip_reasons.get(
                op["id"], "unsupported by backend policy"
            )
            expected = (
                performance_cases(cases or [])
                if phase == "performance"
                else cases or []
            )
            entries = {
                case["case_id"]: {**case, "status": "NotFound", "plan": None}
                for case in expected
                if case["op_id"] == op["id"]
            }
            op_results[op["id"]][phase] = {
                "status": (
                    "Skipped"
                    if policy_skipped
                    else ("NotFound" if enabled else "NotRun")
                ),
                "reference": (
                    ENV_INFO.get("reference_library")
                    if phase == "performance"
                    else "numpy.fft"
                ),
                "total": len(entries),
                "completed": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "missing": len(entries),
                "duration": 0,
                "data_file": f"{op['id']}/{phase}_result.json",
                "cases": entries if enabled else {},
                **({"policy_skipped": True, "skip_reason": skip_reason} if policy_skipped else {}),
                **({"data": {}} if phase == "performance" else {"details": []}),
            }
            if policy_skipped:
                for entry in op_results[op["id"]][phase]["cases"].values():
                    entry.update(
                        {
                            "status": "Skipped",
                            "skip_reason": skip_reason,
                            "error": skip_reason,
                            "plan": None,
                        }
                    )
    for message in raw_results:
        if message.get("op_id") not in op_results:
            continue
        phase = message["phase"]
        fields = [
            ("accuracy", message["result"]),
            ("platform_accuracy", message.get("platform_result", {})),
        ]
        if phase == "performance":
            fields = [("performance", message["result"])]
        for field, result in fields:
            if not result:
                continue
            block = op_results[message["op_id"]][field]
            key = message["case_id"]
            meta = {
                key: message[key]
                for key in (
                    "case_id",
                    "op_id",
                    "api",
                    "rank",
                    "algorithm",
                    "batch_mode",
                    "shape",
                    "batch",
                    "direction",
                    "scale",
                    "skip_reason",
                )
                if key in message
            }
            block["cases"][key] = {**meta, **result}
    for op_result in op_results.values():
        for block in op_result.values():
            if block["status"] == "NotRun":
                block.update({"total": 0, "missing": 0})
                continue
            entries = list(block["cases"].values())
            block["total"] = len(entries)
            block["passed"] = sum(entry["status"] == "Passed" for entry in entries)
            block["failed"] = sum(
                entry["status"] in ("Failed", "Error", "Timeout") for entry in entries
            )
            block["skipped"] = sum(entry["status"] == "Skipped" for entry in entries)
            block["missing"] = sum(entry["status"] == "NotFound" for entry in entries)
            block["completed"] = block["total"] - block["missing"]
            block["duration"] = sum(entry.get("duration", 0) for entry in entries)
            if block["failed"]:
                block["status"] = "Failed"
            elif block["missing"]:
                block["status"] = "Incomplete"
            elif block["total"] and block["passed"] == block["total"]:
                block["status"] = "Passed"
            elif block["skipped"]:
                block["status"] = "Skipped"
        # Preserve the report consumer's existing accuracy.details and
        # performance.data structure alongside the richer per-case records.
        for phase in ("accuracy", "platform_accuracy"):
            block = op_result[phase]
            block["details"] = [
                {
                    "case": entry["case_id"],
                    "status": entry["status"],
                    "message": entry.get("error", "NumPy comparison failed"),
                    "metric": entry.get("metric", {}),
                    "limits": entry.get("limits", {}),
                }
                for entry in block["cases"].values()
                if entry["status"] in ("Failed", "Error", "Timeout")
            ]
        perf = op_result["performance"]
        details = {}
        speeds = []
        for entry in perf["cases"].values():
            if "speedup" not in entry:
                continue
            key = "[" + ",".join(str(n) for n in entry.get("shape", [])) + "]"
            if entry.get("batch", 1) > 1:
                key += f"batch={entry['batch']}"
            key += f"direction={entry.get('direction', 'unknown')}"
            details[key] = {
                "base": entry.get("ref_median_ms", 0),
                "gems": entry.get("flagfft_median_ms", 0),
                "speedup": entry["speedup"],
            }
            if entry["status"] == "Passed" and entry.get("baseline_valid") is not False:
                speeds.append(entry["speedup"])
        if details:
            perf["data"] = {
                "default": {
                    "result": "OK" if perf["status"] == "Passed" else "FAIL",
                    "details": details,
                    "speedup": math.exp(sum(math.log(s) for s in speeds) / len(speeds))
                    if speeds
                    else 0,
                }
            }
    return op_results


def compute_speedup_stats(op_results: dict) -> dict:
    values = [
        case["speedup"]
        for op in op_results.values()
        if op.get("accuracy", {}).get("status", "NotRun") in ("Passed", "NotRun")
        for case in op["performance"]["cases"].values()
        if case.get("status") == "Passed" and case.get("baseline_valid") is not False
    ]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "geometric_mean_speedup": round(
            math.exp(sum(math.log(value) for value in values) / len(values)), 4
        ),
        "min_speedup": round(min(values), 4),
        "max_speedup": round(max(values), 4),
    }


def write_summary(
    output_path: Path, op_results: dict, config: dict, total_duration: float
) -> dict:
    counts = {}
    for phase in ("accuracy", "platform_accuracy", "performance"):
        for status in (
            "Passed",
            "Failed",
            "Incomplete",
            "Skipped",
            "NotRun",
            "NotFound",
        ):
            counts[f"{phase}_{status.lower()}"] = sum(
                op[phase]["status"] == status for op in op_results.values()
            )
    summary = {
        "format_version": FORMAT_VERSION,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "env": ENV_INFO,
        "reference": {
            "implementation": "numpy.fft",
            "dtype_policy": "float64/complex128 reference; device inputs and outputs retain native dtype",
            "normalization": "device inverse is unnormalized; NumPy inverse multiplied by product(shape)",
            "metric": "ctest/flagfft_test.h::error_stats (worst batch rel_l2 and rel_linf)",
            "constants": ACCURACY_CONSTANTS,
        },
        "config": config,
        "result": op_results,
        "summary": {
            "total_ops": len(op_results),
            **counts,
            "total_duration": round(total_duration, 3),
            "speedup_stats": compute_speedup_stats(op_results),
        },
    }
    for op_id, result in op_results.items():
        op_dir = output_path.parent / op_id
        op_dir.mkdir(parents=True, exist_ok=True)
        for phase, details in result.items():
            write_json(op_dir / f"{phase}_result.json", details)
    write_json(output_path, summary)
    pinfo(f"Summary written to {output_path}")
    return summary


INC_COLUMNS = [
    "format_version",
    "phase",
    "case_id",
    "op_id",
    "api",
    "rank",
    "algorithm",
    "batch_mode",
    "shape",
    "batch",
    "direction",
    "scale",
    "status",
    "skip_reason",
    "flagfft_status",
    "platform_status",
    "flagfft_rel_l2",
    "flagfft_rel_linf",
    "platform_rel_l2",
    "platform_rel_linf",
    "limit_rel_l2",
    "limit_rel_linf",
    "flagfft_median_ms",
    "ref_median_ms",
    "speedup",
    "baseline_valid",
    "plan",
    "data_file",
    "duration_s",
    "error",
    "platform_error",
    "backend",
    "written_at",
]


def incremental_row(message: dict) -> dict:
    result = message.get("result", {})
    platform_result = message.get("platform_result", {})
    flag_metric = result.get("metric", {})
    platform_metric = platform_result.get("metric", {})
    limits = result.get("limits", {})
    row = {
        "format_version": FORMAT_VERSION,
        **{
            key: message.get(key, "")
            for key in (
                "phase",
                "case_id",
                "op_id",
                "api",
                "rank",
                "algorithm",
                "batch_mode",
                "batch",
                "direction",
                "scale",
            )
        },
        "shape": "x".join(str(n) for n in message.get("shape", [])),
        "status": result.get("status", "Error"),
        "skip_reason": result.get("skip_reason", message.get("skip_reason", "")),
        "flagfft_status": (
            result.get("status", "") if message["phase"] == "accuracy" else ""
        ),
        "platform_status": platform_result.get("status", ""),
        "flagfft_rel_l2": flag_metric.get("rel_l2", ""),
        "flagfft_rel_linf": flag_metric.get("rel_linf", ""),
        "platform_rel_l2": platform_metric.get("rel_l2", ""),
        "platform_rel_linf": platform_metric.get("rel_linf", ""),
        "limit_rel_l2": limits.get("rel_l2", ""),
        "limit_rel_linf": limits.get("rel_linf", ""),
        "flagfft_median_ms": result.get("flagfft_median_ms", ""),
        "ref_median_ms": result.get("ref_median_ms", ""),
        "speedup": result.get("speedup", ""),
        "baseline_valid": result.get("baseline_valid", ""),
        "plan": result.get("plan") or "",
        "data_file": result.get("data_file", ""),
        "duration_s": round(message.get("duration", 0), 3),
        "error": result.get("error", ""),
        "platform_error": platform_result.get("error", ""),
        "backend": ENV_INFO.get("backend", ""),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    return json_safe(row)


def requested_phases_passed(
    op_results: dict, run_accuracy: bool, run_performance: bool
) -> bool:
    phases = (["accuracy"] if run_accuracy else []) + (
        ["performance"] if run_performance else []
    )
    return bool(op_results) and all(
        op[phase]["status"] == "Passed"
        or (
            op[phase]["status"] == "Skipped"
            and op[phase].get("policy_skipped", False)
        )
        for op in op_results.values()
        for phase in phases
    )


def reanalyze_case(case: dict, case_dir: Path) -> dict:
    value = load_raw(
        case_dir / "input.bin", case["api"], tuple(case["shape"]), case["batch"]
    )
    reference_value = np.asarray(
        value, dtype=np.complex128 if np.iscomplexobj(value) else np.float64
    )
    del value
    try:
        reference = numpy_reference(
            reference_value,
            case["api"],
            tuple(case["shape"]),
            case["direction"],
        )
    finally:
        del reference_value
    case.pop("numpy_sha256", None)
    case["numpy_dtype"] = str(reference.dtype)
    for implementation, field in (
        ("flagfft", "accuracy"),
        ("platform", "platform_accuracy"),
    ):
        stage = case.get(field, {}).get("capture", {"status": "NotFound"})
        case[field] = compare_output(case, case_dir, implementation, reference, stage)
        case[field]["data_file"] = case["data_file"]
    write_json(case_dir / "case.json", case)
    return case


def analyze_only(output_dir: Path) -> int:
    started = time.monotonic()
    manifest_file = output_dir / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"manifest.json not found in {output_dir}")
    manifest = json.loads(manifest_file.read_text())
    artifact_policy = manifest["config"].get("artifact_policy", "all")
    if artifact_policy != "all":
        raise ValueError(
            "--analyze-only requires a result generated with --artifacts all; "
            f"this result used --artifacts {artifact_policy}"
        )
    old_summary_file = output_dir / "summary.json"
    old_summary = (
        json.loads(old_summary_file.read_text()) if old_summary_file.is_file() else {}
    )
    ENV_INFO.update(manifest["env"])
    ENV_INFO["analysis_numpy"] = np.__version__
    skipped_op_ids = set(manifest["config"].get("skipped_ops", []))
    skip_reasons = manifest["config"].get("skip_reasons", {})
    messages = []
    for case in manifest["cases"] if not manifest["config"]["performance_only"] else []:
        case_dir = output_dir / case["op_id"] / case["case_id"]
        record_file = case_dir / "case.json"
        if not record_file.is_file():
            continue
        record = reanalyze_case(json.loads(record_file.read_text()), case_dir)
        messages.append(
            {
                **case,
                "phase": "accuracy",
                "duration": record.get("duration", 0),
                "result": record["accuracy"],
                "platform_result": record["platform_accuracy"],
            }
        )
    baselines = {}
    for message in messages:
        key = case_name(message, performance=True)
        baselines[key] = (
            baselines.get(key, True)
            and message["platform_result"]["status"] == "Passed"
        )
    for case in (
        manifest["performance_cases"] if not manifest["config"]["accuracy_only"] else []
    ):
        path = (
            output_dir / case["op_id"] / "performance" / case["case_id"] / "result.json"
        )
        if not path.is_file():
            continue
        result = json.loads(path.read_text())["performance"]
        result["baseline_valid"] = baselines.get(case["case_id"])
        messages.append(
            {
                **case,
                "phase": "performance",
                "result": result,
                "duration": result.get("duration", 0),
            }
        )
    run_accuracy = not manifest["config"]["performance_only"]
    run_performance = not manifest["config"]["accuracy_only"]
    for case in manifest["cases"]:
        if case["op_id"] in skipped_op_ids and run_accuracy:
            messages.append(policy_skip_message(case, "accuracy"))
    for case in manifest["performance_cases"]:
        if case["op_id"] in skipped_op_ids and run_performance:
            messages.append(policy_skip_message(case, "performance"))
    results = aggregate_results(
        messages,
        manifest["operators"],
        manifest["cases"],
        run_accuracy,
        run_performance,
        skipped_op_ids,
        skip_reasons,
    )
    with (output_dir / "reanalyzed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INC_COLUMNS)
        writer.writeheader()
        writer.writerows(incremental_row(message) for message in messages)
    config = {
        **manifest["config"],
        "analyze_only": True,
        "analysis_duration": time.monotonic() - started,
    }
    duration = old_summary.get("summary", {}).get("total_duration", 0)
    write_summary(output_dir / "summary.json", results, config, duration)
    return 0 if requested_phases_passed(results, run_accuracy, run_performance) else 1


def handle_interrupt(signum, frame) -> None:
    global INTERRUPTED
    if not INTERRUPTED:
        INTERRUPTED = True
        pwarn("Interrupted; stopping workers and saving completed results")


def terminate_workers() -> None:
    for worker in WORKER_PROCESSES:
        if worker.is_alive():
            worker.terminate()
    for worker in WORKER_PROCESSES:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)


def read_op_list_file(path: str) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in Path(path).read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops", help="Comma-separated IDs from the 36-operator acceptance list"
    )
    parser.add_argument(
        "--op-list-file", help="One operator ID per line; # starts a comment"
    )
    parser.add_argument(
        "--start", help="Start at this operator in operators.yaml order"
    )
    parser.add_argument(
        "--combination",
        default="full",
        help="Group filter: full/all, or comma-separated " + ",".join(GROUPS),
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs or all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--accuracy-only", action="store_true")
    mode.add_argument("--performance-only", action="store_true")
    parser.add_argument("--build-dir", default=str(ROOT / "build"))
    parser.add_argument("--capture-bin", help="Override build/ctest/numpy_fft_capture")
    parser.add_argument(
        "--output-dir",
        help="Result directory; default workspace results/<timestamp>_acceptance36",
    )
    parser.add_argument(
        "--incremental-csv", help="CSV path; default <output-dir>/incremental.csv"
    )
    parser.add_argument(
        "--scales",
        help="Positive scales, comma-separated, or all; default matrix scales",
    )
    parser.add_argument(
        "--shapes", help="Exact configured shapes, comma-separated, e.g. 256,64x64"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Select only the first N correctness cases (partial run)",
    )
    parser.add_argument(
        "--analyze-only",
        metavar="RESULT_DIR",
        help="Recompute NumPy comparisons from saved inputs/outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded cases without GPU execution or result files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Independent timeout per native implementation",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--dump-output",
        action="store_true",
        help="Native stdout/stderr are always retained",
    )
    parser.add_argument(
        "--artifacts",
        dest="artifact_policy",
        choices=ARTIFACT_POLICIES,
        default="failed",
        help="Raw correctness artifacts: none, failed, or all (default: failed)",
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.timeout <= 0
        or args.iters <= 0
        or args.warmup < 0
        or (args.max_cases is not None and args.max_cases <= 0)
    ):
        parser.error(
            "timeout/iters/max-cases must be positive; warmup must be nonnegative"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    global INTERRUPTED
    args = parse_args(argv)
    init_colors(args.color)
    if args.analyze_only:
        return analyze_only(Path(args.analyze_only).resolve())
    build_dir = Path(args.build_dir).resolve()
    backend = detect_backend(build_dir)
    ENV_INFO["backend"] = backend
    ops = load_operators(ROOT / "conf" / "operators.yaml")
    matrix = load_test_matrix(ROOT / "conf" / "test_matrix.yaml")
    all_ids = {op["id"] for op in ops}
    selected = None
    if args.ops:
        selected = {part.strip() for part in args.ops.split(",") if part.strip()}
    if args.op_list_file:
        file_ids = set(read_op_list_file(args.op_list_file))
        selected = file_ids if selected is None else selected & file_ids
    if selected is not None:
        unknown = selected - all_ids
        if unknown:
            raise ValueError(f"unknown operator IDs: {', '.join(sorted(unknown))}")
        ops = [op for op in ops if op["id"] in selected]
    if args.start:
        ids = [op["id"] for op in ops]
        if args.start not in ids:
            raise ValueError(f"start operator not selected: {args.start}")
        ops = ops[ids.index(args.start) :]
    combinations = resolve_combination_names(args.combination)
    expanded_cases = expand_test_cases(
        ops, matrix, args.combination, args.scales, parse_shape_filter(args.shapes)
    )
    skip_reasons = {
        op["id"]: reason
        for op in ops
        if (reason := operator_skip_reason(op, backend)) is not None
    }
    skipped_ids = set(skip_reasons)
    runnable_cases = [
        case for case in expanded_cases if case["op_id"] not in skipped_ids
    ]
    if args.max_cases is not None:
        runnable_case_ids = {
            case["case_id"] for case in runnable_cases[: args.max_cases]
        }
    else:
        runnable_case_ids = {case["case_id"] for case in runnable_cases}
    all_cases = []
    for case in expanded_cases:
        if case["op_id"] in skipped_ids:
            all_cases.append({**case, "skip_reason": skip_reasons[case["op_id"]]})
        elif case["case_id"] in runnable_case_ids:
            all_cases.append(case)
    cases = [case for case in all_cases if case["op_id"] not in skipped_ids]
    skipped_cases = [case for case in all_cases if case["op_id"] in skipped_ids]
    active_ids = {case["op_id"] for case in all_cases}
    ops = [op for op in ops if op["id"] in active_ids]
    skipped_ids &= active_ids
    skip_reasons = {op_id: skip_reasons[op_id] for op_id in skipped_ids}
    if not all_cases:
        raise ValueError("no test cases selected")
    perf_cases = performance_cases(all_cases)
    runnable_perf_cases = performance_cases(cases)
    pinfo(
        f"Expanded {len(all_cases)} accuracy cases and {len(perf_cases)} performance cases "
        f"from {len(ops)} operators"
    )
    if skipped_cases:
        pwarn(
            f"Backend {backend} policy skipped {len(skipped_cases)} cases "
            f"across {len(skipped_ids)} operators"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": backend,
                    "operators": [op["id"] for op in ops],
                    "skipped_operators": skip_reasons,
                    "cases": all_cases,
                    "performance_cases": perf_cases,
                },
                indent=2,
            )
        )
        return 0
    capture_bin = (
        Path(args.capture_bin).resolve()
        if args.capture_bin
        else build_dir / "ctest" / "numpy_fft_capture"
    )
    if not args.performance_only and not capture_bin.is_file():
        raise ValueError(
            f"capture executable not found: {capture_bin}; build with FLAGFFT_BUILD_TESTS=ON"
        )
    if not args.accuracy_only and not (build_dir / "flagfft-cli").is_file():
        raise ValueError(f"benchmark executable not found: {build_dir / 'flagfft-cli'}")
    probe_env(build_dir)
    if args.gpus == "all":
        count = ENV_INFO.get("torch", {}).get("device_count", 0)
        if not count:
            raise ValueError("cannot discover GPUs; specify --gpus explicitly")
        gpu_ids = list(range(count))
    else:
        gpu_ids = [int(part.strip()) for part in args.gpus.split(",")]
    if (
        not gpu_ids
        or any(gpu < 0 for gpu in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            ROOT.parent
            / "results"
            / (datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_acceptance36")
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"result directory is not empty: {output_dir}; use a new directory or --analyze-only"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.incremental_csv).resolve()
        if args.incremental_csv
        else output_dir / "incremental.csv"
    )
    if csv_path.exists():
        raise ValueError(f"incremental CSV already exists: {csv_path}")
    config = {
        "ops": [op["id"] for op in ops],
        "skipped_ops": sorted(skipped_ids),
        "skip_reasons": skip_reasons,
        "combinations": combinations,
        "gpus": gpu_ids,
        "accuracy_only": args.accuracy_only,
        "performance_only": args.performance_only,
        "scales": parse_scales(args.scales, matrix.get("scales", [1.0])),
        "shapes": args.shapes,
        "max_cases": args.max_cases,
        "timeout": args.timeout,
        "warmup": args.warmup,
        "iters": args.iters,
        "build_dir": str(build_dir),
        "capture_bin": str(capture_bin),
        "test_matrix": matrix,
        "incremental_csv": str(csv_path),
        "artifact_policy": args.artifact_policy,
    }
    write_json(
        output_dir / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "env": ENV_INFO,
            "config": config,
            "operators": ops,
            "cases": all_cases,
            "performance_cases": perf_cases,
        },
    )
    jobs = {
        case["case_id"]: {"performance_case": case, "accuracy_cases": []}
        for case in runnable_perf_cases
    }
    for case in cases:
        jobs[case_name(case, performance=True)]["accuracy_cases"].append(case)
    # Spawn avoids forking an initialized CUDA/PyTorch runtime.
    context = multiprocessing.get_context("spawn")
    work_queue = context.Queue()
    display_queue = context.Queue()
    for job in jobs.values():
        work_queue.put(job)
    for _ in gpu_ids:
        work_queue.put(None)
    INTERRUPTED = False
    WORKER_PROCESSES.clear()
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    started = time.monotonic()
    messages = []
    total = (0 if args.performance_only else len(all_cases)) + (
        0 if args.accuracy_only else len(perf_cases)
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INC_COLUMNS)
        writer.writeheader()
        stream.flush()
        for gpu_id in gpu_ids:
            worker = context.Process(
                target=worker_proc,
                args=(
                    gpu_id,
                    work_queue,
                    display_queue,
                    capture_bin,
                    build_dir,
                    output_dir,
                    args,
                ),
            )
            worker.start()
            WORKER_PROCESSES.append(worker)

        def record(message):
            messages.append(message)
            writer.writerow(incremental_row(message))
            stream.flush()
            extra = (
                f", platform={message['platform_result']['status']}"
                if "platform_result" in message
                else ""
            )
            pinfo(
                f"[{len(messages)}/{total}] GPU {message['gpu']} {message['phase']} "
                f"{message['case_id']}: {message['status']}{extra}"
            )

        try:
            while any(worker.is_alive() for worker in WORKER_PROCESSES):
                if INTERRUPTED:
                    terminate_workers()
                    break
                try:
                    record(display_queue.get(timeout=0.2))
                except queue.Empty:
                    pass
            # Process joins flush multiprocessing queue feeder threads first.
            for worker in WORKER_PROCESSES:
                worker.join(timeout=5)
            while True:
                try:
                    record(display_queue.get(timeout=0.2))
                except queue.Empty:
                    break
        finally:
            if any(worker.is_alive() for worker in WORKER_PROCESSES):
                terminate_workers()
        for case in skipped_cases:
            if not args.performance_only:
                message = policy_skip_message(case, "accuracy")
                messages.append(message)
                writer.writerow(incremental_row(message))
                stream.flush()
            if not args.accuracy_only:
                message = policy_skip_message(case, "performance")
                messages.append(message)
                writer.writerow(incremental_row(message))
                stream.flush()
    results = aggregate_results(
        messages,
        ops,
        all_cases,
        not args.performance_only,
        not args.accuracy_only,
        skipped_ids,
        skip_reasons,
    )
    config["interrupted"] = INTERRUPTED
    config["worker_exitcodes"] = [worker.exitcode for worker in WORKER_PROCESSES]
    summary = write_summary(
        output_dir / "summary.json", results, config, time.monotonic() - started
    )
    pinfo(json.dumps(summary["summary"]))
    if INTERRUPTED:
        return 130
    return (
        0
        if requested_phases_passed(
            results, not args.performance_only, not args.accuracy_only
        )
        else 1
    )


def cli() -> int:
    try:
        return main()
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        perror(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(cli())
