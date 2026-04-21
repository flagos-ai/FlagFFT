from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

SUPPORTED_RADICES = (17, 16, 13, 11, 9, 8, 7, 6, 5, 4, 3, 2)
SPECIALIZED_BUTTERFLY_RADICES = {2, 4, 8, 16}
SPECIALIZED_INLINE_CODELET_RADICES = set()
SPECIALIZED_DIRECT_CODELET_RADICES = {3, 5, 6, 7, 9, 11, 13}
MAX_LANES = 128
LEAF_MAX_N = 4096
LEAF_SMEM_BUDGET_BYTES = 48 * 1024


@dataclass(frozen=True)
class LeafPlan:
    length: int
    factors: tuple[int, ...]
    remainder: int
    lanes: int
    num_warps: int
    generic_radices: tuple[int, ...]
    smem_size: int


@dataclass(frozen=True)
class FourStepPlan:
    length: int
    n1: int
    n2: int
    row_plan: FFTPlan
    col_plan: FFTPlan


FFTPlan = LeafPlan | FourStepPlan
FFTDecompositionSpec: TypeAlias = str | Sequence[Any] | Mapping[str, Any]

_PLAN_CACHE: dict[int, FFTPlan] = {}
_DIVISOR_CACHE: dict[int, tuple[int, ...]] = {}
_FACTORIZATION_CACHE: dict[int, tuple[tuple[int, ...], ...]] = {}
_BEST_LEAF_FACTORS_CACHE: dict[int, tuple[int, ...]] = {}


def factorize_supported_radices(n: int) -> tuple[tuple[int, ...], int]:
    if n <= 0:
        raise ValueError("FFT length must be positive")

    rem = n
    factors: list[int] = []
    while rem > 1:
        found = False
        for radix in SUPPORTED_RADICES:
            if rem % radix == 0:
                factors.append(radix)
                rem //= radix
                found = True
                break
        if not found:
            break
    return tuple(factors), rem


def _factorize_or_raise(n: int) -> tuple[int, ...]:
    factors, rem = factorize_supported_radices(n)
    if rem != 1:
        raise ValueError(
            f"length {n} is not fully factorable by supported radices, "
            f"remainder={rem}, partial={factors}"
        )
    if not factors:
        raise ValueError("at least one radix stage is required")
    return factors


def _enumerate_supported_factorizations(n: int) -> tuple[tuple[int, ...], ...]:
    cached = _FACTORIZATION_CACHE.get(n)
    if cached is not None:
        return cached

    factorizations: set[tuple[int, ...]] = set()
    for radix in SUPPORTED_RADICES:
        if n % radix != 0:
            continue
        if n == radix:
            factorizations.add((radix,))
            continue
        for tail in _enumerate_supported_factorizations(n // radix):
            factorizations.add((radix,) + tail)

    result = tuple(sorted(factorizations))
    _FACTORIZATION_CACHE[n] = result
    return result


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    return 1 << (lanes - 1).bit_length()


def _choose_lanes(n: int, factors: tuple[int, ...], max_lanes: int = MAX_LANES) -> int:
    counts = [n // radix for radix in factors]
    gcd_all = counts[0]
    for count in counts[1:]:
        gcd_all = math.gcd(gcd_all, count)

    upper = min(gcd_all, max_lanes)
    for candidate in range(upper, 0, -1):
        if gcd_all % candidate == 0:
            return candidate
    return 1


def _score_leaf_factorization(
    n: int, factors: tuple[int, ...]
) -> tuple[int, int, int, int, tuple[int, ...]]:
    lanes = _choose_lanes(n, factors)
    lane_block = lane_block_for(lanes)
    generic_stage_count = sum(
        1
        for radix in factors
        if radix not in SPECIALIZED_BUTTERFLY_RADICES
        and radix not in SPECIALIZED_INLINE_CODELET_RADICES
        and radix not in SPECIALIZED_DIRECT_CODELET_RADICES
    )
    return (
        lanes,
        -lane_block,
        -generic_stage_count,
        -len(factors),
        factors,
    )


def _select_leaf_factors(n: int) -> tuple[int, ...]:
    cached = _BEST_LEAF_FACTORS_CACHE.get(n)
    if cached is not None:
        return cached

    candidates = _enumerate_supported_factorizations(n)
    if not candidates:
        cached = _factorize_or_raise(n)
    else:
        cached = max(candidates, key=lambda factors: _score_leaf_factorization(n, factors))

    _BEST_LEAF_FACTORS_CACHE[n] = cached
    return cached


def _choose_num_warps(lanes: int) -> int:
    warps = max(1, (lane_block_for(lanes) + 31) // 32)
    choice = 1
    while choice < warps:
        choice *= 2
    return min(choice, 8)


def estimate_leaf_smem_bytes(n: int, factors: tuple[int, ...]) -> int:
    if len(factors) <= 1:
        return 0
    smem_n = 1 << (n - 1).bit_length()
    return 4 * smem_n * 4


def make_leaf_plan(n: int, factors: tuple[int, ...], rem: int = 1) -> LeafPlan:
    lanes = _choose_lanes(n, factors)
    generic_radices = tuple(
        sorted(
            radix
            for radix in set(factors)
            if radix not in SPECIALIZED_BUTTERFLY_RADICES
            and radix not in SPECIALIZED_INLINE_CODELET_RADICES
            and radix not in SPECIALIZED_DIRECT_CODELET_RADICES
        )
    )
    return LeafPlan(
        length=n,
        factors=factors,
        remainder=rem,
        lanes=lanes,
        num_warps=_choose_num_warps(lanes),
        generic_radices=generic_radices,
        smem_size=1 << (n - 1).bit_length(),
    )


def should_use_leaf(n: int, factors: tuple[int, ...]) -> bool:
    if n > LEAF_MAX_N:
        return False
    return estimate_leaf_smem_bytes(n, factors) <= LEAF_SMEM_BUDGET_BYTES


def _enumerate_supported_divisors(n: int) -> tuple[int, ...]:
    cached = _DIVISOR_CACHE.get(n)
    if cached is not None:
        return cached

    divisors = {1, n}
    if n > 1:
        for radix in SUPPORTED_RADICES:
            if n % radix != 0 or radix == n:
                continue
            child = n // radix
            for value in _enumerate_supported_divisors(child):
                divisors.add(value)
                divisors.add(value * radix)

    result = tuple(sorted(divisors))
    _DIVISOR_CACHE[n] = result
    return result


def choose_four_step_split(n: int) -> tuple[int, int]:
    if n <= 1:
        raise ValueError(f"length {n} cannot be split further")

    root = math.sqrt(float(n))
    best_score: tuple[float, float] | None = None
    best_divisor: int | None = None
    for divisor in _enumerate_supported_divisors(n):
        if divisor <= 1 or divisor >= n:
            continue
        mate = n // divisor
        balance = abs(math.log(divisor) - math.log(mate))
        score = (balance, abs(divisor - root))
        if best_score is None or score < best_score:
            best_score = score
            best_divisor = divisor
    if best_divisor is None:
        raise ValueError(f"length {n} has no non-trivial supported split")

    n1 = best_divisor
    n2 = n // n1
    return min(n1, n2), max(n1, n2)


def get_fft_plan(n: int) -> FFTPlan:
    cached = _PLAN_CACHE.get(n)
    if cached is not None:
        return cached

    factors = _factorize_or_raise(n)
    if should_use_leaf(n, factors):
        plan: FFTPlan = make_leaf_plan(n, _select_leaf_factors(n))
    else:
        n1, n2 = choose_four_step_split(n)
        plan = FourStepPlan(
            length=n,
            n1=n1,
            n2=n2,
            row_plan=get_fft_plan(n1),
            col_plan=get_fft_plan(n2),
        )
    _PLAN_CACHE[n] = plan
    return plan


def _build_manual_leaf_plan(n: int) -> LeafPlan:
    factors = _factorize_or_raise(n)
    if not should_use_leaf(n, factors):
        raise ValueError(
            f"manual leaf plan for length {n} exceeds the leaf constraints; "
            "add an explicit split instead"
        )
    return make_leaf_plan(n, _select_leaf_factors(n))


def _is_sequence_spec(spec: object) -> bool:
    return isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray))


def _parse_manual_split_spec(
    spec: FFTDecompositionSpec,
) -> tuple[int, int, FFTDecompositionSpec | None, FFTDecompositionSpec | None]:
    if isinstance(spec, Mapping):
        kind = spec.get("kind")
        if kind == "leaf":
            raise ValueError("leaf specs must be handled before split parsing")
        split = spec.get("split")
        if not _is_sequence_spec(split) or len(split) != 2:
            raise ValueError("mapping split spec must provide split=[n1, n2]")
        n1, n2 = split
        return int(n1), int(n2), spec.get("row"), spec.get("col")

    if not _is_sequence_spec(spec):
        raise ValueError(
            "manual split spec must be 'leaf', [n1, n2], [n1, n2, row_spec, col_spec], "
            "or {'split': [n1, n2], 'row': ..., 'col': ...}"
        )

    parts = list(spec)
    if len(parts) == 2:
        n1, n2 = parts
        return int(n1), int(n2), None, None
    if len(parts) == 4:
        n1, n2, row_spec, col_spec = parts
        return int(n1), int(n2), row_spec, col_spec
    raise ValueError("sequence split spec must have length 2 or 4")


def _build_fft_plan_from_spec(n: int, spec: FFTDecompositionSpec) -> FFTPlan:
    if spec == "auto":
        return get_fft_plan(n)
    if spec == "leaf":
        return _build_manual_leaf_plan(n)
    if isinstance(spec, Mapping) and spec.get("kind") == "leaf":
        return _build_manual_leaf_plan(n)

    _factorize_or_raise(n)
    n1, n2, row_spec, col_spec = _parse_manual_split_spec(spec)
    if n1 <= 1 or n2 <= 1:
        raise ValueError("manual split dimensions must both be greater than 1")
    if n1 * n2 != n:
        raise ValueError(f"manual split [{n1}, {n2}] does not match length {n}")

    return FourStepPlan(
        length=n,
        n1=n1,
        n2=n2,
        row_plan=build_fft_plan(n1, row_spec),
        col_plan=build_fft_plan(n2, col_spec),
    )


def build_fft_plan(n: int, split_spec: FFTDecompositionSpec | None = None) -> FFTPlan:
    if split_spec is None:
        return get_fft_plan(n)
    return _build_fft_plan_from_spec(n, split_spec)


def clear_plan_cache() -> None:
    _PLAN_CACHE.clear()
    _DIVISOR_CACHE.clear()
    _FACTORIZATION_CACHE.clear()
    _BEST_LEAF_FACTORS_CACHE.clear()


def plan_depth(plan: FFTPlan) -> int:
    if isinstance(plan, LeafPlan):
        return 1
    return 1 + max(plan_depth(plan.row_plan), plan_depth(plan.col_plan))


def collect_leaf_plans(plan: FFTPlan) -> list[LeafPlan]:
    if isinstance(plan, LeafPlan):
        return [plan]
    return collect_leaf_plans(plan.row_plan) + collect_leaf_plans(plan.col_plan)


def unique_leaf_plans(plan: FFTPlan) -> list[LeafPlan]:
    unique: dict[tuple[int, tuple[int, ...], int], LeafPlan] = {}
    for leaf in collect_leaf_plans(plan):
        key = (leaf.length, leaf.factors, leaf.lanes)
        unique.setdefault(key, leaf)
    return sorted(unique.values(), key=lambda leaf: (leaf.length, leaf.factors, leaf.lanes))


def max_leaf_smem_bytes(plan: FFTPlan) -> int:
    leaf_plans = collect_leaf_plans(plan)
    if not leaf_plans:
        return 0
    return max(estimate_leaf_smem_bytes(leaf.length, leaf.factors) for leaf in leaf_plans)


def describe_fft_plan(plan: FFTPlan) -> str:
    if isinstance(plan, LeafPlan):
        smem_bytes = estimate_leaf_smem_bytes(plan.length, plan.factors)
        lane_block = lane_block_for(plan.lanes)
        return (
            f"leaf(n={plan.length}, factors={plan.factors}, lanes={plan.lanes}/{lane_block}, "
            f"warps={plan.num_warps}, smem={smem_bytes}B)"
        )
    return (
        f"four_step(n={plan.length}, n1={plan.n1}, n2={plan.n2}, "
        f"depth={plan_depth(plan)})"
    )


__all__ = [
    "FFTDecompositionSpec",
    "FFTPlan",
    "FourStepPlan",
    "LeafPlan",
    "MAX_LANES",
    "LEAF_MAX_N",
    "LEAF_SMEM_BUDGET_BYTES",
    "SPECIALIZED_BUTTERFLY_RADICES",
    "SPECIALIZED_DIRECT_CODELET_RADICES",
    "SPECIALIZED_INLINE_CODELET_RADICES",
    "SUPPORTED_RADICES",
    "build_fft_plan",
    "choose_four_step_split",
    "clear_plan_cache",
    "collect_leaf_plans",
    "describe_fft_plan",
    "estimate_leaf_smem_bytes",
    "factorize_supported_radices",
    "get_fft_plan",
    "make_leaf_plan",
    "max_leaf_smem_bytes",
    "plan_depth",
    "should_use_leaf",
    "unique_leaf_plans",
]
