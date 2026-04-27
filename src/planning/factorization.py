from __future__ import annotations

import math

from .types import LeafPlan

SUPPORTED_RADICES = (17, 16, 13, 11, 9, 8, 7, 6, 5, 4, 3, 2)
SPECIALIZED_BUTTERFLY_RADICES = {2, 4, 8, 16}
SPECIALIZED_INLINE_CODELET_RADICES: set[int] = set()
SPECIALIZED_DIRECT_CODELET_RADICES = {3, 5, 6, 7, 9, 11, 13}
MAX_LANES = 128
LEAF_MAX_N = 4096
LEAF_SMEM_BUDGET_BYTES = 48 * 1024
DIRECT_DFT_MAX_N = 64

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


def factorize_or_raise(n: int) -> tuple[int, ...]:
    factors, rem = factorize_supported_radices(n)
    if rem != 1:
        raise ValueError(
            f"length {n} is not fully factorable by supported radices, "
            f"remainder={rem}, partial={factors}"
        )
    if not factors:
        raise ValueError("at least one radix stage is required")
    return factors


def enumerate_supported_factorizations(n: int) -> tuple[tuple[int, ...], ...]:
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
        for tail in enumerate_supported_factorizations(n // radix):
            factorizations.add((radix,) + tail)

    result = tuple(sorted(factorizations))
    _FACTORIZATION_CACHE[n] = result
    return result


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    return 1 << (lanes - 1).bit_length()


def choose_lanes(n: int, factors: tuple[int, ...], max_lanes: int = MAX_LANES) -> int:
    counts = [n // radix for radix in factors]
    gcd_all = counts[0]
    for count in counts[1:]:
        gcd_all = math.gcd(gcd_all, count)

    upper = min(gcd_all, max_lanes)
    for candidate in range(upper, 0, -1):
        if gcd_all % candidate == 0:
            return candidate
    return 1


def score_leaf_factorization(
    n: int, factors: tuple[int, ...]
) -> tuple[int, int, int, int, tuple[int, ...]]:
    lanes = choose_lanes(n, factors)
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


def select_leaf_factors(n: int) -> tuple[int, ...]:
    cached = _BEST_LEAF_FACTORS_CACHE.get(n)
    if cached is not None:
        return cached

    candidates = enumerate_supported_factorizations(n)
    if not candidates:
        cached = factorize_or_raise(n)
    else:
        cached = max(candidates, key=lambda factors: score_leaf_factorization(n, factors))

    _BEST_LEAF_FACTORS_CACHE[n] = cached
    return cached


def choose_num_warps(lanes: int) -> int:
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
    lanes = choose_lanes(n, factors)
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
        num_warps=choose_num_warps(lanes),
        generic_radices=generic_radices,
        smem_size=1 << (n - 1).bit_length(),
    )


def should_use_leaf(n: int, factors: tuple[int, ...]) -> bool:
    if n > LEAF_MAX_N:
        return False
    return estimate_leaf_smem_bytes(n, factors) <= LEAF_SMEM_BUDGET_BYTES


def clear_factorization_caches() -> None:
    _FACTORIZATION_CACHE.clear()
    _BEST_LEAF_FACTORS_CACHE.clear()


__all__ = [
    "DIRECT_DFT_MAX_N",
    "LEAF_MAX_N",
    "LEAF_SMEM_BUDGET_BYTES",
    "MAX_LANES",
    "SPECIALIZED_BUTTERFLY_RADICES",
    "SPECIALIZED_DIRECT_CODELET_RADICES",
    "SPECIALIZED_INLINE_CODELET_RADICES",
    "SUPPORTED_RADICES",
    "choose_lanes",
    "choose_num_warps",
    "clear_factorization_caches",
    "enumerate_supported_factorizations",
    "estimate_leaf_smem_bytes",
    "factorize_or_raise",
    "factorize_supported_radices",
    "lane_block_for",
    "make_leaf_plan",
    "score_leaf_factorization",
    "select_leaf_factors",
    "should_use_leaf",
]
