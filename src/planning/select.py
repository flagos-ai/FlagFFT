from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .factorization import (
    DIRECT_DFT_MAX_N,
    clear_factorization_caches,
    estimate_leaf_smem_bytes,
    factorize_or_raise,
    factorize_supported_radices,
    make_leaf_plan,
    select_leaf_factors,
    should_use_leaf,
)
from .types import (
    DirectDFTPlan,
    FFTDecompositionSpec,
    FFTPlan,
    FFTPlanRequest,
    FourStepPlan,
    LeafPlan,
    PlanNode,
    get_plan_root,
    is_plan_node,
)

_PLAN_CACHE: dict[int, FFTPlan] = {}
_NODE_CACHE: dict[int, PlanNode] = {}
_PLAN_COST_CACHE: dict[int, float] = {}
_DIVISOR_CACHE: dict[int, tuple[int, ...]] = {}


@dataclass(frozen=True)
class PlanCandidate:
    node: PlanNode
    cost: float
    reason: str


def _default_request(n: int, request: FFTPlanRequest | None) -> FFTPlanRequest:
    if request is None:
        return FFTPlanRequest(length=n)
    if request.length != n:
        raise ValueError(f"request length mismatch: request={request.length}, n={n}")
    return request


def _is_sequence_spec(spec: object) -> bool:
    return isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray))


def _enumerate_divisors(n: int) -> tuple[int, ...]:
    cached = _DIVISOR_CACHE.get(n)
    if cached is not None:
        return cached

    divisors: list[int] = []
    root = int(math.isqrt(n))
    for divisor in range(1, root + 1):
        if n % divisor != 0:
            continue
        divisors.append(divisor)
        mate = n // divisor
        if mate != divisor:
            divisors.append(mate)

    result = tuple(sorted(divisors))
    _DIVISOR_CACHE[n] = result
    return result


def _build_manual_leaf_plan(n: int) -> LeafPlan:
    factors = factorize_or_raise(n)
    if not should_use_leaf(n, factors):
        raise ValueError(
            f"manual ct_leaf plan for length {n} exceeds the leaf constraints; "
            "add an explicit split instead"
        )
    return make_leaf_plan(n, select_leaf_factors(n))


def _build_direct_dft_plan(n: int, impl: str = "torch_matmul") -> DirectDFTPlan:
    if n <= 0:
        raise ValueError("direct DFT length must be positive")
    return DirectDFTPlan(length=n, impl=impl)


def _estimate_leaf_warm_cost(n: int) -> float:
    leaf = make_leaf_plan(n, select_leaf_factors(n))
    return float(leaf.length * len(leaf.factors) * leaf.num_warps) / float(leaf.lanes)


def _estimate_direct_dft_cost(n: int) -> float:
    return float(n * n)


def _estimate_node_cost(n: int) -> float:
    cached = _PLAN_COST_CACHE.get(n)
    if cached is not None:
        return cached

    candidates = _build_auto_candidates(n)
    if not candidates:
        raise ValueError(f"length {n} has no supported FFT implementation route")
    cost = min(candidate.cost for candidate in candidates)
    _PLAN_COST_CACHE[n] = cost
    return cost


def _four_step_cost(n1: int, n2: int) -> float:
    return n2 * _estimate_node_cost(n1) + n1 * _estimate_node_cost(n2) + float(n1 * n2)


def _candidate_priority(node: PlanNode) -> int:
    return {
        "ct_leaf": 0,
        "four_step": 1,
        "stockham_autosort": 2,
        "direct_dft": 3,
    }.get(node.kind, 99)


def _build_auto_candidates(n: int) -> list[PlanCandidate]:
    if n <= 0:
        raise ValueError("FFT length must be positive")

    candidates: list[PlanCandidate] = []
    factors, rem = factorize_supported_radices(n)

    if rem == 1 and factors and should_use_leaf(n, factors):
        node = make_leaf_plan(n, select_leaf_factors(n))
        candidates.append(PlanCandidate(node=node, cost=_estimate_leaf_warm_cost(n), reason="ct_leaf"))

    if n <= DIRECT_DFT_MAX_N:
        direct_cost = _estimate_direct_dft_cost(n)
        candidates.append(
            PlanCandidate(
                node=_build_direct_dft_plan(n),
                cost=direct_cost,
                reason="direct_dft_small_n",
            )
        )

    for n1 in _enumerate_divisors(n):
        if n1 <= 1 or n1 >= n:
            continue
        n2 = n // n1
        try:
            row_node = _build_auto_node(n1)
            col_node = _build_auto_node(n2)
        except ValueError:
            continue
        node = FourStepPlan(length=n, n1=n1, n2=n2, row_plan=row_node, col_plan=col_node)
        balance = abs(math.log(float(n1)) - math.log(float(n2)))
        candidates.append(
            PlanCandidate(
                node=node,
                cost=_four_step_cost(n1, n2) + balance,
                reason="four_step",
            )
        )

    return candidates


def _select_candidate(candidates: list[PlanCandidate]) -> PlanCandidate:
    if not candidates:
        raise ValueError("no FFT plan candidates were generated")
    return min(candidates, key=lambda candidate: (candidate.cost, _candidate_priority(candidate.node)))


def _build_auto_node(n: int) -> PlanNode:
    cached = _NODE_CACHE.get(n)
    if cached is not None:
        return cached

    candidate = _select_candidate(_build_auto_candidates(n))
    _NODE_CACHE[n] = candidate.node
    _PLAN_COST_CACHE[n] = candidate.cost
    return candidate.node


def enumerate_plan_candidates(n: int) -> list[PlanCandidate]:
    return sorted(
        _build_auto_candidates(n),
        key=lambda candidate: (candidate.cost, _candidate_priority(candidate.node)),
    )


def choose_four_step_split(n: int) -> tuple[int, int]:
    if n <= 1:
        raise ValueError(f"length {n} cannot be split further")

    best_score: tuple[float, float, int] | None = None
    best_split: tuple[int, int] | None = None
    for n1 in _enumerate_divisors(n):
        if n1 <= 1 or n1 >= n:
            continue
        n2 = n // n1
        try:
            cost = _four_step_cost(n1, n2)
        except ValueError:
            continue
        balance = abs(math.log(float(n1)) - math.log(float(n2)))
        score = (cost, balance, abs(n1 - n2))
        if best_score is None or score < best_score:
            best_score = score
            best_split = (n1, n2)
    if best_split is None:
        raise ValueError(f"length {n} has no non-trivial supported split")
    return best_split


def _parse_manual_split_spec(
    spec: FFTDecompositionSpec,
) -> tuple[int, int, FFTDecompositionSpec | None, FFTDecompositionSpec | None]:
    if isinstance(spec, Mapping):
        split = spec.get("split")
        if split is None and "n1" in spec and "n2" in spec:
            split = [spec["n1"], spec["n2"]]
        if not _is_sequence_spec(split) or len(split) != 2:
            raise ValueError("mapping split spec must provide split=[n1, n2] or n1/n2")
        n1, n2 = split
        return int(n1), int(n2), spec.get("row", spec.get("row_plan")), spec.get(
            "col", spec.get("col_plan")
        )

    if not _is_sequence_spec(spec):
        raise ValueError(
            "manual split spec must be 'leaf', 'ct_leaf', 'direct_dft', [n1, n2], "
            "[n1, n2, row_spec, col_spec], or {'split': [n1, n2], 'row': ..., 'col': ...}"
        )

    parts = list(spec)
    if len(parts) == 2:
        n1, n2 = parts
        return int(n1), int(n2), None, None
    if len(parts) == 4:
        n1, n2, row_spec, col_spec = parts
        return int(n1), int(n2), row_spec, col_spec
    raise ValueError("sequence split spec must have length 2 or 4")


def _coerce_exact_plan(n: int, spec: object) -> FFTPlan | None:
    if isinstance(spec, FFTPlan):
        plan = spec
    elif is_plan_node(spec):
        plan = FFTPlan(root=spec, source="manual")
    elif isinstance(spec, Mapping) and ("root" in spec or "schema_version" in spec):
        from .serde import plan_from_dict

        plan = plan_from_dict(spec)
    elif isinstance(spec, Mapping) and (
        (spec.get("kind") == "ct_leaf" and "factors" in spec)
        or (spec.get("kind") == "direct_dft" and "length" in spec)
        or (
            spec.get("kind") == "four_step"
            and ("row" in spec or "row_plan" in spec)
            and ("col" in spec or "col_plan" in spec)
            and isinstance(spec.get("row", spec.get("row_plan")), Mapping)
            and isinstance(spec.get("col", spec.get("col_plan")), Mapping)
        )
        or (spec.get("kind") == "stockham_autosort" and "length" in spec and "factors" in spec)
    ):
        from .serde import node_from_dict

        plan = FFTPlan(root=node_from_dict(spec), source="manual")
    else:
        return None

    if plan.length != n:
        raise ValueError(f"plan length mismatch: plan={plan.length}, requested={n}")
    return plan


def _build_node_from_spec(n: int, spec: FFTDecompositionSpec | None) -> PlanNode:
    if spec is None or spec == "auto":
        return _build_auto_node(n)

    exact = _coerce_exact_plan(n, spec)
    if exact is not None:
        return exact.root

    if spec in ("leaf", "ct_leaf"):
        return _build_manual_leaf_plan(n)
    if spec in ("direct", "direct_dft"):
        return _build_direct_dft_plan(n)

    if isinstance(spec, Mapping):
        kind = spec.get("kind")
        if kind in ("leaf", "ct_leaf"):
            return _build_manual_leaf_plan(n)
        if kind in ("direct", "direct_dft"):
            return _build_direct_dft_plan(n, impl=str(spec.get("impl", "torch_matmul")))
        if kind == "stockham_autosort":
            from .serde import node_from_dict

            node = node_from_dict(spec)
            if node.length != n:
                raise ValueError(f"stockham plan length mismatch: plan={node.length}, requested={n}")
            return node

    n1, n2, row_spec, col_spec = _parse_manual_split_spec(spec)
    if n1 <= 1 or n2 <= 1:
        raise ValueError("manual split dimensions must both be greater than 1")
    if n1 * n2 != n:
        raise ValueError(f"manual split [{n1}, {n2}] does not match length {n}")

    return FourStepPlan(
        length=n,
        n1=n1,
        n2=n2,
        row_plan=_build_node_from_spec(n1, row_spec),
        col_plan=_build_node_from_spec(n2, col_spec),
    )


def build_fft_plan(
    n: int,
    split_spec: FFTDecompositionSpec | FFTPlan | PlanNode | None = None,
    *,
    request: FFTPlanRequest | None = None,
) -> FFTPlan:
    resolved_request = _default_request(n, request)

    exact = _coerce_exact_plan(n, split_spec)
    if exact is not None:
        return FFTPlan(
            root=exact.root,
            request=resolved_request if request is not None else exact.request,
            source=exact.source,
            estimated_cost=exact.estimated_cost,
            tags=exact.tags,
        )

    if split_spec is None:
        if request is None:
            cached = _PLAN_CACHE.get(n)
            if cached is not None:
                return cached
        node = _build_auto_node(n)
        plan = FFTPlan(
            root=node,
            request=resolved_request,
            source="auto",
            estimated_cost=_estimate_node_cost(n),
        )
        if request is None:
            _PLAN_CACHE[n] = plan
        return plan

    node = _build_node_from_spec(n, split_spec)
    return FFTPlan(root=node, request=resolved_request, source="manual")


def get_fft_plan(n: int) -> FFTPlan:
    return build_fft_plan(n)


def clear_plan_cache() -> None:
    _PLAN_CACHE.clear()
    _NODE_CACHE.clear()
    _PLAN_COST_CACHE.clear()
    _DIVISOR_CACHE.clear()
    clear_factorization_caches()


__all__ = [
    "PlanCandidate",
    "build_fft_plan",
    "choose_four_step_split",
    "clear_plan_cache",
    "enumerate_plan_candidates",
    "get_fft_plan",
]
