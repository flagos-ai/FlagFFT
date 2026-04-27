from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import field, dataclass
from typing import Any, Literal, TypeAlias

PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FFTPlanRequest:
    length: int
    op: str = "fft"
    dtype: str = "complex64"
    device: str = "cuda"
    direction: str = "forward"
    norm: str | None = None
    batch: int | None = None
    input_layout: str = "contiguous"
    output_order: str = "natural"


@dataclass(frozen=True)
class LeafPlan:
    length: int
    factors: tuple[int, ...]
    remainder: int
    lanes: int
    num_warps: int
    generic_radices: tuple[int, ...]
    smem_size: int
    kind: Literal["ct_leaf"] = field(default="ct_leaf", init=False)


CTLeafPlan = LeafPlan


@dataclass(frozen=True)
class DirectDFTPlan:
    length: int
    impl: str = "torch_matmul"
    kind: Literal["direct_dft"] = field(default="direct_dft", init=False)


@dataclass(frozen=True)
class StockhamPlan:
    length: int
    factors: tuple[int, ...]
    stages: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    kind: Literal["stockham_autosort"] = field(default="stockham_autosort", init=False)


@dataclass(frozen=True)
class FourStepPlan:
    length: int
    n1: int
    n2: int
    row_plan: PlanNode
    col_plan: PlanNode
    kind: Literal["four_step"] = field(default="four_step", init=False)


PlanNode: TypeAlias = LeafPlan | FourStepPlan | DirectDFTPlan | StockhamPlan
FFTDecompositionSpec: TypeAlias = str | Sequence[Any] | Mapping[str, Any]


@dataclass(frozen=True)
class FFTPlan:
    root: PlanNode
    request: FFTPlanRequest | None = None
    schema_version: int = PLAN_SCHEMA_VERSION
    source: str = "auto"
    estimated_cost: float | None = None
    tags: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request is None:
            object.__setattr__(self, "request", FFTPlanRequest(length=self.root.length))
        elif self.request.length != self.root.length:
            raise ValueError(
                f"plan request length mismatch: request={self.request.length}, root={self.root.length}"
            )

    @property
    def length(self) -> int:
        return self.root.length

    @property
    def kind(self) -> str:
        return self.root.kind


def is_plan_node(value: object) -> bool:
    return isinstance(value, (LeafPlan, FourStepPlan, DirectDFTPlan, StockhamPlan))


def get_plan_root(plan: FFTPlan | PlanNode) -> PlanNode:
    if isinstance(plan, FFTPlan):
        return plan.root
    if is_plan_node(plan):
        return plan
    raise TypeError(f"expected FFTPlan or plan node, got {type(plan).__name__}")


def wrap_plan(
    plan: FFTPlan | PlanNode,
    request: FFTPlanRequest | None = None,
    *,
    source: str = "manual",
    estimated_cost: float | None = None,
    tags: Mapping[str, Any] | None = None,
) -> FFTPlan:
    if isinstance(plan, FFTPlan):
        return plan
    return FFTPlan(
        root=plan,
        request=request,
        source=source,
        estimated_cost=estimated_cost,
        tags={} if tags is None else tags,
    )


__all__ = [
    "CTLeafPlan",
    "DirectDFTPlan",
    "FFTDecompositionSpec",
    "FFTPlan",
    "FFTPlanRequest",
    "FourStepPlan",
    "LeafPlan",
    "PLAN_SCHEMA_VERSION",
    "PlanNode",
    "StockhamPlan",
    "get_plan_root",
    "is_plan_node",
    "wrap_plan",
]
