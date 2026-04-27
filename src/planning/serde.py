from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .factorization import make_leaf_plan
from .types import (
    DirectDFTPlan,
    FFTPlan,
    FFTPlanRequest,
    FourStepPlan,
    LeafPlan,
    PLAN_SCHEMA_VERSION,
    PlanNode,
    StockhamPlan,
    get_plan_root,
)


def request_to_dict(request: FFTPlanRequest) -> dict[str, Any]:
    return {
        "op": request.op,
        "length": request.length,
        "dtype": request.dtype,
        "device": request.device,
        "direction": request.direction,
        "norm": request.norm,
        "batch": request.batch,
        "input_layout": request.input_layout,
        "output_order": request.output_order,
    }


def request_from_dict(data: Mapping[str, Any]) -> FFTPlanRequest:
    return FFTPlanRequest(
        op=str(data.get("op", "fft")),
        length=int(data["length"]),
        dtype=str(data.get("dtype", "complex64")),
        device=str(data.get("device", "cuda")),
        direction=str(data.get("direction", "forward")),
        norm=data.get("norm"),
        batch=None if data.get("batch") is None else int(data["batch"]),
        input_layout=str(data.get("input_layout", "contiguous")),
        output_order=str(data.get("output_order", "natural")),
    )


def node_to_dict(node: PlanNode) -> dict[str, Any]:
    if isinstance(node, LeafPlan):
        return {
            "kind": node.kind,
            "length": node.length,
            "factors": list(node.factors),
            "remainder": node.remainder,
            "lanes": node.lanes,
            "num_warps": node.num_warps,
            "generic_radices": list(node.generic_radices),
            "smem_size": node.smem_size,
        }
    if isinstance(node, DirectDFTPlan):
        return {
            "kind": node.kind,
            "length": node.length,
            "impl": node.impl,
        }
    if isinstance(node, StockhamPlan):
        return {
            "kind": node.kind,
            "length": node.length,
            "factors": list(node.factors),
            "stages": [dict(stage) for stage in node.stages],
        }
    if isinstance(node, FourStepPlan):
        return {
            "kind": node.kind,
            "length": node.length,
            "n1": node.n1,
            "n2": node.n2,
            "row": node_to_dict(node.row_plan),
            "col": node_to_dict(node.col_plan),
        }
    raise TypeError(f"unsupported plan node type: {type(node).__name__}")


def _kind_from_dict(data: Mapping[str, Any]) -> str:
    kind = data.get("kind")
    if kind is not None:
        return str(kind)
    if "root" in data:
        return "plan"
    if "split" in data or ("n1" in data and "n2" in data):
        return "four_step"
    if "factors" in data:
        return "ct_leaf"
    raise ValueError("plan node mapping must include a kind field")


def node_from_dict(data: Mapping[str, Any]) -> PlanNode:
    kind = _kind_from_dict(data)
    if kind in {"leaf", "ct_leaf"}:
        length = int(data["length"])
        factors = tuple(int(value) for value in data["factors"])
        remainder = int(data.get("remainder", 1))
        if math.prod(factors) * remainder != length:
            raise ValueError(
                f"ct_leaf factors do not match length: length={length}, factors={factors}, remainder={remainder}"
            )
        base = make_leaf_plan(length, factors, remainder)
        return LeafPlan(
            length=length,
            factors=factors,
            remainder=int(data.get("remainder", base.remainder)),
            lanes=int(data.get("lanes", base.lanes)),
            num_warps=int(data.get("num_warps", base.num_warps)),
            generic_radices=tuple(
                int(value) for value in data.get("generic_radices", base.generic_radices)
            ),
            smem_size=int(data.get("smem_size", base.smem_size)),
        )
    if kind in {"direct", "direct_dft"}:
        return DirectDFTPlan(length=int(data["length"]), impl=str(data.get("impl", "torch_matmul")))
    if kind == "stockham_autosort":
        return StockhamPlan(
            length=int(data["length"]),
            factors=tuple(int(value) for value in data.get("factors", ())),
            stages=tuple(dict(stage) for stage in data.get("stages", ())),
        )
    if kind == "four_step":
        split = data.get("split")
        if split is not None:
            n1, n2 = split
        else:
            n1, n2 = data["n1"], data["n2"]
        row = data.get("row", data.get("row_plan"))
        col = data.get("col", data.get("col_plan"))
        if row is None or col is None:
            raise ValueError("four_step plan node must include row and col child plans")
        n1 = int(n1)
        n2 = int(n2)
        length = int(data.get("length", n1 * n2))
        row_plan = node_from_dict(row)
        col_plan = node_from_dict(col)
        if length != n1 * n2:
            raise ValueError(f"four_step split [{n1}, {n2}] does not match length {length}")
        if row_plan.length != n1 or col_plan.length != n2:
            raise ValueError(
                f"four_step child length mismatch: row={row_plan.length}/{n1}, col={col_plan.length}/{n2}"
            )
        return FourStepPlan(
            length=length,
            n1=n1,
            n2=n2,
            row_plan=row_plan,
            col_plan=col_plan,
        )
    raise ValueError(f"unsupported plan node kind: {kind}")


def plan_to_dict(plan: FFTPlan | PlanNode) -> dict[str, Any]:
    if isinstance(plan, FFTPlan):
        wrapped = plan
    else:
        root = get_plan_root(plan)
        wrapped = FFTPlan(root=root)
    return {
        "schema_version": wrapped.schema_version,
        "source": wrapped.source,
        "request": request_to_dict(wrapped.request),
        "estimated_cost": wrapped.estimated_cost,
        "tags": dict(wrapped.tags),
        "root": node_to_dict(wrapped.root),
    }


def plan_from_dict(data: Mapping[str, Any]) -> FFTPlan:
    if "root" not in data:
        root = node_from_dict(data)
        return FFTPlan(root=root, request=FFTPlanRequest(length=root.length), source="manual")

    root = node_from_dict(data["root"])
    request_data = data.get("request")
    request = (
        FFTPlanRequest(length=root.length)
        if request_data is None
        else request_from_dict(request_data)
    )
    schema_version = int(data.get("schema_version", PLAN_SCHEMA_VERSION))
    if schema_version != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported FFT plan schema version {schema_version}; expected {PLAN_SCHEMA_VERSION}"
        )
    return FFTPlan(
        root=root,
        request=request,
        schema_version=schema_version,
        source=str(data.get("source", "json")),
        estimated_cost=data.get("estimated_cost"),
        tags=dict(data.get("tags", {})),
    )


def plan_to_json(plan: FFTPlan | PlanNode, *, indent: int | None = 2) -> str:
    return json.dumps(plan_to_dict(plan), indent=indent, sort_keys=True)


def plan_from_json(raw: str) -> FFTPlan:
    return plan_from_dict(json.loads(raw))


def save_fft_plan(plan: FFTPlan | PlanNode, path: str | Path) -> None:
    Path(path).write_text(plan_to_json(plan) + "\n")


def load_fft_plan(path: str | Path) -> FFTPlan:
    return plan_from_json(Path(path).read_text())


__all__ = [
    "load_fft_plan",
    "node_from_dict",
    "node_to_dict",
    "plan_from_dict",
    "plan_from_json",
    "plan_to_dict",
    "plan_to_json",
    "request_from_dict",
    "request_to_dict",
    "save_fft_plan",
]
