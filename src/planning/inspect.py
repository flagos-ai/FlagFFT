from __future__ import annotations

from typing import Any

from .factorization import estimate_leaf_smem_bytes, lane_block_for
from .types import DirectDFTPlan, FFTPlan, FourStepPlan, LeafPlan, PlanNode, StockhamPlan, get_plan_root


def iter_plan_nodes(plan: FFTPlan | PlanNode) -> list[PlanNode]:
    root = get_plan_root(plan)
    nodes = [root]
    if isinstance(root, FourStepPlan):
        nodes.extend(iter_plan_nodes(root.row_plan))
        nodes.extend(iter_plan_nodes(root.col_plan))
    return nodes


def plan_depth(plan: FFTPlan | PlanNode) -> int:
    root = get_plan_root(plan)
    if isinstance(root, FourStepPlan):
        return 1 + max(plan_depth(root.row_plan), plan_depth(root.col_plan))
    return 1


def collect_leaf_plans(plan: FFTPlan | PlanNode) -> list[LeafPlan]:
    return [node for node in iter_plan_nodes(plan) if isinstance(node, LeafPlan)]


def unique_leaf_plans(plan: FFTPlan | PlanNode) -> list[LeafPlan]:
    unique: dict[tuple[int, tuple[int, ...], int], LeafPlan] = {}
    for leaf in collect_leaf_plans(plan):
        key = (leaf.length, leaf.factors, leaf.lanes)
        unique.setdefault(key, leaf)
    return sorted(unique.values(), key=lambda leaf: (leaf.length, leaf.factors, leaf.lanes))


def max_leaf_smem_bytes(plan: FFTPlan | PlanNode) -> int:
    leaf_plans = collect_leaf_plans(plan)
    if not leaf_plans:
        return 0
    return max(estimate_leaf_smem_bytes(leaf.length, leaf.factors) for leaf in leaf_plans)


def _describe_node(node: PlanNode) -> str:
    if isinstance(node, LeafPlan):
        smem_bytes = estimate_leaf_smem_bytes(node.length, node.factors)
        lane_block = lane_block_for(node.lanes)
        return (
            f"ct_leaf(n={node.length}, factors={node.factors}, lanes={node.lanes}/{lane_block}, "
            f"warps={node.num_warps}, smem={smem_bytes}B)"
        )
    if isinstance(node, DirectDFTPlan):
        return f"direct_dft(n={node.length}, impl={node.impl})"
    if isinstance(node, StockhamPlan):
        return f"stockham_autosort(n={node.length}, factors={node.factors}, stages={len(node.stages)})"
    if isinstance(node, FourStepPlan):
        return (
            f"four_step(n={node.length}, n1={node.n1}, n2={node.n2}, "
            f"depth={plan_depth(node)})"
        )
    raise TypeError(f"unsupported plan node type: {type(node).__name__}")


def describe_fft_plan(plan: FFTPlan | PlanNode) -> str:
    return _describe_node(get_plan_root(plan))


def format_plan_tree(plan: FFTPlan | PlanNode, *, indent: str = "  ") -> str:
    def visit(node: PlanNode, depth: int, label: str | None = None) -> list[str]:
        prefix = indent * depth
        head = _describe_node(node)
        if label is not None:
            head = f"{label}: {head}"
        lines = [prefix + head]
        if isinstance(node, FourStepPlan):
            lines.extend(visit(node.row_plan, depth + 1, "row"))
            lines.extend(visit(node.col_plan, depth + 1, "col"))
        return lines

    return "\n".join(visit(get_plan_root(plan), 0))


def plan_path(plan: FFTPlan | PlanNode) -> dict[str, Any]:
    node = get_plan_root(plan)
    if isinstance(node, FourStepPlan):
        return {
            "kind": node.kind,
            "length": node.length,
            "n1": node.n1,
            "n2": node.n2,
            "row": plan_path(node.row_plan),
            "col": plan_path(node.col_plan),
        }
    return {"kind": node.kind, "length": node.length}


def plan_path_string(plan: FFTPlan | PlanNode) -> str:
    node = get_plan_root(plan)
    if isinstance(node, FourStepPlan):
        return f"four_step(row={plan_path_string(node.row_plan)}, col={plan_path_string(node.col_plan)})"
    return node.kind


__all__ = [
    "collect_leaf_plans",
    "describe_fft_plan",
    "format_plan_tree",
    "iter_plan_nodes",
    "max_leaf_smem_bytes",
    "plan_depth",
    "plan_path",
    "plan_path_string",
    "unique_leaf_plans",
]
