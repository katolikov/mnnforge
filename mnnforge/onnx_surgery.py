"""Phase 5 — rewrite the ONNX with custom-op nodes for fused subgraphs.

For each accepted occurrence we:
  1. Excise the chain's interior nodes from model.graph.node
  2. Introduce a single new node of op_type='MnnForge_<fp>',
     domain='com.mnnforge', inputs=boundary_inputs, outputs=[boundary_output]
  3. Carry attributes describing the chain (kernel_name, op_kinds, …) so
     the generated MNN Execution can verify what it received

When MNNConvert later processes optimized.onnx, the unknown op_type falls
through `DefaultonnxOpConverter` and becomes:
    OpType_Extra
    extra.type   = "MnnForge_<fp>"
    extra.engine = "ONNX"
    extra.attr   = [(kernel_name, …), (op_kinds, …), …]

Our generated MNN-side dispatch in FuseExecution.cpp recognizes that prefix
and routes to the matching MnnForge<Fp>Execution class.
"""
from __future__ import annotations
from typing import List, Set, Tuple

import onnx
from onnx import helper

from .log import Logger
from .onnx_fsm import Pattern


CUSTOM_DOMAIN = "com.mnnforge"


def _resolve_overlap(patterns: List[Pattern], top_n: int, log: Logger
                     ) -> List[Tuple[Pattern, int]]:
    """Higher-score patterns claim ops first. Returns list of
    (pattern, occurrence_idx) in apply order."""
    accepted = patterns[:top_n]
    claimed: Set[int] = set()
    plan: List[Tuple[Pattern, int]] = []
    for p in accepted:
        ordered = sorted(
            enumerate(p.occurrences),
            key=lambda x: x[1].onnx_indices[0],
        )
        for occ_i, occ in ordered:
            if any(i in claimed for i in occ.onnx_indices):
                log.vinfo(f"  drop occ {p.fingerprint}#{occ_i}: overlap")
                continue
            for i in occ.onnx_indices:
                claimed.add(i)
            plan.append((p, occ_i))
    return plan


def _make_custom_node(p: Pattern, occ_idx: int, occ,
                      kernel_name: str) -> onnx.NodeProto:
    """Build a single ONNX NodeProto for an accepted occurrence."""
    op_kinds_str = ";".join(f"{n}:{k}:{pos}" for n, k, pos in p.op_kinds)
    node = helper.make_node(
        op_type=f"MnnForge_{p.fingerprint}",
        inputs=list(occ.boundary_inputs),
        outputs=[occ.boundary_output],
        name=f"MnnForge_{p.fingerprint}_{occ_idx}",
        domain=CUSTOM_DOMAIN,
    )
    node.attribute.append(helper.make_attribute("kernel_name", kernel_name))
    node.attribute.append(helper.make_attribute("op_kinds", op_kinds_str))
    node.attribute.append(helper.make_attribute("fingerprint", p.fingerprint))
    return node


def rewrite_onnx(model: onnx.ModelProto, patterns: List[Pattern],
                 top_n: int, log: Logger) -> Tuple[onnx.ModelProto, int]:
    """Return (new_model, n_fused). The original model is not mutated."""
    plan = _resolve_overlap(patterns, top_n, log)
    if not plan:
        log.info("no occurrences accepted (after overlap resolution)")
        return model, 0

    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    graph = new_model.graph

    # Mark indices to remove and the new nodes to insert.
    to_remove: Set[int] = set()
    new_nodes: List[Tuple[int, onnx.NodeProto]] = []  # (insert_at_index, node)
    for p, occ_i in plan:
        occ = p.occurrences[occ_i]
        kernel_name = f"mnnforge_{p.fingerprint}"
        node = _make_custom_node(p, occ_i, occ, kernel_name)
        # Insert at the position of the chain head; remove all chain ops.
        head_idx = occ.onnx_indices[0]
        new_nodes.append((head_idx, node))
        to_remove.update(occ.onnx_indices)

    # Build the new node list preserving graph order.
    head_to_node = {head: node for head, node in new_nodes}
    rebuilt: List[onnx.NodeProto] = []
    for i, n in enumerate(graph.node):
        if i in to_remove:
            if i in head_to_node:
                rebuilt.append(head_to_node[i])
            continue
        rebuilt.append(n)

    del graph.node[:]
    graph.node.extend(rebuilt)

    # Make sure the model's opset_import has our custom domain registered.
    if not any(o.domain == CUSTOM_DOMAIN for o in new_model.opset_import):
        new_model.opset_import.append(helper.make_opsetid(CUSTOM_DOMAIN, 1))

    log.ok(f"rewrote ONNX: {len(plan)} fused, "
           f"{len(model.graph.node)} -> {len(rebuilt)} nodes")
    return new_model, len(plan)
