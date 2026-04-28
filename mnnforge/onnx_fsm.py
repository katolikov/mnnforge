"""Phase 2 — frequent subgraph mining on the ONNX graph.

We work directly on `onnx.ModelProto`. Patterns are linear chains of
elementwise ops; each chain becomes a custom op node in Phase 5.

Pattern shape produced is identical to the original `fsm.py` (Pattern,
Chain, ChainStep dataclasses) so kernel_synth.py works unchanged.

Bug-defended:
  * graph outputs and graph inputs are immutable boundaries
  * tensors with multiple consumers can't be absorbed
  * non-commutative BinaryOp (Sub, Div) records `chain_pos` in the
    consumer so kernel_synth honors operand order
  * empty / single-input nodes are filtered out
  * nodes with `domain != ""` are skipped (they're already custom)
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import onnx

from .log import Logger


# Map ONNX op_type → ("BinaryOp"|"UnaryOp"|named, sub_kind enum-int)
# sub_kind values mirror MNN's BinaryOpOperation / UnaryOpOperation
# enums in schema/default/MNN.fbs (sufficient subset for v1).
ONNX_TO_KIND: Dict[str, Tuple[str, int]] = {
    # BinaryOp
    "Add":   ("BinaryOp", 0),
    "Sub":   ("BinaryOp", 1),
    "Mul":   ("BinaryOp", 2),
    "Div":   ("BinaryOp", 3),
    # UnaryOp
    "Abs":         ("UnaryOp", 0),
    "Neg":         ("UnaryOp", 1),
    # ONNX has no "Square"; x*x is the BinaryOp Mul path.
    "Sqrt":        ("UnaryOp", 5),
    # ONNX "Reciprocal"
    "Reciprocal":  ("UnaryOp", 15),
    "Exp":         ("UnaryOp", 7),
    "Log":         ("UnaryOp", 8),
    "Tanh":        ("UnaryOp", 18),
    "Sigmoid":     ("UnaryOp", 23),
    "Gelu":        ("UnaryOp", 31),   # ONNX 20+; older models use Erf-based forms.
    # named scalar-shaped activations
    "Relu":        ("ReLU",  0),
    # ReLU6 isn't a stock ONNX op; it'd appear as Clip(0,6).
}

# Non-commutative binary ops. If the chain output feeds operand idx 1, we
# encode (operand0 - chain_var) instead of (chain_var - operand0).
NONCOMMUTATIVE_BINARY = {1, 3}    # SUB, DIV (and REALDIV which we don't use)

ELEMENTWISE_OP_NAMES = {"BinaryOp", "UnaryOp", "ReLU"}


@dataclass
class ChainStep:
    """One step in a chain.

    onnx_idx: index into model.graph.node
    type_name, sub_kind: see kernel_synth.synthesize_kernel
    chain_pos: -1 for the head; otherwise which input slot of this op
               consumes the chain output (0 or 1)
    extra_input_tensors: tensor names this op consumes that are NOT the
               chain output (these become boundary inputs of the chain)
    """
    onnx_idx: int
    type_name: str
    sub_kind: int
    chain_pos: int
    extra_input_tensors: List[str]


@dataclass
class Chain:
    steps: List[ChainStep]
    boundary_inputs: List[str]
    boundary_output: str

    @property
    def onnx_indices(self) -> List[int]:
        return [s.onnx_idx for s in self.steps]


@dataclass
class Pattern:
    fingerprint: str
    op_kinds: List[Tuple[str, int, int]]   # (type_name, sub_kind, chain_pos)
    occurrences: List[Chain] = field(default_factory=list)
    score: float = 0.0


# ------------- internals -------------

def _classify(node: onnx.NodeProto) -> Optional[Tuple[str, int]]:
    """Classify an ONNX node into our (type_name, sub_kind) pair.
    Returns None if the node is not a supported elementwise primitive."""
    if node.domain not in ("", "ai.onnx"):
        return None     # custom domain: leave alone
    return ONNX_TO_KIND.get(node.op_type)


def _build_consumer_index(graph: onnx.GraphProto
                          ) -> Tuple[Dict[str, List[int]], Dict[str, int],
                                     Dict[str, int]]:
    """Return:
       consumers: tensor_name -> list of node-idx that consume it
       counts:    tensor_name -> consumer count
       producer:  tensor_name -> single producer node-idx (or absent)
    """
    consumers: Dict[str, List[int]] = {}
    counts: Dict[str, int] = {}
    producer: Dict[str, int] = {}
    for i, n in enumerate(graph.node):
        for t in n.input:
            if not t:
                continue
            consumers.setdefault(t, []).append(i)
            counts[t] = counts.get(t, 0) + 1
        for t in n.output:
            if t:
                producer[t] = i
    return consumers, counts, producer


def _graph_boundary_tensor_names(graph: onnx.GraphProto) -> Set[str]:
    """Tensors that are graph inputs OR graph outputs OR initializers — must
    not be absorbed mid-chain."""
    out: Set[str] = set()
    out.update(i.name for i in graph.input)
    out.update(o.name for o in graph.output)
    return out


def _find_chains(graph: onnx.GraphProto, max_len: int) -> List[Chain]:
    consumers, counts, _producer = _build_consumer_index(graph)
    output_names = {o.name for o in graph.output}
    used = [False] * len(graph.node)
    chains: List[Chain] = []

    for start in range(len(graph.node)):
        if used[start]:
            continue
        head_node = graph.node[start]
        head_kind = _classify(head_node)
        if head_kind is None:
            continue
        # filter empty inputs (ONNX allows missing optional inputs as "")
        head_inputs = [t for t in head_node.input if t]
        if head_kind[0] == "BinaryOp" and len(head_inputs) < 2:
            continue
        if head_kind[0] != "BinaryOp" and len(head_inputs) < 1:
            continue
        # initialize chain
        steps: List[ChainStep] = [ChainStep(
            onnx_idx=start, type_name=head_kind[0], sub_kind=head_kind[1],
            chain_pos=-1, extra_input_tensors=[],
        )]
        boundary_inputs: List[str] = list(head_inputs)
        last_node = head_node
        last_idx = start

        while len(steps) < max_len:
            outs = [t for t in last_node.output if t]
            if len(outs) != 1:
                break
            mid_t = outs[0]
            if mid_t in output_names:
                break    # graph output — stop, never absorb
            if counts.get(mid_t, 0) != 1:
                break    # multi-consumer or unconsumed
            cons_list = consumers.get(mid_t, [])
            if len(cons_list) != 1:
                break
            cons_idx = cons_list[0]
            if used[cons_idx]:
                break
            cons = graph.node[cons_idx]
            cons_kind = _classify(cons)
            if cons_kind is None:
                break
            cons_inputs = [t for t in cons.input if t]
            if cons_kind[0] == "BinaryOp" and len(cons_inputs) < 2:
                break
            if cons_kind[0] != "BinaryOp" and len(cons_inputs) < 1:
                break
            try:
                chain_pos = cons_inputs.index(mid_t)
            except ValueError:
                break
            extra = [t for j, t in enumerate(cons_inputs) if j != chain_pos]

            steps.append(ChainStep(
                onnx_idx=cons_idx, type_name=cons_kind[0],
                sub_kind=cons_kind[1], chain_pos=chain_pos,
                extra_input_tensors=extra,
            ))
            boundary_inputs.extend(extra)
            last_node = cons
            last_idx = cons_idx

        if len(steps) >= 2:
            for s in steps:
                used[s.onnx_idx] = True
            outs = [t for t in last_node.output if t]
            chains.append(Chain(
                steps=steps,
                boundary_inputs=boundary_inputs,
                boundary_output=outs[0],
            ))
    return chains


def mine(model: onnx.ModelProto, log: Logger,
         max_pattern_size: int = 6) -> List[Pattern]:
    refs = list(model.graph.node)
    log.info(f"FSM(ONNX): {len(refs)} nodes; "
             f"max chain length {max_pattern_size}")

    chains = _find_chains(model.graph, max_len=max_pattern_size)
    log.info(f"FSM: discovered {len(chains)} maximal linear elementwise chains")

    groups: Dict[str, Pattern] = {}
    for c in chains:
        kinds = [(s.type_name, s.sub_kind, s.chain_pos) for s in c.steps]
        sig = "->".join(f"{n}:{k}:{p}" for n, k, p in kinds)
        fp = hashlib.sha1(sig.encode()).hexdigest()[:8]
        if fp not in groups:
            groups[fp] = Pattern(fingerprint=fp, op_kinds=kinds)
        groups[fp].occurrences.append(c)

    for p in groups.values():
        p.score = float(len(p.occurrences) * (len(p.op_kinds) - 1))

    patterns = sorted(groups.values(), key=lambda p: p.score, reverse=True)
    for p in patterns:
        kinds = " -> ".join(
            f"{n}({k},pos={pos})" for n, k, pos in p.op_kinds
        )
        log.info(f"  pattern {p.fingerprint}: {len(p.occurrences)}× "
                 f"len={len(p.op_kinds)} score={p.score:.0f}  [{kinds}]")
    return patterns
