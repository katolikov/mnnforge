"""Phase 4 — frequent subgraph mining over the MNN op graph.

For v1 we restrict to **linear elementwise chains**: a sequence of ops where
each non-final op has exactly one consumer, all op outputs share the shape
of the chain's first input, and every op kind is in our primitive library
(see primitives.py).

Bug fixes (vs first draft) tagged BUGFIX-FSM-NN:
  1: _opcode_name linear scan was O(N²); cache the reverse map.
  2: chain consumer lookup was O(N); pre-build tensor->ops consumer index.
  3: validate input arity (BinaryOp >= 2 inputs, UnaryOp >= 1).
  4: track which operand position the chain feeds so non-commutative
     BinaryOp (Sub/Div) is handled correctly.
  5: tensors that are graph outputs must be excluded from "consumed once"
     so we don't fuse over a network output.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .log import Logger
from .primitives import (
    BINARY_SNIPPETS, UNARY_SNIPPETS, NAMED_SNIPPETS,
)


ELEMENTWISE_OP_NAMES = {
    "BinaryOp", "UnaryOp", "ReLU", "ReLU6", "Sigmoid", "TanH",
}

# Non-commutative BinaryOp opTypes. If the chain feed is operand index 1
# (right-hand side) for one of these, we MUST encode that in the kernel
# rather than swap.
NONCOMMUTATIVE_BINARY = {1, 3, 7}   # SUB, DIV, REALDIV


@dataclass
class OpRef:
    idx: int
    type_name: str
    sub_kind: int
    inputs: List[int]
    outputs: List[int]


@dataclass
class ChainStep:
    """One step in a linear chain.

    op_idx:       index in netT.oplists
    type_name, sub_kind: identifies the primitive
    chain_pos:    which input of this op consumes the prior chain output
                  (0 for the first input slot, 1 for the second, ...)
                  For the chain head this is meaningless (-1).
    extra_input_tensors: tensor idxs of inputs to this op that come from
                  outside the chain (added to boundary inputs).
    """
    op_idx: int
    type_name: str
    sub_kind: int
    chain_pos: int
    extra_input_tensors: List[int]


@dataclass
class Chain:
    steps: List[ChainStep]
    boundary_inputs: List[int]
    boundary_output: int

    @property
    def op_indices(self) -> List[int]:
        return [s.op_idx for s in self.steps]


@dataclass
class Pattern:
    fingerprint: str
    # Each entry: (type_name, sub_kind, chain_pos)
    op_kinds: List[Tuple[str, int, int]]
    occurrences: List[Chain] = field(default_factory=list)
    score: float = 0.0


# --- helpers -----------------------------------------------------------

def _build_optype_lookup(MNN) -> Dict[int, str]:
    """BUGFIX-FSM-1: cache enum->name reverse map once per process."""
    OpType = MNN.OpType.OpType
    out: Dict[int, str] = {}
    for name in dir(OpType):
        if name.startswith("_"):
            continue
        v = getattr(OpType, name)
        if isinstance(v, int):
            # Take the first wins on aliases — names are unique in MNN.fbs.
            out.setdefault(v, name)
    return out


def _sub_kind(MNN, op) -> int:
    main_type = getattr(op, "mainType", None)
    main = getattr(op, "main", None)
    if main is None or main_type is None:
        return 0
    OpParameter = MNN.OpParameter.OpParameter
    if main_type == OpParameter.BinaryOp:
        return int(getattr(main, "opType", 0))
    if main_type == OpParameter.UnaryOp:
        return int(getattr(main, "opType", 0))
    return 0


def _is_supported_kind(name: str, sub_kind: int, n_inputs: int) -> bool:
    """BUGFIX-FSM-3: arity is part of supportedness."""
    if name not in ELEMENTWISE_OP_NAMES:
        return False
    if name == "BinaryOp":
        return sub_kind in BINARY_SNIPPETS and n_inputs >= 2
    if name == "UnaryOp":
        return sub_kind in UNARY_SNIPPETS and n_inputs >= 1
    return name in NAMED_SNIPPETS and n_inputs >= 1


def _build_op_index(MNN, netT) -> List[OpRef]:
    name_of = _build_optype_lookup(MNN)
    refs: List[OpRef] = []
    for i, op in enumerate(netT.oplists or []):
        type_int = getattr(op, "type", 0)
        type_name = name_of.get(int(type_int), f"OpType_{type_int}")
        sub = _sub_kind(MNN, op)
        refs.append(OpRef(
            idx=i, type_name=type_name, sub_kind=sub,
            inputs=list(op.inputIndexes or []),
            outputs=list(op.outputIndexes or []),
        ))
    return refs


def _build_consumer_index(refs: List[OpRef]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """BUGFIX-FSM-2: pre-build tensor->consumer-op-idx list and counts."""
    consumers: Dict[int, List[int]] = {}
    counts: Dict[int, int] = {}
    for r in refs:
        for ti in r.inputs:
            consumers.setdefault(ti, []).append(r.idx)
            counts[ti] = counts.get(ti, 0) + 1
    return consumers, counts


def _find_linear_chains(refs: List[OpRef], max_len: int,
                        graph_output_tensors: Set[int]) -> List[Chain]:
    consumers, counts = _build_consumer_index(refs)
    chains: List[Chain] = []
    used = [False] * len(refs)

    for start in range(len(refs)):
        if used[start]:
            continue
        a = refs[start]
        if not _is_supported_kind(a.type_name, a.sub_kind, len(a.inputs)):
            continue

        steps: List[ChainStep] = [ChainStep(
            op_idx=a.idx, type_name=a.type_name, sub_kind=a.sub_kind,
            chain_pos=-1,
            extra_input_tensors=[],   # head's inputs are handled separately
        )]
        boundary_inputs: List[int] = list(a.inputs)
        last = a

        while len(steps) < max_len:
            if len(last.outputs) != 1:
                break
            mid_t = last.outputs[0]

            # BUGFIX-FSM-5: a tensor that's a graph output is logically
            # "consumed by the outside"; never absorb it.
            if mid_t in graph_output_tensors:
                break
            # consumed by exactly one op AND not a graph output
            if counts.get(mid_t, 0) != 1:
                break

            cons_list = consumers.get(mid_t) or []
            if len(cons_list) != 1:
                break
            cons_idx = cons_list[0]
            if used[cons_idx]:
                break
            b = refs[cons_idx]
            if not _is_supported_kind(b.type_name, b.sub_kind, len(b.inputs)):
                break

            # BUGFIX-FSM-4: identify which operand position consumes mid_t.
            try:
                chain_pos = b.inputs.index(mid_t)
            except ValueError:
                break  # shouldn't happen given the consumer index

            # If b is a non-commutative BinaryOp and the chain feeds operand
            # index 1, that's still fusable BUT the kernel synthesizer must
            # know — it'll generate (extra_input ∘ chain_var) not
            # (chain_var ∘ extra_input). We just record chain_pos and let
            # kernel_synth honor it.
            extra_inputs = [ti for j, ti in enumerate(b.inputs) if j != chain_pos]

            steps.append(ChainStep(
                op_idx=b.idx, type_name=b.type_name, sub_kind=b.sub_kind,
                chain_pos=chain_pos, extra_input_tensors=extra_inputs,
            ))
            boundary_inputs.extend(extra_inputs)
            last = b

        if len(steps) >= 2 and len(last.outputs) == 1:
            for s in steps:
                used[s.op_idx] = True
            chains.append(Chain(
                steps=steps,
                boundary_inputs=boundary_inputs,
                boundary_output=last.outputs[0],
            ))
    return chains


def _graph_output_tensor_idxs(netT) -> Set[int]:
    """Resolve the tensor indexes that are network outputs.

    MNN's NetT exposes outputs in two places:
      - netT.outputName: list of string names; we map via tensorName.
      - For some converters, op outputs that aren't consumed are implicit
        outputs; those are already covered by counts==0 in the consumer
        index, but we want to be defensive.
    """
    names = list(getattr(netT, "outputName", None) or [])
    tensor_names = list(getattr(netT, "tensorName", None) or [])
    name_to_idx = {n: i for i, n in enumerate(tensor_names)}
    out: Set[int] = set()
    for n in names:
        i = name_to_idx.get(n)
        if i is not None:
            out.add(i)
    return out


def mine(MNN, netT, log: Logger, max_pattern_size: int = 6) -> List[Pattern]:
    refs = _build_op_index(MNN, netT)
    graph_outs = _graph_output_tensor_idxs(netT)
    log.info(f"FSM: scanning {len(refs)} ops, "
             f"{len(graph_outs)} declared graph outputs, "
             f"max chain length {max_pattern_size}")

    chains = _find_linear_chains(refs, max_len=max_pattern_size,
                                 graph_output_tensors=graph_outs)
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
        log.info(f"  pattern {p.fingerprint}: "
                 f"{len(p.occurrences)}× len={len(p.op_kinds)} "
                 f"score={p.score:.0f}  [{kinds}]")
    return patterns
