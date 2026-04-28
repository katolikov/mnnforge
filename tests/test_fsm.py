"""Phase 4 — FSM tests using stub MNN/netT objects.

We don't depend on flatc-generated MNN bindings here; instead we build
duck-typed stubs that match the small surface of MNN that fsm.py uses
(OpType.OpType, OpParameter.OpParameter, op.type/main/mainType/inputIndexes/
outputIndexes, netT.oplists/tensorName/outputName).
"""
import pytest

from mnnforge.log import Logger
from mnnforge.fsm import (
    mine, _build_op_index, _build_consumer_index,
    NONCOMMUTATIVE_BINARY,
)


# ------------------------- stubs -------------------------

class _Enum:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Mod:
    """MNN module stub: exposes OpType.OpType and OpParameter.OpParameter."""

    class OpType:
        # OpType numbering is deliberately arbitrary in the stub. We only
        # need fsm.py to map int -> name; the names must match
        # ELEMENTWISE_OP_NAMES in fsm.py for things to be eligible.
        OpType = _Enum(
            BinaryOp=1, UnaryOp=2, ReLU=3, ReLU6=4, Sigmoid=5, TanH=6,
            Const=10, Convolution=11, Extra=512,
        )

    class OpParameter:
        OpParameter = _Enum(
            BinaryOp=100, UnaryOp=101, Extra=102, NONE=0,
        )


class _Op:
    def __init__(self, type_, inputs, outputs,
                 main_type=0, sub_kind=0):
        self.type = type_
        self.inputIndexes = list(inputs)
        self.outputIndexes = list(outputs)
        if main_type:
            class _M:
                def __init__(self, k):
                    self.opType = k
            self.main = _M(sub_kind)
            self.mainType = main_type
        else:
            self.main = None
            self.mainType = 0


class _NetT:
    def __init__(self, ops, tensors, outputs):
        self.oplists = ops
        self.tensorName = tensors
        self.outputName = outputs


# ------------------------- helpers -------------------------

def _binary(sub, ins, outs):
    return _Op(_Mod.OpType.OpType.BinaryOp, ins, outs,
               main_type=_Mod.OpParameter.OpParameter.BinaryOp,
               sub_kind=sub)


def _unary(sub, ins, outs):
    return _Op(_Mod.OpType.OpType.UnaryOp, ins, outs,
               main_type=_Mod.OpParameter.OpParameter.UnaryOp,
               sub_kind=sub)


def _relu(ins, outs):
    return _Op(_Mod.OpType.OpType.ReLU, ins, outs)


# ------------------------- tests -------------------------

def test_op_index_resolves_name():
    net = _NetT(
        ops=[_binary(0, [0, 1], [2]), _unary(7, [2], [3])],
        tensors=["t0", "t1", "t2", "t3"],
        outputs=["t3"],
    )
    refs = _build_op_index(_Mod, net)
    assert [r.type_name for r in refs] == ["BinaryOp", "UnaryOp"]
    assert [r.sub_kind for r in refs] == [0, 7]


def test_consumer_index_counts():
    net = _NetT(
        ops=[
            _binary(0, [0, 1], [2]),   # consumes t0, t1
            _unary(7, [2], [3]),       # consumes t2
            _unary(7, [2], [4]),       # also consumes t2 -> t2 is used twice
        ],
        tensors=["t0", "t1", "t2", "t3", "t4"],
        outputs=["t3", "t4"],
    )
    refs = _build_op_index(_Mod, net)
    consumers, counts = _build_consumer_index(refs)
    assert counts[2] == 2
    assert counts[0] == 1 and counts[1] == 1


def test_simple_chain_mined():
    """Add(t0,t1) -> Sigmoid -> graph_output"""
    net = _NetT(
        ops=[
            _binary(0, [0, 1], [2]),   # Add
            _unary(23, [2], [3]),      # Sigmoid
        ],
        tensors=["t0", "t1", "t2", "t3"],
        outputs=["t3"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    assert len(patterns) == 1
    p = patterns[0]
    assert len(p.op_kinds) == 2
    assert p.op_kinds[0][0] == "BinaryOp"
    assert p.op_kinds[1][0] == "UnaryOp"
    assert len(p.occurrences) == 1
    occ = p.occurrences[0]
    # boundary inputs are t0, t1; output is t3
    assert occ.boundary_inputs == [0, 1]
    assert occ.boundary_output == 3


def test_chain_not_extended_through_graph_output():
    """If a tensor is a graph output it must NOT be absorbed."""
    net = _NetT(
        ops=[
            _binary(0, [0, 1], [2]),    # Add -> t2
            _unary(7, [2], [3]),        # Exp -> t3   (t2 is also graph output)
        ],
        tensors=["t0", "t1", "t2", "t3"],
        outputs=["t2", "t3"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    # The chain Add->Exp must NOT swallow t2: t2 is a graph output and
    # must be preserved. So the chain should not form across the boundary.
    # Result: zero accepted chains (each op alone is length-1).
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_chain_breaks_on_multi_consumer():
    """Mid-tensor consumed twice cannot be fused away."""
    net = _NetT(
        ops=[
            _binary(0, [0, 1], [2]),
            _unary(7, [2], [3]),
            _unary(8, [2], [4]),     # second consumer of t2
        ],
        tensors=["t0", "t1", "t2", "t3", "t4"],
        outputs=["t3", "t4"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_chain_pos_recorded_for_noncommutative():
    """Sub with chain on RHS: chain_pos should be 1."""
    # t0 -> Sigmoid -> t1; then BinaryOp Sub with inputs (t2, t1) -> t3
    # The chain output t1 enters Sub at operand position 1.
    net = _NetT(
        ops=[
            _unary(23, [0], [1]),                  # Sigmoid head
            _binary(1, [2, 1], [3]),               # Sub, chain at pos 1
        ],
        tensors=["t0", "t1", "t2", "t3"],
        outputs=["t3"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    assert len(patterns) == 1
    kinds = patterns[0].op_kinds
    assert kinds[0] == ("UnaryOp", 23, -1)
    assert kinds[1] == ("BinaryOp", 1, 1)
    assert 1 in NONCOMMUTATIVE_BINARY  # Sub indeed non-commutative


def test_chain_pattern_count_repeats_across_occurrences():
    """Two identical chains in a graph should land in the same pattern."""
    net = _NetT(
        ops=[
            _binary(0, [0, 1], [2]),  _unary(23, [2], [3]),
            _binary(0, [4, 5], [6]),  _unary(23, [6], [7]),
        ],
        tensors=[f"t{i}" for i in range(8)],
        outputs=["t3", "t7"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    assert len(patterns) == 1
    assert len(patterns[0].occurrences) == 2


def test_unsupported_op_breaks_chain():
    """A Convolution between two unaries must split the chain."""
    net = _NetT(
        ops=[
            _unary(7, [0], [1]),
            _Op(_Mod.OpType.OpType.Convolution, [1], [2]),
            _unary(8, [2], [3]),
        ],
        tensors=[f"t{i}" for i in range(4)],
        outputs=["t3"],
    )
    patterns = mine(_Mod, net, Logger(verbose=False))
    # Only length-1 candidates exist; min chain length is 2 -> no patterns.
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_max_pattern_size_caps_chain():
    """A long chain of ReLUs should be capped at max_pattern_size."""
    ops = [_relu([i], [i + 1]) for i in range(10)]
    tensors = [f"t{i}" for i in range(11)]
    net = _NetT(ops=ops, tensors=tensors, outputs=["t10"])
    patterns = mine(_Mod, net, Logger(verbose=False), max_pattern_size=3)
    # Chain capping doesn't necessarily limit each output to length 3 only —
    # FSM just stops extending past max_len, then starts a new chain.
    for p in patterns:
        for occ in p.occurrences:
            assert len(occ.steps) <= 3
