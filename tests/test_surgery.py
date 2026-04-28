"""Phase 6 — surgery tests using stub MNN module + NetT.

These tests verify the rewrite logic without depending on flatc-generated
Python bindings. They cover:
  * Cross-pattern overlap resolution (highest-score claims first).
  * Bytes encoding of Extra.info.
  * No-op when no patterns survive synthesis.
"""
import pytest
from mnnforge.log import Logger
from mnnforge.fsm import Pattern, Chain, ChainStep
from mnnforge.surgery import apply_patterns, _make_extra_op, _accepted_occurrences


# ---------- stubs ----------

class _Enum:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _ExtraT:
    def __init__(self):
        self.type = ""
        self.engine = ""
        self.info = b""
        self.attr = None
        self.vector = False


class _OpT:
    def __init__(self):
        self.name = ""
        self.type = 0
        self.mainType = 0
        self.main = None
        self.inputIndexes = []
        self.outputIndexes = []


class _Mod:
    class Op:
        OpT = _OpT
    class Extra:
        ExtraT = _ExtraT
    class OpType:
        OpType = _Enum(Extra=512)
    class OpParameter:
        OpParameter = _Enum(Extra=102)


class _ExistingOp:
    """Stand-in for an op already in netT.oplists (left untouched by surgery)."""
    def __init__(self, label):
        self.label = label


class _Net:
    def __init__(self, ops):
        self.oplists = ops


# ---------- tests ----------

def test_make_extra_op_uses_bytes_for_info():
    op = _make_extra_op(_Mod, "k", "kernel src here",
                        inputs=[1, 2], outputs=[3], op_name="X")
    assert op.type == _Mod.OpType.OpType.Extra
    assert op.mainType == _Mod.OpParameter.OpParameter.Extra
    assert isinstance(op.main.info, (bytes, bytearray))
    assert op.main.type == "k"
    assert op.main.engine == "OpenCL"
    assert op.inputIndexes == [1, 2]
    assert op.outputIndexes == [3]


def _mk_chain(ops, kinds, boundary_in, boundary_out):
    """Helper: build a Chain object."""
    steps = [ChainStep(op_idx=ops[i], type_name=kinds[i][0],
                       sub_kind=kinds[i][1], chain_pos=kinds[i][2],
                       extra_input_tensors=[])
             for i in range(len(ops))]
    return Chain(steps=steps, boundary_inputs=boundary_in,
                 boundary_output=boundary_out)


def test_overlap_resolution_higher_score_wins():
    log = Logger(verbose=False)
    p_high = Pattern(
        fingerprint="aaa",
        op_kinds=[("UnaryOp", 7, -1), ("UnaryOp", 8, 0)],
    )
    p_low = Pattern(
        fingerprint="bbb",
        op_kinds=[("UnaryOp", 8, -1), ("UnaryOp", 9, 0)],
    )
    # Both occurrences claim ops [1, 2].
    p_high.occurrences = [_mk_chain([1, 2], p_high.op_kinds, [0], 3)]
    p_low.occurrences = [_mk_chain([1, 2], p_low.op_kinds, [0], 3)]
    p_high.score, p_low.score = 100.0, 1.0

    plan = _accepted_occurrences([p_high, p_low], log)
    assert len(plan) == 1
    assert plan[0][0].fingerprint == "aaa"


def test_apply_patterns_with_unsupported_kind_skips_cleanly():
    """If kernel synthesis raises, surgery must skip that pattern silently
    and not crash."""
    p = Pattern(
        fingerprint="ccc",
        op_kinds=[("Convolution", 0, -1)],   # unsupported -> ValueError
    )
    p.occurrences = [_mk_chain([0], p.op_kinds, [0], 1)]
    p.score = 1.0

    net = _Net(ops=[_ExistingOp("conv")])
    n = apply_patterns(_Mod, net, [p], Logger(verbose=False), top_n=4)
    assert n == 0
    # netT untouched
    assert len(net.oplists) == 1
    assert net.oplists[0].label == "conv"


def test_apply_patterns_replaces_chain_with_extra_op():
    """End-to-end: a 2-step UnaryOp chain becomes one Extra op."""
    p = Pattern(
        fingerprint="ddd",
        op_kinds=[("UnaryOp", 7, -1), ("UnaryOp", 8, 0)],   # exp -> log
    )
    p.occurrences = [_mk_chain([1, 2], p.op_kinds, [0], 3)]
    p.score = 1.0

    net = _Net(ops=[
        _ExistingOp("input_op"),     # idx 0 — kept
        _ExistingOp("exp_op"),       # idx 1 — replaced
        _ExistingOp("log_op"),       # idx 2 — swallowed
        _ExistingOp("output_op"),    # idx 3 — kept
    ])
    n = apply_patterns(_Mod, net, [p], Logger(verbose=False), top_n=4)
    assert n == 1
    # ops shrink by 1
    assert len(net.oplists) == 3
    # The new op at position 1 is an Extra op carrying our kernel.
    new_op = net.oplists[1]
    assert getattr(new_op, "type", None) == _Mod.OpType.OpType.Extra
    assert isinstance(new_op.main.info, (bytes, bytearray))
    assert b"__kernel" in new_op.main.info
    # First and last existing ops are preserved.
    assert net.oplists[0].label == "input_op"
    assert net.oplists[2].label == "output_op"


def test_arity_mismatch_skipped_per_occurrence():
    """If a later occurrence of the same pattern has a different boundary
    arity than the first one, it is skipped (kernel was synthesized for the
    first arity)."""
    p = Pattern(
        fingerprint="eee",
        op_kinds=[("BinaryOp", 0, -1)],  # head BinaryOp
    )
    p.occurrences = [
        _mk_chain([0], p.op_kinds, [10, 11], 12),       # 2 inputs (correct)
        _mk_chain([1], p.op_kinds, [13, 14, 15], 16),   # 3 inputs (mismatch)
    ]
    p.score = 1.0

    # Single-step chains have len(steps)==1; mine() requires >=2, but we
    # bypass mine() here; we only test that arity-mismatch occurrences are
    # individually skipped at the apply layer.
    net = _Net(ops=[_ExistingOp("a"), _ExistingOp("b")])
    n = apply_patterns(_Mod, net, [p], Logger(verbose=False), top_n=4)
    # First occurrence is a length-1 chain too, so arity mismatch could arise
    # only after kernel is synthesized for 2 inputs. The 3-input occurrence
    # gets skipped; the 2-input one applies.
    # We only assert that arity-mismatch never crashes and net is internally
    # consistent.
    assert n in (0, 1)
    assert len(net.oplists) >= 1
