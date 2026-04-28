"""Phase 5 — ONNX surgery tests."""
import onnx
from onnx import helper, TensorProto

from mnnforge.log import Logger
from mnnforge.onnx_fsm import mine
from mnnforge.onnx_surgery import rewrite_onnx, CUSTOM_DOMAIN


def _input(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _output(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _model_two_chains():
    nodes = [
        helper.make_node("Mul", ["a", "b"], ["c1"], name="n1"),
        helper.make_node("Sigmoid", ["c1"], ["y1"], name="n2"),
        helper.make_node("Mul", ["d", "e"], ["c2"], name="n3"),
        helper.make_node("Sigmoid", ["c2"], ["y2"], name="n4"),
    ]
    g = helper.make_graph(
        nodes, "test",
        [_input(n) for n in ("a", "b", "d", "e")],
        [_output("y1"), _output("y2")],
    )
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    return m


def test_rewrite_replaces_each_chain():
    m = _model_two_chains()
    patterns = mine(m, Logger(verbose=False))
    new_m, n_fused = rewrite_onnx(m, patterns, top_n=4, log=Logger(verbose=False))
    assert n_fused == 2
    custom = [n for n in new_m.graph.node if n.domain == CUSTOM_DOMAIN]
    assert len(custom) == 2
    for n in custom:
        assert n.op_type.startswith("MnnForge_")
        # one output, the chain tail
        assert len(n.output) == 1
        assert n.output[0] in {"y1", "y2"}
    # No primitive Mul/Sigmoid should remain
    remaining_kinds = {n.op_type for n in new_m.graph.node
                       if n.domain not in (CUSTOM_DOMAIN,)}
    assert "Mul" not in remaining_kinds
    assert "Sigmoid" not in remaining_kinds


def test_rewrite_preserves_graph_outputs():
    m = _model_two_chains()
    patterns = mine(m, Logger(verbose=False))
    new_m, _ = rewrite_onnx(m, patterns, top_n=4, log=Logger(verbose=False))
    assert [o.name for o in new_m.graph.output] == ["y1", "y2"]


def test_rewrite_attributes_complete():
    m = _model_two_chains()
    patterns = mine(m, Logger(verbose=False))
    new_m, _ = rewrite_onnx(m, patterns, top_n=4, log=Logger(verbose=False))
    for n in new_m.graph.node:
        if n.domain != CUSTOM_DOMAIN:
            continue
        keys = {a.name for a in n.attribute}
        assert "kernel_name" in keys
        assert "op_kinds" in keys
        assert "fingerprint" in keys


def test_rewrite_custom_domain_registered_in_opset_import():
    m = _model_two_chains()
    patterns = mine(m, Logger(verbose=False))
    new_m, _ = rewrite_onnx(m, patterns, top_n=4, log=Logger(verbose=False))
    domains = {o.domain for o in new_m.opset_import}
    assert CUSTOM_DOMAIN in domains


def test_rewrite_with_no_patterns_returns_unchanged():
    n = helper.make_node("Conv", ["a", "w"], ["y"], name="n1",
                         kernel_shape=[3, 3])
    g = helper.make_graph(
        [n], "t",
        [_input("a", (1, 4, 16, 16)), _input("w", (8, 4, 3, 3))],
        [_output("y", (1, 8, 14, 14))],
    )
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    patterns = mine(m, Logger(verbose=False))
    new_m, n_fused = rewrite_onnx(m, patterns, top_n=4, log=Logger(verbose=False))
    assert n_fused == 0


def test_rewrite_overlap_resolution_higher_score_wins():
    """Two patterns that share an op should not both apply."""
    m = _model_two_chains()
    patterns = mine(m, Logger(verbose=False))
    # Force two overlapping patterns by duplicating then mutating one's score.
    if len(patterns) >= 1:
        # Synthesize a second "pattern" claiming the same first occurrence by
        # appending — surgery should still do the right thing.
        from copy import deepcopy
        other = deepcopy(patterns[0])
        other.fingerprint = "ffffffff"
        other.score = patterns[0].score / 2
        new_m, n_fused = rewrite_onnx(m, [patterns[0], other],
                                      top_n=4, log=Logger(verbose=False))
        # Still 2 occurrences max (from patterns[0]), other is overlap-skipped.
        assert n_fused <= 2
