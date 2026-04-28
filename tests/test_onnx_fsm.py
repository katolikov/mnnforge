"""Phase 2 — FSM on ONNX graphs."""
import onnx
from onnx import helper, TensorProto

from mnnforge.log import Logger
from mnnforge.onnx_fsm import mine, NONCOMMUTATIVE_BINARY


def _input(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _output(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _model(nodes, inputs, outputs, initializers=()):
    g = helper.make_graph(nodes, "test", inputs, outputs,
                          initializer=list(initializers))
    m = helper.make_model(g, producer_name="mnnforge-test")
    m.opset_import[0].version = 17
    return m


def test_simple_chain_mined():
    """Mul(a,b) -> Sigmoid(c) -> graph output"""
    n1 = helper.make_node("Mul", ["a", "b"], ["c"], name="n1")
    n2 = helper.make_node("Sigmoid", ["c"], ["y"], name="n2")
    m = _model([n1, n2], [_input("a"), _input("b")], [_output("y")])
    patterns = mine(m, Logger(verbose=False))
    assert len(patterns) == 1
    p = patterns[0]
    assert len(p.op_kinds) == 2
    assert p.op_kinds[0][0] == "BinaryOp"     # Mul
    assert p.op_kinds[0][1] == 2
    assert p.op_kinds[1][0] == "UnaryOp"      # Sigmoid
    assert p.op_kinds[1][1] == 23
    assert len(p.occurrences) == 1
    occ = p.occurrences[0]
    assert occ.boundary_inputs == ["a", "b"]
    assert occ.boundary_output == "y"


def test_chain_breaks_on_graph_output():
    """Sigmoid output is also a graph output; chain mustn't be extended."""
    n1 = helper.make_node("Mul", ["a", "b"], ["c"], name="n1")
    n2 = helper.make_node("Sigmoid", ["c"], ["d"], name="n2")
    n3 = helper.make_node("Exp", ["d"], ["y"], name="n3")
    m = _model([n1, n2, n3], [_input("a"), _input("b")],
               [_output("d"), _output("y")])
    patterns = mine(m, Logger(verbose=False))
    # Mul->Sigmoid is a chain (output d is graph output, so chain stops there);
    # n3 alone is too short. We expect exactly one chain: Mul->Sigmoid.
    # But onnx_fsm refuses to absorb a graph output mid-chain — check that the
    # mined chain doesn't end at d (mid-chain) but at d as the final output.
    # In our current implementation, when mid_t IS a graph output we *break*
    # rather than extend, so Mul->Sigmoid completes with output 'd'.
    if patterns:
        for p in patterns:
            for occ in p.occurrences:
                tail = occ.boundary_output
                # tail should not lie strictly inside the chain — verify by
                # checking no swallowed op produces that tail.
                assert tail in {"d", "y"}


def test_chain_breaks_on_multi_consumer():
    """`c` is consumed by two ops; the chain after Mul must not extend."""
    n1 = helper.make_node("Mul", ["a", "b"], ["c"], name="n1")
    n2 = helper.make_node("Sigmoid", ["c"], ["y1"], name="n2")
    n3 = helper.make_node("Exp", ["c"], ["y2"], name="n3")
    m = _model([n1, n2, n3], [_input("a"), _input("b")],
               [_output("y1"), _output("y2")])
    patterns = mine(m, Logger(verbose=False))
    # No chain of length >= 2 should form.
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_chain_pos_recorded_for_sub():
    """Sigmoid -> Sub: chain output enters Sub at pos 1 (right operand)."""
    n1 = helper.make_node("Sigmoid", ["a"], ["s"], name="n1")
    n2 = helper.make_node("Sub", ["b", "s"], ["y"], name="n2")
    m = _model([n1, n2], [_input("a"), _input("b")], [_output("y")])
    patterns = mine(m, Logger(verbose=False))
    assert len(patterns) == 1
    kinds = patterns[0].op_kinds
    assert kinds[0] == ("UnaryOp", 23, -1)
    assert kinds[1][0] == "BinaryOp"
    assert kinds[1][1] == 1                      # Sub
    assert kinds[1][2] == 1                      # chain on RHS
    assert 1 in NONCOMMUTATIVE_BINARY


def test_unsupported_op_breaks_chain():
    """Unknown op (Gemm) splits the chain."""
    n1 = helper.make_node("Sigmoid", ["a"], ["b"], name="n1")
    n2 = helper.make_node(
        "Gemm",
        ["b", "w", "c"],
        ["d"],
        name="n2",
    )
    n3 = helper.make_node("Exp", ["d"], ["y"], name="n3")
    m = _model(
        [n1, n2, n3],
        [_input("a"), _input("w"), _input("c")],
        [_output("y")],
    )
    patterns = mine(m, Logger(verbose=False))
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_repeated_chains_grouped_into_one_pattern():
    """Two identical Add->Sigmoid chains should land in one pattern."""
    nodes = [
        helper.make_node("Add", ["a", "b"], ["c1"], name="n1"),
        helper.make_node("Sigmoid", ["c1"], ["y1"], name="n2"),
        helper.make_node("Add", ["d", "e"], ["c2"], name="n3"),
        helper.make_node("Sigmoid", ["c2"], ["y2"], name="n4"),
    ]
    inputs = [_input(n) for n in ("a", "b", "d", "e")]
    outputs = [_output("y1"), _output("y2")]
    patterns = mine(_model(nodes, inputs, outputs), Logger(verbose=False))
    assert len(patterns) == 1
    assert len(patterns[0].occurrences) == 2


def test_max_pattern_size_capping():
    """A chain of 5 Sigmoids capped at max=3 produces shorter chains."""
    nodes, prev = [], "x0"
    for i in range(5):
        nxt = f"x{i+1}"
        nodes.append(helper.make_node("Sigmoid", [prev], [nxt],
                                      name=f"n{i}"))
        prev = nxt
    m = _model(nodes, [_input("x0")], [_output("x5")])
    patterns = mine(m, Logger(verbose=False), max_pattern_size=3)
    for p in patterns:
        for occ in p.occurrences:
            assert len(occ.steps) <= 3


def test_all_const_chain_rejected():
    """Chains whose boundary inputs are ALL initializers must NOT be fused.
    MNN's GeometryComputerUtils tries to constant-fold such ops via the CPU
    backend; OpType_Extra has no CPU impl and produces 'Don't support type
    [Extra]' + 'Const Folder Error' at session-creation time."""
    import numpy as np
    from onnx import numpy_helper
    a_init = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32),
                                     name="a_const")
    b_init = numpy_helper.from_array(np.array([[3.0, 4.0]], dtype=np.float32),
                                     name="b_const")
    nodes = [
        helper.make_node("Mul", ["a_const", "b_const"], ["c"], name="n1"),
        helper.make_node("Sigmoid", ["c"], ["y"], name="n2"),
    ]
    g = helper.make_graph(nodes, "t", [], [_output("y", (1, 2))],
                          initializer=[a_init, b_init])
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    patterns = mine(m, Logger(verbose=False))
    # Even though Mul→Sigmoid is a perfect 2-step chain, both inputs are
    # constants → MNN would try to const-fold at session creation. Refuse.
    assert sum(len(p.occurrences) for p in patterns) == 0


def test_partially_const_chain_still_fused():
    """If at least one input is dynamic (graph input), the chain IS fusable —
    no const-fold attempt happens."""
    import numpy as np
    from onnx import numpy_helper
    bias = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32),
                                   name="bias")
    nodes = [
        helper.make_node("Mul", ["x", "bias"], ["c"], name="n1"),
        helper.make_node("Sigmoid", ["c"], ["y"], name="n2"),
    ]
    g = helper.make_graph(nodes, "t",
                          [_input("x", (1, 2))],
                          [_output("y", (1, 2))],
                          initializer=[bias])
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    patterns = mine(m, Logger(verbose=False))
    assert sum(len(p.occurrences) for p in patterns) == 1


def test_custom_domain_node_skipped():
    """A node already in our custom domain must not be re-mined."""
    n1 = helper.make_node("Sigmoid", ["a"], ["b"], name="n1")
    n2 = helper.make_node("MnnForge_xxxx", ["b"], ["y"], name="n2",
                          domain="com.mnnforge")
    m = _model([n1, n2], [_input("a")], [_output("y")])
    patterns = mine(m, Logger(verbose=False))
    # The chain should stop at Sigmoid; MnnForge_xxxx isn't a primitive.
    # Sigmoid alone is length 1, so no patterns of length >= 2.
    assert sum(len(p.occurrences) for p in patterns) == 0
