"""Phase 5 — kernel synthesizer."""
import re
import pytest
from mnnforge.kernel_synth import synthesize_kernel, required_boundary_input_count


def _has(pattern, text, flags=0):
    return re.search(pattern, text, flags) is not None


def test_unary_chain_sigmoid_then_relu():
    src = synthesize_kernel(
        "k_demo",
        op_kinds=[("UnaryOp", 23, -1), ("ReLU", 0, 0)],   # Sigmoid -> ReLU
        n_boundary_inputs=1,
    )
    assert "__kernel void k_demo" in src
    assert "in0" in src and "in1" not in src
    # Sigmoid helper should be present.
    assert "_mnn_sigmoid" in src
    # Final write goes to out0.
    assert _has(r"write_imagef\(out0", src)
    # No FP16 helper needed beyond pragma.
    assert "MNN_SUPPORT_FP16" in src


def test_binary_head_mul_then_unary_sigmoid():
    src = synthesize_kernel(
        "k_mul_sig",
        op_kinds=[("BinaryOp", 2, -1), ("UnaryOp", 23, 0)],  # Mul, Sigmoid
        n_boundary_inputs=2,
    )
    # Two image inputs.
    assert "in0" in src and "in1" in src and "in2" not in src
    # Mul expression on the head: v0 * v1
    assert _has(r"r0\s*=\s*\(v0\s*\*\s*v1\)", src)
    assert "_mnn_sigmoid" in src


def test_non_commutative_chain_pos_1():
    """Sub with chain output as second operand: must produce (extra - chain)."""
    # Step 0: BinaryOp Add (head, 2 boundary inputs v0 + v1) -> r0
    # Step 1: BinaryOp Sub (sub_kind=1) with chain at pos 1 means:
    #         result = v_extra - r0  (NOT r0 - v_extra)
    src = synthesize_kernel(
        "k_sub_pos1",
        op_kinds=[("BinaryOp", 0, -1), ("BinaryOp", 1, 1)],
        n_boundary_inputs=3,
    )
    # Expect:  r1 = (v2 - r0)
    assert _has(r"r1\s*=\s*\(v2\s*-\s*r0\)", src), src


def test_non_commutative_chain_pos_0():
    """Same Sub but chain at pos 0 → (chain - extra)."""
    src = synthesize_kernel(
        "k_sub_pos0",
        op_kinds=[("BinaryOp", 0, -1), ("BinaryOp", 1, 0)],
        n_boundary_inputs=3,
    )
    assert _has(r"r1\s*=\s*\(r0\s*-\s*v2\)", src), src


def test_arity_helper():
    # 2 BinaryOps + 1 UnaryOp head: head=1 input + 2× +1 = 3 inputs total.
    assert required_boundary_input_count([
        ("UnaryOp", 7, -1),
        ("BinaryOp", 0, 0),
        ("BinaryOp", 2, 0),
    ]) == 3
    # 3-step BinaryOp chain head: head=2 inputs + 2× +1 = 4 inputs total.
    assert required_boundary_input_count([
        ("BinaryOp", 0, -1),
        ("BinaryOp", 0, 0),
        ("BinaryOp", 0, 0),
    ]) == 4


def test_insufficient_inputs_raises():
    with pytest.raises(ValueError):
        synthesize_kernel("k", [("BinaryOp", 0, -1)], n_boundary_inputs=1)


def test_kernel_signature_matches_fuseexecution_layout():
    """FuseExecution sets args in this order:
         all inputs, then all outputs, then int W0, W1, W2.
       Our kernel's parameter list MUST mirror that.
    """
    src = synthesize_kernel(
        "k_layout",
        op_kinds=[("UnaryOp", 7, -1)],   # exp
        n_boundary_inputs=1,
    )
    # Find the kernel signature line.
    m = re.search(r"__kernel void k_layout\(([^)]+)\)", src, re.DOTALL)
    assert m, "kernel signature missing"
    params = [p.strip() for p in m.group(1).split(",")]
    # Expected order: in0, out0, W0, W1, W2
    assert params[0].startswith("__read_only image2d_t in0")
    assert params[1].startswith("__write_only image2d_t out0")
    assert "int W0" in params[2]
    assert "int W1" in params[3]
    assert "int W2" in params[4]


def test_global_id_bounds_check():
    src = synthesize_kernel("k_bounds", [("UnaryOp", 7, -1)], 1)
    # Must early-return on out-of-range global ids.
    assert _has(r"if\s*\(cb\s*>=\s*W0", src)


def test_unsupported_unary_kind():
    with pytest.raises(ValueError):
        synthesize_kernel("k_bad", [("UnaryOp", 9999, -1)], 1)


def test_unsupported_op_name():
    with pytest.raises(ValueError):
        synthesize_kernel("k_bad", [("Convolution", 0, -1)], 1)
