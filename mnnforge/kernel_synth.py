"""Phase 5 — synthesize an OpenCL .cl kernel for a single fused pattern.

Bug fixes (vs first draft) tagged BUGFIX-KS-NN:
  1: signature now (type_name, sub_kind, chain_pos) to match the new FSM.
  2: honor chain_pos for non-commutative BinaryOp (Sub, Div).
  3: bin_idx bounds check moved before use.
  4: dropped unused locals.
  5: input arity for the kernel is derived from the steps, not passed in,
     so we never have arity mismatches with FSM.
"""
from __future__ import annotations
from typing import List, Tuple

from .primitives import (
    BINARY_SNIPPETS, UNARY_SNIPPETS, NAMED_SNIPPETS, HELPERS, fmt,
)


PROLOGUE = (
    "#ifdef MNN_SUPPORT_FP16\n"
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#endif\n"
    "__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | "
    "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
)


def required_boundary_input_count(
    op_kinds: List[Tuple[str, int, int]]
) -> int:
    """Compute how many distinct boundary inputs a chain with these
    (type_name, sub_kind, chain_pos) steps needs.

    Rule:  the head op contributes len(head.inputs) — but at this layer
    we don't know that, so we charge the head as 1 (UnaryOp/Named) and
    +1 if head is BinaryOp.  Each subsequent step contributes its
    extra-input count: 0 for unary/named, 1 for BinaryOp.
    """
    if not op_kinds:
        return 0
    n = 0
    for i, (name, _sub, _pos) in enumerate(op_kinds):
        if i == 0:
            n += 2 if name == "BinaryOp" else 1
        else:
            n += 1 if name == "BinaryOp" else 0
    return n


def synthesize_kernel(
    kernel_name: str,
    op_kinds: List[Tuple[str, int, int]],
    n_boundary_inputs: int,
) -> str:
    helpers_used: set = set()
    body_lines: List[str] = []

    # Read all boundary inputs upfront.
    for k in range(n_boundary_inputs):
        body_lines.append(
            f"  float4 v{k} = convert_float4(read_imagef(in{k}, SAMPLER, "
            f"(int2)(pos, hb)));"
        )

    # Boundary-input cursor: the head op consumes the first 1 or 2;
    # each subsequent BinaryOp consumes one more.
    bin_cursor = 0
    cur_var = ""

    for i, (name, sub, pos) in enumerate(op_kinds):
        rname = f"r{i}"

        if i == 0:
            # Head op: ALL operands come from boundary inputs.
            if name == "BinaryOp":
                if bin_cursor + 2 > n_boundary_inputs:
                    raise ValueError("not enough boundary inputs for head BinaryOp")
                tmpl = BINARY_SNIPPETS.get(sub)
                if tmpl is None:
                    raise ValueError(f"unsupported BinaryOp sub_kind={sub}")
                lhs, rhs = f"v{bin_cursor}", f"v{bin_cursor+1}"
                bin_cursor += 2
                expr = fmt(tmpl, lhs, rhs)
            elif name == "UnaryOp":
                snip = UNARY_SNIPPETS.get(sub)
                if snip is None:
                    raise ValueError(f"unsupported UnaryOp sub_kind={sub}")
                if bin_cursor + 1 > n_boundary_inputs:
                    raise ValueError("not enough boundary inputs for head UnaryOp")
                expr = fmt(snip.expr, f"v{bin_cursor}")
                helpers_used.update(snip.needs)
                bin_cursor += 1
            else:
                snip = NAMED_SNIPPETS.get(name)
                if snip is None:
                    raise ValueError(f"unsupported op kind '{name}'")
                if bin_cursor + 1 > n_boundary_inputs:
                    raise ValueError(f"not enough boundary inputs for head {name}")
                expr = fmt(snip.expr, f"v{bin_cursor}")
                helpers_used.update(snip.needs)
                bin_cursor += 1
        else:
            # Subsequent step: chain output is one operand; BinaryOp pulls
            # the other from the next boundary input. UnaryOp/Named consume
            # only the chain output. BUGFIX-KS-2: pos drives operand order.
            if name == "BinaryOp":
                if bin_cursor + 1 > n_boundary_inputs:   # BUGFIX-KS-3
                    raise ValueError(
                        f"not enough boundary inputs at step {i} "
                        f"(need cursor {bin_cursor+1}, have {n_boundary_inputs})"
                    )
                tmpl = BINARY_SNIPPETS.get(sub)
                if tmpl is None:
                    raise ValueError(f"unsupported BinaryOp sub_kind={sub}")
                extra = f"v{bin_cursor}"
                bin_cursor += 1
                if pos == 0:
                    lhs, rhs = cur_var, extra
                else:
                    lhs, rhs = extra, cur_var      # BUGFIX-KS-2
                expr = fmt(tmpl, lhs, rhs)
            elif name == "UnaryOp":
                snip = UNARY_SNIPPETS.get(sub)
                if snip is None:
                    raise ValueError(f"unsupported UnaryOp sub_kind={sub}")
                expr = fmt(snip.expr, cur_var)
                helpers_used.update(snip.needs)
            else:
                snip = NAMED_SNIPPETS.get(name)
                if snip is None:
                    raise ValueError(f"unsupported op kind '{name}'")
                expr = fmt(snip.expr, cur_var)
                helpers_used.update(snip.needs)

        body_lines.append(f"  float4 {rname} = {expr};")
        cur_var = rname

    body_lines.append(
        f"  write_imagef(out0, (int2)(pos, hb), {cur_var});"
    )

    helpers = "".join(HELPERS[h] for h in sorted(helpers_used))
    in_params = ", ".join(
        f"__read_only image2d_t in{k}" for k in range(n_boundary_inputs)
    )
    sig = (
        f"__kernel void {kernel_name}(\n"
        f"    {in_params},\n"
        f"    __write_only image2d_t out0,\n"
        f"    __private const int W0,\n"
        f"    __private const int W1,\n"
        f"    __private const int W2)\n"
        f"{{\n"
        f"  int cb  = get_global_id(0);\n"
        f"  int w   = get_global_id(1);\n"
        f"  int hb  = get_global_id(2);\n"
        f"  if (cb >= W0 || w >= W1 || hb >= W2) return;\n"
        f"  int pos = mad24(cb, W1, w);\n"
    )

    return PROLOGUE + helpers + sig + "\n".join(body_lines) + "\n}\n"
