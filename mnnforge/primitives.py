"""OpenCL snippet library for primitive MNN ops.

Each entry tells the kernel synthesizer how to translate a single MNN op
into one (or a few) OpenCL float4 statements operating on `image2d_t` reads
already loaded into local variables.

Naming convention:
  * Inputs to the snippet are referred to by `${in0}`, `${in1}`, …
  * The snippet must produce its result as the expression body, not a
    statement — the synthesizer wraps it in an assignment.
  * Each snippet operates per-pixel in NC4HW4 layout (float4 lanes).
  * Snippets must NOT do reductions across channel/spatial axes (those
    require a different kernel template; we exclude them in FSM).

We deliberately keep this list small and elementwise-only for v1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class Snippet:
    """An elementwise OpenCL float4 snippet."""
    arity: int                      # number of float4 inputs
    expr: str                       # OpenCL expression with ${in0}/${in1} placeholders
    needs: tuple = ()               # extra macros / helper functions to prepend


# Small helper functions — emitted once into the kernel preamble if any
# snippet that references them is used.
HELPERS = {
    "gelu": (
        "inline float4 _mnn_gelu(float4 x){\n"
        "  float4 v = 0.79788458f * (0.044715f * x * x * x + x);\n"
        "  float4 x2 = v * v;\n"
        "  float4 d = (v * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)))) /\n"
        "             (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));\n"
        "  return (1.0f + d) * x * 0.5f;\n"
        "}\n"
    ),
    "sigmoid": (
        "inline float4 _mnn_sigmoid(float4 x){\n"
        "  return native_recip((float4)1.0f + native_exp(-x));\n"
        "}\n"
    ),
}


# BinaryOp opType enum values defined in MNN.fbs (BinaryOpOperation):
# 0:ADD 1:SUB 2:MUL 3:DIV 4:MAX_TEMP 5:MIN_TEMP 6:POW 7:REALDIV ... — we
# only care about the most common five for v1.
BINARY_SNIPPETS: Dict[int, str] = {
    0: "(${in0} + ${in1})",                  # ADD
    1: "(${in0} - ${in1})",                  # SUB
    2: "(${in0} * ${in1})",                  # MUL
    3: "(${in0} / ${in1})",                  # DIV / REALDIV
    7: "(${in0} / ${in1})",                  # REALDIV
}


# UnaryOp.opType enum values (UnaryOpOperation):
# 0:ABS 1:NEG 2:FLOOR 3:CEIL 4:SQUARE 5:SQRT 6:RSQRT 7:EXP 8:LOG 9:SIN
# 10:COS 11:TAN 12:ASIN 13:ACOS 14:ATAN 15:RECIPROCAL ...
# 17:SIGN 18:TANH 23:SIGMOID  31:GELU  (numbering is approximate; consult MNN.fbs)
UNARY_SNIPPETS: Dict[int, Snippet] = {
    0:  Snippet(1, "fabs(${in0})"),
    1:  Snippet(1, "(-${in0})"),
    4:  Snippet(1, "(${in0} * ${in0})"),
    5:  Snippet(1, "native_sqrt(${in0})"),
    6:  Snippet(1, "native_rsqrt(${in0})"),
    7:  Snippet(1, "native_exp(${in0})"),
    8:  Snippet(1, "native_log(${in0})"),
    15: Snippet(1, "native_recip(${in0})"),
    18: Snippet(1, "tanh(${in0})"),
    23: Snippet(1, "_mnn_sigmoid(${in0})", needs=("sigmoid",)),
    31: Snippet(1, "_mnn_gelu(${in0})",   needs=("gelu",)),
}


# OpType.ReLU and ReLU6 take their own slot (not via UnaryOp).
NAMED_SNIPPETS: Dict[str, Snippet] = {
    "ReLU":  Snippet(1, "fmax(${in0}, (float4)0.0f)"),
    "ReLU6": Snippet(1, "clamp(${in0}, (float4)0.0f, (float4)6.0f)"),
    "Sigmoid": Snippet(1, "_mnn_sigmoid(${in0})", needs=("sigmoid",)),
    "TanH":  Snippet(1, "tanh(${in0})"),
}


def fmt(template: str, *ins: str) -> str:
    out = template
    for i, name in enumerate(ins):
        out = out.replace(f"${{in{i}}}", name)
    return out
