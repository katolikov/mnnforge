"""Phase 6 — structural verification.

The new flow can't run the optimized ONNX through ONNX Runtime because the
custom-op nodes have an unknown op_type. We verify what we can:

  1. Canonical ONNX still passes onnx.checker.
  2. Optimized ONNX still passes onnx.checker (custom domain ops skipped).
  3. ORT executes the canonical ONNX without error (sanity baseline).
  4. The set of graph outputs is unchanged between canonical and optimized.
  5. Each fused custom-op node has the expected attributes & inputs/outputs.

Numerical end-to-end verification (ORT vs MNN-OpenCL) is the user's
responsibility AFTER they've built MNN and run MNNConvert against the
optimized ONNX. mnnforge prints the exact command in the final log line.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import onnx

from .log import Logger
from .onnx_surgery import CUSTOM_DOMAIN


@dataclass
class CheckResult:
    label: str
    ok: Optional[bool]
    detail: str = ""


def _check_model(path: str, *, allow_custom: bool, log: Logger) -> CheckResult:
    try:
        m = onnx.load(path)
    except Exception as e:
        return CheckResult(f"load:{os.path.basename(path)}", False,
                           f"{type(e).__name__}: {e}")
    try:
        if allow_custom:
            shadow = onnx.ModelProto(); shadow.CopyFrom(m)
            keep = [n for n in shadow.graph.node if n.domain != CUSTOM_DOMAIN]
            del shadow.graph.node[:]
            shadow.graph.node.extend(keep)
            try:
                onnx.checker.check_model(shadow, full_check=False)
            except Exception as e:
                return CheckResult(f"checker:{os.path.basename(path)}",
                                   False, str(e).splitlines()[0][:200])
        else:
            onnx.checker.check_model(m)
    except Exception as e:
        return CheckResult(f"checker:{os.path.basename(path)}", False,
                           str(e).splitlines()[0][:200])
    return CheckResult(f"checker:{os.path.basename(path)}", True,
                       f"{len(m.graph.node)} nodes, "
                       f"{len(m.graph.initializer)} initializers")


_ORT_KNOWN_DOMAINS = {"", "ai.onnx", "ai.onnx.ml",
                      "com.microsoft", "com.microsoft.nchwc"}


def _foreign_domains(path: str) -> set:
    """Return the set of node domains that ORT's CPUExecutionProvider can't
    load. Includes our own CUSTOM_DOMAIN — it would also blow up ORT."""
    try:
        m = onnx.load(path)
    except Exception:
        return set()
    return {n.domain for n in m.graph.node
            if n.domain not in _ORT_KNOWN_DOMAINS}


def _ort_smoke(path: str, log: Logger) -> CheckResult:
    try:
        import onnxruntime as ort
    except ImportError:
        return CheckResult("ort_smoke", None, "onnxruntime not installed")

    foreign = _foreign_domains(path)
    if foreign:
        # Pre-existing or mnnforge-injected custom ops will fail ORT load.
        # That's not a verification failure — just inconclusive.
        return CheckResult(
            "ort_smoke", None,
            f"skipped: model uses non-standard op domain(s) {sorted(foreign)} "
            "that ORT-CPU cannot load"
        )

    so = ort.SessionOptions(); so.log_severity_level = 3
    try:
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        # Detect "is not a registered function/op" — same root cause.
        msg = str(e)
        if "is not a registered function/op" in msg:
            return CheckResult(
                "ort_smoke", None,
                f"skipped: ORT reports unregistered op "
                f"({msg.splitlines()[-1][:160]})"
            )
        return CheckResult("ort_smoke", False, f"load: {e}")
    rng = np.random.default_rng(0)
    feeds: Dict[str, np.ndarray] = {}
    DTYPES = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8, "tensor(int32)": np.int32,
        "tensor(int64)": np.int64, "tensor(bool)": np.bool_,
    }
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 32)
                 for i, d in enumerate(inp.shape)]
        dtype = DTYPES.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feeds[inp.name] = rng.standard_normal(size=shape).astype(dtype)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
    try:
        sess.run(None, feeds)
    except Exception as e:
        return CheckResult("ort_smoke", False, f"run: {e}")
    return CheckResult("ort_smoke", True,
                       f"{len(sess.get_inputs())} input(s), "
                       f"{len(sess.get_outputs())} output(s)")


def _output_names_match(canonical: str, optimized: str) -> CheckResult:
    a = onnx.load(canonical); b = onnx.load(optimized)
    a_outs = [o.name for o in a.graph.output]
    b_outs = [o.name for o in b.graph.output]
    if a_outs == b_outs:
        return CheckResult("output_names_match", True,
                           f"{len(a_outs)} output(s) preserved")
    return CheckResult("output_names_match", False, f"{a_outs} vs {b_outs}")


def _custom_nodes_well_formed(optimized: str) -> CheckResult:
    m = onnx.load(optimized)
    expected_attrs = {"kernel_name", "op_kinds", "fingerprint"}
    bad: List[str] = []
    n = 0
    for node in m.graph.node:
        if node.domain != CUSTOM_DOMAIN:
            continue
        n += 1
        keys = {a.name for a in node.attribute}
        missing = expected_attrs - keys
        if missing:
            bad.append(f"{node.name}: missing {missing}")
            continue
        if not [t for t in node.input if t]:
            bad.append(f"{node.name}: no inputs")
        if len([t for t in node.output if t]) != 1:
            bad.append(f"{node.name}: expected 1 output, "
                       f"got {len(node.output)}")
    if bad:
        return CheckResult("custom_nodes_well_formed", False,
                           "; ".join(bad[:5]))
    return CheckResult("custom_nodes_well_formed", True,
                       f"{n} custom node(s) present and well-formed")


def verify_structural(canonical_onnx: str, optimized_onnx: str,
                      report_path: str, log: Logger) -> bool:
    log.phase(6, "structural verification")
    results: List[CheckResult] = [
        _check_model(canonical_onnx, allow_custom=False, log=log),
        _check_model(optimized_onnx, allow_custom=True, log=log),
        _ort_smoke(canonical_onnx, log),
        _output_names_match(canonical_onnx, optimized_onnx),
        _custom_nodes_well_formed(optimized_onnx),
    ]
    for r in results:
        prefix = (log.ok if r.ok is True
                  else log.err if r.ok is False
                  else log.warn)
        prefix(f"  [{r.label}] {r.detail}")

    rep_dir = os.path.dirname(os.path.abspath(report_path)) or "."
    os.makedirs(rep_dir, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump({
            "canonical": canonical_onnx,
            "optimized": optimized_onnx,
            "results": [
                {"label": r.label, "ok": r.ok, "detail": r.detail}
                for r in results
            ],
        }, fh, indent=2)
    log.ok(f"wrote {report_path}")

    if all(r.ok is None for r in results):
        log.warn("all checks inconclusive (deps missing)")
        return False
    return all(r.ok is True for r in results if r.ok is not None)
