"""Phase 7 — verify ORT(canonical onnx) vs MNN(.mnn) on OpenCL backend.

Bug fixes (vs first draft) tagged BUGFIX-V-NN:
   1: pymnn placeholder takes (shape, format[, dtype]); F.float was wrong.
   2: var.write(numpy_array) — NOT .tobytes().
   3: var.read() returns ndarray; call once.
   4: OpenCL output may be in NC4HW4 layout; F.convert(out, F.NCHW) before read.
   5: net.forward() may return a single var or list — normalize.
  15: keep dynamic-dim defaults reasonable AND pass the same array to ORT
      and MNN (dict shared, not regenerated).
  16: ascontiguousarray to avoid implicit copies losing dtype/strides.
  17: graceful CPU fallback when OpenCL unavailable (e.g. macOS arm).
  18: shape-mismatch resilience: try a reshape before declaring fail.
  19: skip 2nd MNN run if fused == original.
  20: never silently report success when nothing actually compared.
  22: dirname-or-cwd guard for the report path.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .log import Logger


FORWARD_CPU = 0
FORWARD_OPENCL = 3


@dataclass
class CompareResult:
    label: str
    ok: Optional[bool]
    per_output: List[Dict] = field(default_factory=list)
    note: str = ""


def _make_random_inputs(spec_pairs: List[Tuple[str, Tuple, np.dtype]],
                        seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for name, shape, dtype in spec_pairs:
        concrete = []
        for i, d in enumerate(shape):
            if isinstance(d, int) and d > 0:
                concrete.append(d)
            else:
                concrete.append(1 if i == 0 else 32)   # BUGFIX-V-15
        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(size=concrete).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = np.zeros(concrete, dtype=dtype)
        else:
            arr = np.zeros(concrete, dtype=dtype)
        out[name] = np.ascontiguousarray(arr)            # BUGFIX-V-16
    return out


def _run_ort(onnx_path: str, log: Logger
             ) -> Optional[Tuple[List[np.ndarray], Dict[str, np.ndarray], List[str]]]:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warn("onnxruntime missing — cannot establish ORT ground truth")
        return None
    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        log.err(f"ORT load failed for {onnx_path}: {e}")
        return None
    DTYPES = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8, "tensor(int32)": np.int32,
        "tensor(int64)": np.int64, "tensor(bool)": np.bool_,
    }
    spec = [(i.name, tuple(i.shape), DTYPES.get(i.type, np.float32))
            for i in sess.get_inputs()]
    inputs = _make_random_inputs(spec)
    out_names = [o.name for o in sess.get_outputs()]
    try:
        outs = sess.run(out_names, inputs)
    except Exception as e:
        log.err(f"ORT run failed: {e}")
        return None
    return outs, inputs, out_names


def _run_mnn(mnn_path: str, inputs: Dict[str, np.ndarray],
             out_names: List[str], log: Logger,
             forward: int = FORWARD_OPENCL) -> Optional[List[np.ndarray]]:
    try:
        import MNN
        import MNN.expr as F
        from MNN import nn
    except ImportError:
        log.warn("pymnn not installed (`pip install MNN`) — skipping MNN run")
        return None

    config = {"backend": forward, "precision": "high"}
    try:
        rt = nn.create_runtime_manager((config,))
        net = nn.load_module_from_file(
            mnn_path,
            list(inputs.keys()),
            out_names,
            runtime_manager=rt,
        )
    except Exception as e:
        log.err(f"MNN load failed for {mnn_path} on backend={forward}: {e}")
        if forward != FORWARD_CPU:                       # BUGFIX-V-17
            log.warn("retrying on CPU backend for diagnostic comparison")
            return _run_mnn(mnn_path, inputs, out_names, log, FORWARD_CPU)
        return None

    var_inputs = []
    for name, arr in inputs.items():
        v = F.placeholder(list(arr.shape), F.NCHW)       # BUGFIX-V-1
        v.name = name
        v.write(arr)                                     # BUGFIX-V-2
        var_inputs.append(v)
    try:
        var_outputs = net.forward(var_inputs)
    except Exception as e:
        log.err(f"MNN forward failed: {e}")
        return None
    if not isinstance(var_outputs, (list, tuple)):       # BUGFIX-V-5
        var_outputs = [var_outputs]

    outs: List[np.ndarray] = []
    for v in var_outputs:
        v = F.convert(v, F.NCHW)                         # BUGFIX-V-4
        arr = np.array(v.read(), copy=True)              # BUGFIX-V-3
        outs.append(arr)
    return outs


def _compare(a_outs: List[np.ndarray], b_outs: List[np.ndarray],
             names: List[str], atol: float, rtol: float
             ) -> Tuple[bool, List[Dict]]:
    if len(a_outs) != len(b_outs):
        return False, [{"error": f"output count {len(a_outs)} vs {len(b_outs)}"}]
    ok = True
    rows: List[Dict] = []
    for a, b, n in zip(a_outs, b_outs, names):
        if a.shape != b.shape and a.size == b.size:      # BUGFIX-V-18
            try:
                b = b.reshape(a.shape)
            except Exception:
                pass
        if a.shape != b.shape:
            ok = False
            rows.append({"output": n, "shape_a": list(a.shape),
                         "shape_b": list(b.shape), "error": "shape mismatch"})
            continue
        a64 = a.astype(np.float64); b64 = b.astype(np.float64)
        diff = np.abs(a64 - b64)
        max_abs = float(diff.max() if diff.size else 0.0)
        denom = np.maximum(np.abs(a64), np.abs(b64))
        denom[denom == 0.0] = 1.0
        rel = float((diff / denom).max() if diff.size else 0.0)
        passed = bool(np.allclose(a64, b64, atol=atol, rtol=rtol))
        ok = ok and passed
        rows.append({"output": n, "shape": list(a.shape),
                     "max_abs_err": max_abs, "max_rel_err": rel,
                     "pass": passed})
    return ok, rows


def verify(canonical_onnx: str, original_mnn: str, fused_mnn: str,
           report_path: str, log: Logger,
           atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    log.phase(7, "verify ORT vs MNN-OpenCL (baseline + fused)")
    ort_result = _run_ort(canonical_onnx, log)
    if ort_result is None:
        log.warn("no ORT ground truth — verification inconclusive")
        return False
    ort_outs, inputs, out_names = ort_result
    log.ok(f"ORT produced {len(ort_outs)} output(s)")
    results: List[CompareResult] = []

    # Baseline.
    mnn_outs = _run_mnn(original_mnn, inputs, out_names, log, FORWARD_OPENCL)
    if mnn_outs is None:
        results.append(CompareResult("baseline_ort_vs_mnn_original",
                                     None, note="MNN run unavailable"))
    else:
        ok, rows = _compare(ort_outs, mnn_outs, out_names, atol, rtol)
        results.append(CompareResult("baseline_ort_vs_mnn_original", ok, rows))
        (log.ok if ok else log.warn)(
            f"baseline ORT vs MNN(original): {'PASS' if ok else 'FAIL'}"
        )

    # Fused.
    if os.path.realpath(fused_mnn) == os.path.realpath(original_mnn):  # BUGFIX-V-19
        log.info("fused == original (no patterns applied); skipping 2nd run")
    else:
        fused_outs = _run_mnn(fused_mnn, inputs, out_names, log, FORWARD_OPENCL)
        if fused_outs is None:
            results.append(CompareResult("ort_vs_mnn_fused",
                                         None, note="MNN run unavailable"))
        else:
            ok, rows = _compare(ort_outs, fused_outs, out_names, atol, rtol)
            results.append(CompareResult("ort_vs_mnn_fused", ok, rows))
            (log.ok if ok else log.err)(
                f"fused ORT vs MNN(fused): {'PASS' if ok else 'FAIL'}"
            )

    payload = {
        "canonical_onnx": canonical_onnx,
        "original_mnn": original_mnn, "fused_mnn": fused_mnn,
        "atol": atol, "rtol": rtol,
        "results": [
            {"label": r.label, "ok": r.ok, "note": r.note,
             "per_output": r.per_output} for r in results
        ],
    }
    rep_dir = os.path.dirname(os.path.abspath(report_path)) or "."   # BUGFIX-V-22
    os.makedirs(rep_dir, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.ok(f"wrote {report_path}")

    if all(r.ok is None for r in results):                # BUGFIX-V-20
        log.warn("verification inconclusive (no comparisons could run)")
        return False
    return all(r.ok is True for r in results)
