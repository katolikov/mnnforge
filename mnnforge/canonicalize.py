"""Phase 1 — ONNX canonicalization.

Delegates to the project's existing untracked optimize_onnx_for_mnn.py at
the repo root if present (preferred — it carries bit-exact ORT verification
of every pass and is hand-tuned for MNN-converter compatibility). Falls
back to a minimal in-package canonicalizer otherwise so the CLI still works
on a clean tree.

We never modify or augment optimize_onnx_for_mnn.py; we import it.
"""
from __future__ import annotations
import os
import sys
from typing import List

from .log import Logger


def _try_import_root_optimizer(mnn_root: str):
    """Try to import the user's optimize_onnx_for_mnn.py from the repo root."""
    candidate = os.path.join(mnn_root, "optimize_onnx_for_mnn.py")
    if not os.path.isfile(candidate):
        return None
    if mnn_root not in sys.path:
        sys.path.insert(0, mnn_root)
    try:
        import optimize_onnx_for_mnn  # type: ignore
        return optimize_onnx_for_mnn
    except Exception:
        return None


def _fallback_canonicalize(in_path: str, out_path: str, log: Logger) -> None:
    """Minimal canonicalizer used when the repo-root script is unavailable."""
    import onnx
    log.info("using built-in fallback canonicalizer (no PReLU/fold passes)")
    m = onnx.load(in_path)

    # Drop unreferenced initializers — always safe.
    referenced = set()
    for n in m.graph.node:
        referenced.update(n.input)
    referenced.update(o.name for o in m.graph.input)
    keep = [i for i in m.graph.initializer if i.name in referenced]
    removed = len(m.graph.initializer) - len(keep)
    if removed > 0:
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)
        log.info(f"  removed {removed} dead initializer(s)")

    # Best-effort shape inference (helps later FSM with rank-aware fingerprints).
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False, check_type=False)
    except Exception as e:
        log.warn(f"  shape inference failed: {type(e).__name__}: {e}")

    onnx.save(m, out_path)


def canonicalize(mnn_root: str, in_path: str, out_path: str,
                 log: Logger, verify: bool = True) -> None:
    """Run Phase-1 canonicalization. Writes out_path."""
    log.phase(1, "canonicalize ONNX")

    mod = _try_import_root_optimizer(mnn_root)
    if mod is None:
        _fallback_canonicalize(in_path, out_path, log)
        log.ok(f"wrote {out_path}")
        return

    log.info(f"using {mnn_root}/optimize_onnx_for_mnn.py")
    res = mod.optimize(
        in_path, out_path,
        passes=None,            # all passes
        verify=verify,
        verbose=log.verbose,
    )
    total = sum(res.pass_results.values()) if hasattr(res, "pass_results") else 0
    log.ok(f"canonicalized: {total} total transformations applied")
