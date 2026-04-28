"""Phase 0 — preflight checks.

Validates inputs before any expensive work:
  * mnn_root looks like an MNN tree (schema + converter + opencl backend).
  * Refuses to ever touch schema/private/ or source/internal/.
  * onnx model loads and passes onnx.checker (warning-only on checker fail).
  * onnxruntime importable for ground-truth verification.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from .log import Logger


@dataclass
class Preflight:
    mnn_root: str
    onnx_path: str
    flatc_path: str
    converter_bin: str  # path to MNNConvert (may not exist yet — built later)
    schema_dir: str     # mnn_root/schema/default


REQUIRED_PATHS = [
    "schema/default/MNN.fbs",
    "tools/converter/CMakeLists.txt",
    "source/backend/opencl/execution/image/FuseExecution.cpp",  # the runtime path we rely on
]

FORBIDDEN_PATHS = [
    "schema/private",
    "source/internal",
]


def _norm(p: str) -> str:
    return os.path.realpath(os.path.expanduser(p))


def run(mnn_root: str, onnx_path: str, log: Logger) -> Preflight:
    mnn_root = _norm(mnn_root)
    onnx_path = _norm(onnx_path)

    log.phase(0, "preflight")

    if not os.path.isdir(mnn_root):
        raise SystemExit(f"mnn_root is not a directory: {mnn_root}")
    if not os.path.isfile(onnx_path):
        raise SystemExit(f"onnx model not found: {onnx_path}")

    for rel in REQUIRED_PATHS:
        full = os.path.join(mnn_root, rel)
        if not os.path.exists(full):
            raise SystemExit(f"mnn_root missing required path: {rel}")
    log.ok(f"mnn_root looks valid: {mnn_root}")

    # We never touch private code paths.
    for rel in FORBIDDEN_PATHS:
        full = os.path.join(mnn_root, rel)
        if os.path.exists(full):
            log.vinfo(f"present (will NOT be touched): {rel}")

    # ONNX load + checker (non-fatal on checker failure — many real models
    # carry warnings; we just record).
    try:
        import onnx
    except ImportError as e:
        raise SystemExit(f"onnx package required: {e}")

    try:
        m = onnx.load(onnx_path)
        log.ok(f"onnx loaded: {len(m.graph.node)} nodes, "
               f"{len(m.graph.initializer)} initializers")
    except Exception as e:
        raise SystemExit(f"failed to load onnx model: {e}")

    try:
        onnx.checker.check_model(m)
        log.ok("onnx checker passed")
    except Exception as e:
        log.warn(f"onnx checker emitted issues (continuing): "
                 f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")

    # ORT (warn-only — verify phase will hard-fail later if needed).
    try:
        import onnxruntime  # noqa: F401
        log.ok("onnxruntime available (CPU provider used for ground truth)")
    except ImportError:
        log.warn("onnxruntime not installed — verification will be skipped")

    flatc = os.path.join(mnn_root, "3rd_party", "flatbuffers", "tmp", "flatc")
    converter = os.path.join(mnn_root, "build_mnnforge", "MNNConvert")
    schema_dir = os.path.join(mnn_root, "schema", "default")

    return Preflight(
        mnn_root=mnn_root,
        onnx_path=onnx_path,
        flatc_path=flatc,
        converter_bin=converter,
        schema_dir=schema_dir,
    )
