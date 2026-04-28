"""mnnforge CLI driver."""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .log import Logger
from .preflight import run as preflight_run
from .canonicalize import canonicalize
from .convert import ensure_converter, convert as mnn_convert
from .mnn_fbs import load_mnn, save_mnn
from .fsm import mine
from .surgery import apply_patterns
from .verify import verify


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mnnforge",
        description=(
            "ONNX→MNN custom-op fusion for the OpenCL backend. "
            "Mines repeated elementwise op chains and replaces them with "
            "single OpType_Extra ops carrying runtime-compiled OpenCL kernels."
        ),
    )
    p.add_argument("mnn_root", help="path to MNN source tree")
    p.add_argument("onnx", help="path to input .onnx model")
    p.add_argument("--workdir", default=None,
                   help="working directory for intermediate files (default: alongside model)")
    p.add_argument("--top-n", type=int, default=4,
                   help="max number of fused patterns (default 4)")
    p.add_argument("--max-pattern-size", type=int, default=6,
                   help="max chain length for FSM (default 6)")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--skip-canonicalize", action="store_true")
    p.add_argument("--skip-fuse", action="store_true",
                   help="convert + verify only, no fusion")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--no-ort-verify-canon", action="store_true",
                   help="skip ORT verification inside Phase 1 canonicalize "
                        "(faster; relies on Phase 7 instead)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--version", action="version", version=f"mnnforge {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    log = Logger(verbose=args.verbose)
    log.info(f"mnnforge {__version__}")

    pre = preflight_run(args.mnn_root, args.onnx, log)

    base = os.path.splitext(os.path.basename(pre.onnx_path))[0]
    workdir = (os.path.realpath(args.workdir) if args.workdir
               else os.path.dirname(pre.onnx_path) or ".")
    os.makedirs(workdir, exist_ok=True)
    log.info(f"workdir: {workdir}")

    canon_onnx = os.path.join(workdir, f"{base}.canon.onnx")
    original_mnn = os.path.join(workdir, f"{base}.original.mnn")
    fused_mnn = os.path.join(workdir, f"{base}.fused.mnn")
    report = os.path.join(workdir, f"{base}.mnnforge.report.json")

    # ---- Phase 1
    if args.skip_canonicalize:
        log.info("phase 1 skipped (--skip-canonicalize); using onnx as-is")
        canon_onnx = pre.onnx_path
    else:
        canonicalize(
            pre.mnn_root, pre.onnx_path, canon_onnx, log,
            verify=not args.no_ort_verify_canon,
        )

    # ---- Phase 2
    log.phase(2, "convert ONNX -> MNN (stock MNNConvert)")
    converter = ensure_converter(pre.mnn_root, log)
    mnn_convert(converter, canon_onnx, original_mnn, log)

    # ---- Phases 3..6
    if args.skip_fuse:
        log.info("phases 3-6 skipped (--skip-fuse); fused = original")
        fused_mnn = original_mnn
    else:
        log.phase(3, "parse .mnn flatbuffer")
        netT, _raw, MNN = load_mnn(pre.mnn_root, original_mnn, log)

        log.phase(4, "frequent subgraph mining")
        patterns = mine(MNN, netT, log, max_pattern_size=args.max_pattern_size)

        if not patterns:
            log.info("no fusable patterns discovered — fused = original")
            fused_mnn = original_mnn
        else:
            log.phase(5, "synthesize OpenCL kernels")
            log.phase(6, "rewrite .mnn op-spans -> OpType_Extra")
            n = apply_patterns(MNN, netT, patterns, log, top_n=args.top_n)
            if n == 0:
                log.info("no occurrences fused — fused = original")
                fused_mnn = original_mnn
            else:
                save_mnn(MNN, netT, fused_mnn, log)

    # ---- Phase 7
    if args.skip_verify:
        log.info("phase 7 skipped (--skip-verify)")
        log.ok("done (no verification performed)")
        return 0

    ok = verify(canon_onnx, original_mnn, fused_mnn, report, log,
                atol=args.atol, rtol=args.rtol)
    if ok:
        log.ok("verification PASSED — fused model is numerically equivalent")
        return 0
    log.err("verification FAILED — see report for details: " + report)
    return 2
