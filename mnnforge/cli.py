"""mnnforge CLI driver — ONNX-side flow.

Usage:
    python -m mnnforge <mnn_root> <model.onnx> [options]

Phases:
  0 preflight            (validate inputs)
  1 canonicalize         (model.canon.onnx)
  2 FSM on ONNX graph
  3 kernel synthesis
  4 emit into MNN tree   (.cl + Execution.{hpp,cpp} + FuseExecution.cpp dispatch)
  5 rewrite ONNX         (model.optimized.onnx with MnnForge_<fp> custom nodes)
  6 structural verify    (ORT round-trip on canonical, schema check on optimized)
"""
from __future__ import annotations
import argparse
import os
from typing import List, Optional

import onnx

from . import __version__
from .log import Logger
from .preflight import run as preflight_run
from .canonicalize import canonicalize
from .onnx_fsm import mine
from .onnx_surgery import rewrite_onnx
from .mnn_emit import emit_all, rollback as mnn_rollback
from .verify import verify_structural


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mnnforge",
        description=(
            "ONNX→MNN custom-op fusion. Mines repeated elementwise op "
            "chains in your ONNX, emits OpenCL kernels + Execution classes "
            "into the MNN source tree, and rewrites the ONNX so MNNConvert "
            "produces a .mnn that uses them."
        ),
    )
    p.add_argument("mnn_root", help="path to MNN source tree")
    p.add_argument("onnx", nargs="?", help="path to input .onnx model "
                                          "(omit when using --rollback)")
    p.add_argument("--workdir", default=None,
                   help="dir for ONNX outputs (default: alongside model)")
    p.add_argument("--top-n", type=int, default=4,
                   help="max number of fused patterns (default 4)")
    p.add_argument("--max-pattern-size", type=int, default=6,
                   help="max chain length for FSM (default 6)")
    p.add_argument("--skip-canonicalize", action="store_true")
    p.add_argument("--skip-emit", action="store_true",
                   help="don't write into MNN tree (analysis only)")
    p.add_argument("--skip-rewrite", action="store_true",
                   help="don't write the optimized ONNX")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--no-ort-verify-canon", action="store_true",
                   help="skip per-pass ORT verification in canonicalize")
    p.add_argument("--rollback", action="store_true",
                   help="restore FuseExecution.cpp from backup, "
                        "remove generated files, and exit")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--version", action="version",
                   version=f"mnnforge {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    log = Logger(verbose=args.verbose)
    log.info(f"mnnforge {__version__}")

    if args.rollback:
        if not os.path.isdir(args.mnn_root):
            log.err(f"mnn_root invalid: {args.mnn_root}")
            return 1
        n = mnn_rollback(args.mnn_root, log)
        log.ok(f"rollback complete ({n} file(s) removed)")
        return 0

    if not args.onnx:
        log.err("onnx path required (or use --rollback)")
        return 1

    pre = preflight_run(args.mnn_root, args.onnx, log)
    base = os.path.splitext(os.path.basename(pre.onnx_path))[0]
    workdir = (os.path.realpath(args.workdir) if args.workdir
               else os.path.dirname(pre.onnx_path) or ".")
    os.makedirs(workdir, exist_ok=True)
    log.info(f"workdir: {workdir}")

    canon_onnx = os.path.join(workdir, f"{base}.canon.onnx")
    optimized_onnx = os.path.join(workdir, f"{base}.optimized.onnx")
    report = os.path.join(workdir, f"{base}.mnnforge.report.json")

    # ---- Phase 1
    if args.skip_canonicalize:
        log.info("phase 1 skipped — using onnx as-is")
        canon_onnx = pre.onnx_path
    else:
        canonicalize(pre.mnn_root, pre.onnx_path, canon_onnx, log,
                     verify=not args.no_ort_verify_canon)

    log.phase(2, "load + FSM on the ONNX graph")
    model = onnx.load(canon_onnx)
    patterns = mine(model, log, max_pattern_size=args.max_pattern_size)

    if not patterns:
        log.warn("no fusable patterns discovered — nothing to emit")
        if not (args.skip_rewrite or args.skip_emit):
            onnx.save(model, optimized_onnx)
            log.ok(f"copied canonical ONNX to {optimized_onnx}")
        return 0

    # ---- Phase 4
    if args.skip_emit:
        log.info("phase 4 skipped — analysis only")
        emissions = []
    else:
        log.phase(4, "emit kernels + Execution classes into MNN tree")
        emissions = emit_all(pre.mnn_root, patterns, args.top_n, log)

    # ---- Phase 5
    if args.skip_rewrite:
        log.info("phase 5 skipped — not writing optimized ONNX")
        n_fused = 0
    else:
        log.phase(5, "rewrite ONNX with custom-op nodes")
        new_model, n_fused = rewrite_onnx(model, patterns, args.top_n, log)
        onnx.save(new_model, optimized_onnx)
        log.ok(f"wrote {optimized_onnx} ({n_fused} subgraph(s) replaced)")

    # ---- Phase 6
    if args.skip_verify:
        log.info("phase 6 skipped (--skip-verify)")
        log.ok("done")
        return 0

    if not args.skip_rewrite:
        ok = verify_structural(canon_onnx, optimized_onnx, report, log)
        if ok:
            log.ok("structural verification PASSED")
            log.info("Next: build MNN, then run "
                     f"`MNNConvert -f ONNX --modelFile {optimized_onnx} "
                     "--MNNModel out.mnn`")
            return 0
        log.err(f"structural verification FAILED — see {report}")
        return 2

    log.ok("done")
    return 0
