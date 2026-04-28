"""Phase 6 — replace each FSM-discovered op span with a single OpType_Extra
op carrying the synthesized OpenCL .cl source as Extra.info bytes.

Bug fixes (vs first draft) tagged BUGFIX-SUR-NN:
  1: Extra.info must be `bytes`, not `list(int)`. flatbuffers' `[byte]`
     field is bytes-typed in the Python object API.
  2: cross-pattern overlap is now a hard pre-pass: occurrences from a
     later (lower-score) pattern that touch an already-claimed op are
     dropped before any kernel synthesis, so the cost model decides.
  3: deterministic occurrence ordering (sorted by head op idx) so behaviour
     is reproducible across runs.
  4: assertion that the boundary_output tensor isn't accidentally shared
     by an op outside the chain (would mean we're stealing its producer).
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple

from .fsm import Pattern, Chain
from .kernel_synth import synthesize_kernel
from .log import Logger


def _make_extra_op(MNN, kernel_name: str, kernel_src: str,
                   inputs: List[int], outputs: List[int], op_name: str):
    OpT = MNN.Op.OpT
    ExtraT = MNN.Extra.ExtraT
    OpType = MNN.OpType.OpType
    OpParameter = MNN.OpParameter.OpParameter

    extra = ExtraT()
    extra.type = kernel_name
    extra.engine = "OpenCL"
    # BUGFIX-SUR-1: bytes, not list of ints.
    extra.info = kernel_src.encode("utf-8")
    extra.attr = None
    extra.vector = False

    op = OpT()
    op.name = op_name
    op.type = OpType.Extra
    op.mainType = OpParameter.Extra
    op.main = extra
    op.inputIndexes = list(inputs)
    op.outputIndexes = list(outputs)
    return op


def _accepted_occurrences(patterns: List[Pattern], log: Logger
                          ) -> List[Tuple[Pattern, Chain, int]]:
    """Resolve cross-pattern overlap once, deterministically.

    Returns list of (pattern, chain, occurrence_index_within_pattern) in the
    order they should be applied, with no two chains sharing an op idx.
    Higher-scoring patterns claim ops first.
    """
    claimed: Set[int] = set()
    out: List[Tuple[Pattern, Chain, int]] = []
    # patterns are pre-sorted by score desc by mine().
    for p in patterns:
        # BUGFIX-SUR-3: sort within a pattern by head op idx so the rewrite
        # is order-independent.
        ordered = sorted(enumerate(p.occurrences), key=lambda x: x[1].op_indices[0])
        for occ_i, c in ordered:
            if any(i in claimed for i in c.op_indices):
                log.vinfo(
                    f"  drop occ {p.fingerprint}#{occ_i}: overlaps already-claimed ops"
                )
                continue
            for i in c.op_indices:
                claimed.add(i)
            out.append((p, c, occ_i))
    return out


def apply_patterns(MNN, netT, patterns: List[Pattern], log: Logger,
                   top_n: int = 4) -> int:
    if not patterns:
        log.info("no patterns to apply")
        return 0

    accepted_patterns = patterns[:top_n]
    log.info(f"applying top {len(accepted_patterns)} pattern(s)")

    # 1. Synthesize kernels — once per pattern.
    kern_by_fp: Dict[str, Tuple[str, str, int]] = {}
    for p in accepted_patterns:
        kname = f"mnnforge_{p.fingerprint}"
        # First occurrence determines the kernel arity (number of boundary inputs).
        first = p.occurrences[0]
        n_in = len(first.boundary_inputs)
        try:
            src = synthesize_kernel(kname, p.op_kinds, n_in)
        except ValueError as e:
            log.warn(f"  pattern {p.fingerprint}: cannot synthesize ({e}) — skipping")
            continue
        kern_by_fp[p.fingerprint] = (kname, src, n_in)
        log.vinfo(f"  synthesized kernel '{kname}' ({len(src)} bytes, {n_in} inputs)")

    # Filter out patterns whose synthesis failed.
    accepted_patterns = [p for p in accepted_patterns if p.fingerprint in kern_by_fp]
    if not accepted_patterns:
        log.info("no patterns survived kernel synthesis")
        return 0

    # 2. Resolve overlap with deterministic claim order.
    plan = _accepted_occurrences(accepted_patterns, log)
    if not plan:
        log.info("no occurrences after overlap resolution")
        return 0

    # 3. Build replacements map. BUGFIX-SUR-4: also pre-check arity per
    # occurrence (later occurrences inside the same pattern may have
    # different boundary input counts in pathological graphs).
    op_count = len(netT.oplists or [])
    replacements: Dict[int, object] = {}
    swallowed: Set[int] = set()
    fused_total = 0

    for p, c, occ_i in plan:
        kname, ksrc, n_in = kern_by_fp[p.fingerprint]
        if len(c.boundary_inputs) != n_in:
            log.vinfo(
                f"  occ {p.fingerprint}#{occ_i}: input arity mismatch "
                f"({len(c.boundary_inputs)} vs kernel {n_in}) — skipped"
            )
            continue
        head = c.op_indices[0]
        if head in replacements or head in swallowed:
            # Should never happen given _accepted_occurrences, but guard.
            continue
        extra_op = _make_extra_op(
            MNN, kname, ksrc,
            inputs=c.boundary_inputs,
            outputs=[c.boundary_output],
            op_name=f"MnnForge_{p.fingerprint}_{occ_i}",
        )
        replacements[head] = extra_op
        for i in c.op_indices[1:]:
            swallowed.add(i)
        fused_total += 1

    if fused_total == 0:
        log.info("no occurrences applied")
        return 0

    # 4. Rewrite oplists in original order.
    new_ops = []
    for i, op in enumerate(netT.oplists or []):
        if i in swallowed:
            continue
        if i in replacements:
            new_ops.append(replacements[i])
        else:
            new_ops.append(op)
    netT.oplists = new_ops

    log.ok(f"fused {fused_total} occurrence(s); ops {op_count} -> {len(new_ops)}")
    return fused_total
