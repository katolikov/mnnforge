"""verify.py tests — focus on the inconclusive-when-foreign-domain path."""
import json
import os

import onnx
from onnx import helper, TensorProto

from mnnforge.log import Logger
from mnnforge.verify import (
    _ort_smoke, _foreign_domains, verify_structural,
)


def _input(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _output(name, shape=(1, 4, 8, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _save(model: onnx.ModelProto, tmp_path, name: str) -> str:
    p = str(tmp_path / name)
    onnx.save(model, p)
    return p


def test_foreign_domains_detects_user_custom_ops(tmp_path):
    """Models with op_types from a custom domain (e.g. 'src.customops') must
    be flagged. This was the root cause of the 'YasdofInputDownscale' ORT
    error reported by users."""
    n = helper.make_node(
        "YasdofInputDownscale", ["a"], ["y"],
        domain="src.customops", name="custom",
    )
    g = helper.make_graph([n], "t", [_input("a")], [_output("y")])
    m = helper.make_model(g, producer_name="t")
    m.opset_import.append(helper.make_opsetid("src.customops", 1))
    m.opset_import[0].version = 17
    p = _save(m, tmp_path, "with_custom.onnx")
    assert _foreign_domains(p) == {"src.customops"}


def test_ort_smoke_returns_none_on_foreign_domain(tmp_path):
    """ORT smoke must report ok=None (inconclusive), NOT False, when the
    model carries domains ORT can't load."""
    n = helper.make_node(
        "YasdofInputDownscale", ["a"], ["y"],
        domain="src.customops", name="custom",
    )
    g = helper.make_graph([n], "t", [_input("a")], [_output("y")])
    m = helper.make_model(g, producer_name="t")
    m.opset_import.append(helper.make_opsetid("src.customops", 1))
    m.opset_import[0].version = 17
    p = _save(m, tmp_path, "with_custom.onnx")
    result = _ort_smoke(p, Logger(verbose=False))
    assert result.ok is None
    assert "src.customops" in result.detail


def test_ort_smoke_runs_ok_on_clean_model(tmp_path):
    n = helper.make_node("Sigmoid", ["a"], ["y"], name="sig")
    g = helper.make_graph([n], "t", [_input("a")], [_output("y")])
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    p = _save(m, tmp_path, "clean.onnx")
    result = _ort_smoke(p, Logger(verbose=False))
    assert result.ok is True


def test_verify_structural_passes_when_canonical_has_foreign_ops(tmp_path):
    """End-to-end: even when the canonical model has user custom ops, as long
    as the structural checks pass, verification must not fail."""
    # Canonical: just a Sigmoid + a custom node consumed downstream.
    n_sig = helper.make_node("Sigmoid", ["a"], ["s"], name="sig")
    n_cust = helper.make_node("YasdofInputDownscale", ["s"], ["y"],
                              domain="src.customops", name="custom")
    g = helper.make_graph([n_sig, n_cust], "t",
                          [_input("a")], [_output("y")])
    m = helper.make_model(g, producer_name="t")
    m.opset_import.append(helper.make_opsetid("src.customops", 1))
    m.opset_import[0].version = 17
    canon = _save(m, tmp_path, "canon.onnx")
    # Optimized: same model — pretend mnnforge found nothing to fuse.
    opt = _save(m, tmp_path, "opt.onnx")

    rep = str(tmp_path / "report.json")
    ok = verify_structural(canon, opt, rep, Logger(verbose=False))
    # Result should be True: every conclusive check passes; the ORT smoke
    # is inconclusive (None) and is excluded from the verdict.
    assert ok is True
    payload = json.loads(open(rep).read())
    smoke = next(r for r in payload["results"] if r["label"] == "ort_smoke")
    assert smoke["ok"] is None
    assert "skipped" in smoke["detail"]
