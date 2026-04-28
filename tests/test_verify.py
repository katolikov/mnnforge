"""Phase 7 — verify module unit tests (no real ORT/MNN required for these).

We test the comparison helper directly with synthetic arrays and confirm
the JSON report machinery works.
"""
import json
import os
import tempfile
import numpy as np
import pytest

from mnnforge.log import Logger
from mnnforge.verify import _compare, verify, _make_random_inputs


def test_compare_passes_on_identical():
    a = [np.array([1.0, 2.0, 3.0])]
    ok, rows = _compare(a, a, ["out0"], atol=1e-6, rtol=1e-6)
    assert ok is True
    assert rows[0]["max_abs_err"] == 0.0


def test_compare_fails_on_different_values():
    a = [np.array([1.0, 2.0])]
    b = [np.array([1.0, 9.0])]
    ok, rows = _compare(a, b, ["out0"], atol=1e-6, rtol=1e-6)
    assert ok is False
    assert rows[0]["max_abs_err"] >= 6.0


def test_compare_reshape_recovers_compatible_shapes():
    """Trailing 1-dims sometimes differ between ORT and MNN: shape (1,3) vs
    (1,3,1,1). _compare should reshape and pass when total size matches."""
    a = [np.array([[1.0, 2.0, 3.0]])]                 # shape (1, 3)
    b = [np.array([[[[1.0]], [[2.0]], [[3.0]]]])]     # shape (1, 3, 1, 1)
    ok, rows = _compare(a, b, ["out0"], atol=1e-6, rtol=1e-6)
    assert ok is True


def test_compare_count_mismatch():
    ok, rows = _compare([np.array([1.0])], [np.array([1.0]), np.array([2.0])],
                        ["a"], atol=1e-6, rtol=1e-6)
    assert ok is False
    assert "output count" in rows[0].get("error", "")


def test_compare_shape_mismatch_unrecoverable():
    a = [np.array([1.0, 2.0])]            # size 2
    b = [np.array([1.0, 2.0, 3.0])]       # size 3 — cannot reshape
    ok, rows = _compare(a, b, ["x"], atol=1e-6, rtol=1e-6)
    assert ok is False
    assert "shape mismatch" in rows[0].get("error", "")


def test_make_random_inputs_handles_dynamic_dims():
    spec = [("x", (None, 3, "?", "?"), np.float32)]
    out = _make_random_inputs(spec, seed=42)
    assert out["x"].shape == (1, 3, 32, 32)
    assert out["x"].dtype == np.float32
    assert out["x"].flags["C_CONTIGUOUS"]


def test_make_random_inputs_int_dtype_zeroed():
    spec = [("idx", (4,), np.int32)]
    out = _make_random_inputs(spec)
    assert out["idx"].dtype == np.int32
    assert (out["idx"] == 0).all()


def test_verify_inconclusive_when_ort_missing(monkeypatch, tmp_path):
    """If ORT can't load (we simulate by giving an absent file), verify
    reports inconclusive without raising."""
    canon = str(tmp_path / "no.onnx")
    orig = str(tmp_path / "a.mnn")
    fused = str(tmp_path / "b.mnn")
    rep = str(tmp_path / "report.json")
    # Force ORT to raise.
    import mnnforge.verify as V

    def fail(*a, **k): return None
    monkeypatch.setattr(V, "_run_ort", fail)
    ok = verify(canon, orig, fused, rep, Logger(verbose=False))
    assert ok is False


def test_verify_writes_report_when_only_one_path(tmp_path, monkeypatch):
    """ORT runs OK, but MNN unavailable: the report file is still written
    and the result is inconclusive (False)."""
    import mnnforge.verify as V

    fake_outs = [np.array([1.0, 2.0, 3.0])]
    fake_inputs = {"in": np.array([[1.0]])}
    fake_names = ["out0"]
    monkeypatch.setattr(V, "_run_ort",
                        lambda p, log: (fake_outs, fake_inputs, fake_names))
    monkeypatch.setattr(V, "_run_mnn", lambda *a, **k: None)

    rep = str(tmp_path / "rep.json")
    ok = V.verify("canon.onnx", "a.mnn", "b.mnn", rep, Logger(verbose=False))
    assert ok is False
    assert os.path.exists(rep)
    payload = json.loads(open(rep).read())
    assert "results" in payload
    assert all(r["ok"] is None for r in payload["results"])
