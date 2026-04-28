"""Logger and preflight basic behavior."""
import os
import tempfile
import pytest

from mnnforge.log import Logger
from mnnforge import preflight


def test_logger_methods_dont_raise(capsys):
    log = Logger(verbose=True)
    log.phase(0, "x"); log.info("hi"); log.vinfo("v")
    log.warn("w"); log.err("e"); log.ok("k")
    out = capsys.readouterr()
    # Logger writes to stderr by default — capsys captures both streams.
    assert "Phase 0" in out.err
    assert "hi" in out.err


def test_preflight_rejects_missing_mnn_root(tmp_path):
    onnx = tmp_path / "x.onnx"
    onnx.write_bytes(b"not really onnx")
    with pytest.raises(SystemExit):
        preflight.run(str(tmp_path / "doesnotexist"), str(onnx),
                      Logger(verbose=False))


def test_preflight_rejects_missing_onnx(tmp_path):
    fake_root = tmp_path / "mnn"
    fake_root.mkdir()
    with pytest.raises(SystemExit):
        preflight.run(str(fake_root), str(tmp_path / "no.onnx"),
                      Logger(verbose=False))


def test_preflight_rejects_invalid_mnn_root(tmp_path):
    """A directory that's not an MNN tree must be rejected."""
    onnx = tmp_path / "x.onnx"
    onnx.write_bytes(b"\x08\x01\x12\x00")  # tiny non-onnx bytes -> fails to load
    fake_root = tmp_path / "fake"
    fake_root.mkdir()
    with pytest.raises(SystemExit):
        preflight.run(str(fake_root), str(onnx), Logger(verbose=False))
