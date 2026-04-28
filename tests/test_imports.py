"""Smoke: every module must be importable in isolation.

Catches accidental top-level imports of optional deps and circular imports.
"""
import importlib

MODULES = [
    "mnnforge",
    "mnnforge.cli",
    "mnnforge.log",
    "mnnforge.preflight",
    "mnnforge.canonicalize",
    "mnnforge.convert",
    "mnnforge.mnn_fbs",
    "mnnforge.fsm",
    "mnnforge.kernel_synth",
    "mnnforge.surgery",
    "mnnforge.verify",
    "mnnforge.primitives",
]


def test_imports():
    for m in MODULES:
        importlib.import_module(m)


def test_version_present():
    import mnnforge
    assert isinstance(mnnforge.__version__, str)
    # Loosely validate semver-ish.
    assert mnnforge.__version__.count(".") >= 1


def test_argparser_help_does_not_crash():
    from mnnforge.cli import _build_argparser
    p = _build_argparser()
    # If argparse internals choke, this raises. argparse hard-exits on
    # --help, so we only assert that construction succeeds.
    assert p.prog == "mnnforge"
