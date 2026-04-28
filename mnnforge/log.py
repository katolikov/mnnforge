"""Uniform stderr logger for the pipeline."""
from __future__ import annotations
import sys
import time
from typing import Optional


class Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._t0 = time.monotonic()

    def _emit(self, prefix: str, msg: str) -> None:
        elapsed = time.monotonic() - self._t0
        sys.stderr.write(f"[{elapsed:7.2f}s] {prefix} {msg}\n")
        sys.stderr.flush()

    def phase(self, n: int, name: str) -> None:
        self._emit("===", f"Phase {n}: {name}")

    def info(self, msg: str) -> None:
        self._emit("   ", msg)

    def vinfo(self, msg: str) -> None:
        if self.verbose:
            self._emit("   ", msg)

    def warn(self, msg: str) -> None:
        self._emit("!! ", msg)

    def err(self, msg: str) -> None:
        self._emit("XX ", msg)

    def ok(self, msg: str) -> None:
        self._emit(" ✓ ", msg)
