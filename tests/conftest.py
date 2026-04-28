"""Shared fixtures: make sure the package is importable when tests are run
directly from the tools/mnnforge directory (`python -m pytest tests/`)."""
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
