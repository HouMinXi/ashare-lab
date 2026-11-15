"""Skips that say why, and a way to prove nothing was skipped.

Several research tests can only run where Qlib and torch are installed. A skip
is honest there, but it is also invisible: a host without the stack reports
green while never running the assertion. Setting ASHARE_REQUIRE_STACK=1 turns
those skips into failures, so a run that claims to have checked the training
path can prove it did.
"""
from __future__ import annotations

import importlib.util

import pytest

def _available(name: str) -> bool:
    # find_spec imports the parent of a dotted name, so a missing parent
    # raises instead of returning None; a stubbed module with no spec raises
    # ValueError. Both mean the same thing here.
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require(*modules: str):
    missing = [name for name in modules if not _available(name)]
    if not missing:
        loaded = [importlib.import_module(name) for name in modules]
        return loaded[0] if len(loaded) == 1 else tuple(loaded)
    message = "needs " + " and ".join(missing)
    # allow_module_level lets the same helper run at import time (where a
    # plain skip is a collection error) and inside a test.
    pytest.skip(
        message + "; run this file where the training stack is installed",
        allow_module_level=True,
    )
