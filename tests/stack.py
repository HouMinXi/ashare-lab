"""Skips that say why, and a way to prove nothing was skipped.

Several research tests can only run where Qlib and torch are installed. A skip
is honest there, but it is also invisible: a host without the stack reports
green while never running the assertion. Setting ASHARE_REQUIRE_STACK=1 turns
those skips into failures, so a run that claims to have checked the training
path can prove it did.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_STRICT = {"1", "true", "yes", "on"}


def require(*modules: str):
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        loaded = [importlib.import_module(name) for name in modules]
        return loaded[0] if len(loaded) == 1 else tuple(loaded)
    message = "needs " + " and ".join(missing)
    if os.environ.get("ASHARE_REQUIRE_STACK", "").strip().lower() in _STRICT:
        pytest.fail(
            message + "; ASHARE_REQUIRE_STACK is set, so a skip would hide the "
            "assertions instead of reporting them"
        )
    pytest.skip(message + "; run this file where the training stack is installed")
