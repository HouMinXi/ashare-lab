"""Two contracts of _write_sidecar that had no test.

Both were verified by injection: reverting each fix turns the matching case red.
"""
from __future__ import annotations

import json
from pathlib import Path

from ashare_lab.paper.signal_quality import _write_sidecar


def test_a_key_that_is_both_cleared_and_set_keeps_the_new_value(tmp_path: Path) -> None:
    """Clearing runs before the update, so "clear and set the same key" means
    "set". Reversing the order would make the field disappear silently."""
    path = tmp_path / "d.meta.json"
    path.write_text(json.dumps({"lagged_ic_t5": 0.1}))

    _write_sidecar(path, {"lagged_ic_t5": 0.9}, clear=("lagged_ic_t5",))

    assert json.loads(path.read_text()) == {"lagged_ic_t5": 0.9}


def test_bytes_that_are_not_utf8_are_replaced_not_raised(tmp_path: Path) -> None:
    """A closed writer can leave a partial multi-byte sequence. The step that
    calls this must not die on it: the file is unreadable, so start over."""
    path = tmp_path / "d.meta.json"
    path.write_bytes(b"\xff\xfe{\"psi\": 0.5}")

    _write_sidecar(path, {"psi": 0.9})

    assert json.loads(path.read_text()) == {"psi": 0.9}
