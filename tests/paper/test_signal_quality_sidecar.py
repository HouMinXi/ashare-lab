"""A sidecar must not keep an error that a later success has resolved."""

from __future__ import annotations

import json
from pathlib import Path

from ashare_lab.paper.signal_quality import _write_sidecar


def test_success_clears_a_previous_error(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-23.meta.json"
    _write_sidecar(path, {"lagged_ic_t5": None, "lagged_ic_error": "ArrowKeyError: boom"})

    _write_sidecar(path, {"lagged_ic_t5": 0.36, "lagged_ic_t5_n": 996}, clear=("lagged_ic_error",))

    meta = json.loads(path.read_text())
    assert meta["lagged_ic_t5"] == 0.36
    assert "lagged_ic_error" not in meta
    assert meta["lagged_ic_t5_n"] == 996, "keys outside clear must survive"


def test_clearing_is_limited_to_the_named_keys(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-24.meta.json"
    _write_sidecar(path, {"lagged_ic_t5": 0.1, "note": "keep me"})

    _write_sidecar(path, {"lagged_ic_t5": 0.2}, clear=("lagged_ic_error",))

    meta = json.loads(path.read_text())
    assert meta["lagged_ic_t5"] == 0.2
    assert meta["note"] == "keep me"


def test_without_clear_the_error_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-25.meta.json"
    _write_sidecar(path, {"lagged_ic_error": "still broken"})

    _write_sidecar(path, {"lagged_ic_t5": None})

    meta = json.loads(path.read_text())
    assert meta["lagged_ic_error"] == "still broken"
    assert meta["lagged_ic_t5"] is None


def test_success_clears_the_missing_parquet_reason(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-26.meta.json"
    _write_sidecar(
        path,
        {"lagged_ic_t5": None, "lagged_ic_t5_n": 0, "lagged_ic_t5_reason": "missing_t5_parquet"},
    )

    _write_sidecar(
        path,
        {"lagged_ic_t5": 0.11, "lagged_ic_t5_n": 998},
        clear=("lagged_ic_error", "lagged_ic_t5_reason"),
    )

    meta = json.loads(path.read_text())
    assert meta["lagged_ic_t5"] == 0.11
    assert "lagged_ic_t5_reason" not in meta


def test_a_non_object_sidecar_is_replaced_not_merged(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-27.meta.json"
    path.write_text("[1, 2, 3]")

    _write_sidecar(path, {"lagged_ic_t5": 0.2}, clear=("lagged_ic_error",))

    assert json.loads(path.read_text()) == {"lagged_ic_t5": 0.2}


def test_clearing_a_key_that_is_not_there_is_harmless(tmp_path: Path) -> None:
    path = tmp_path / "2026-10-28.meta.json"
    _write_sidecar(path, {"lagged_ic_t5": 0.1})

    _write_sidecar(path, {"lagged_ic_t5": 0.2}, clear=("lagged_ic_error",))

    assert json.loads(path.read_text()) == {"lagged_ic_t5": 0.2}
