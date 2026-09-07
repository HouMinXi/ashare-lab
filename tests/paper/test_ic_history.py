"""IC history must record lagged_ic_t5, never the live TRA `ic` (always NaN)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

from ashare_lab.paper.pipeline import append_ic_history

_MOD = "ashare_lab.paper.pipeline"


def _walk_back(_date: dt.date) -> dt.date:
    # Isolates the unit from the exchange calendar. Production walks
    # previous_trading_day (skips weekends/holidays); this mock subtracts
    # one calendar day so fixtures can pick T-5 as trade_date minus 5 days.
    return _date - dt.timedelta(days=1)


def test_append_records_t5_lagged_ic(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    t5_meta = pred_dir / "2026-08-26.meta.json"
    t5_meta.write_text(json.dumps({"lagged_ic_t5": -0.042, "ic": None}))

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (tmp_path / "data" / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_append_skips_missing_lagged_ic(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-09-04.parquet").touch()
    (pred_dir / "2026-08-28.meta.json").write_text(json.dumps({"ic": None}))

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-09-04.parquet", "2026-09-04")

    history = tmp_path / "data" / "ic_history.tsv"
    assert not history.exists()


def test_append_ignores_live_ic_field(tmp_path, monkeypatch):
    """A present live `ic` must not be recorded; that field is NaN out of sample."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-31.meta.json").write_text(json.dumps({"ic": 0.19}))
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": 0.096, "ic": None})
    )

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (tmp_path / "data" / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t0.096"]


def test_empty_placeholder_does_not_block_valid_ic(tmp_path, monkeypatch):
    """An old empty-IC row for T-5 must not keep a later valid value out."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.042, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\t\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_non_numeric_placeholder_does_not_block_valid_ic(tmp_path, monkeypatch):
    """A 'None' string from the old writer must not keep a later valid value out."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.042, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\tNone\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_finite_ic_is_not_overwritten(tmp_path, monkeypatch):
    """A numeric T-5 row stays; backfill does not rewrite it."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.099, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\t-0.042\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_nan_placeholder_does_not_block_valid_ic(tmp_path, monkeypatch):
    """A TSV nan token is numeric but not finite; replace it."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.042, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\tnan\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_inf_placeholder_does_not_block_valid_ic(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.042, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\tinf\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]


def test_placeholder_before_finite_same_date_keeps_finite(tmp_path, monkeypatch):
    """Corrupt TSV: placeholder then a finite row for the same date.
    Keep the finite value; do not wipe both and rewrite."""
    monkeypatch.setattr(f"{_MOD}.PROJECT_ROOT", tmp_path)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "2026-08-31.parquet").touch()
    (pred_dir / "2026-08-26.meta.json").write_text(
        json.dumps({"lagged_ic_t5": -0.099, "ic": None})
    )
    hist = tmp_path / "data"
    hist.mkdir()
    (hist / "ic_history.tsv").write_text("2026-08-26\tNone\n2026-08-26\t-0.042\n")

    with patch(f"{_MOD}.previous_trading_day", side_effect=_walk_back):
        append_ic_history(pred_dir / "2026-08-31.parquet", "2026-08-31")

    rows = (hist / "ic_history.tsv").read_text().strip().splitlines()
    assert rows == ["2026-08-26\t-0.042"]
