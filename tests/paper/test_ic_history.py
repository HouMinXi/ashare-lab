"""IC history must record lagged_ic_t5, never the live TRA `ic` (always NaN)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

from ashare_lab.paper.pipeline import append_ic_history

_MOD = "ashare_lab.paper.pipeline"


def _walk_back(_date: dt.date) -> dt.date:
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
