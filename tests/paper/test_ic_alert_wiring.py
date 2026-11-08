"""The IC alert answers to the monitor, once, and says what it saw."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from ashare_lab.paper import signal_quality as sq

NEGATIVE = [-0.05 + 0.01 * ((i % 3) - 1) for i in range(60)]
POSITIVE = [0.05 + 0.01 * ((i % 3) - 1) for i in range(60)]


def _seed_history(tmp_path, values) -> None:
    (tmp_path / "ic_history.tsv").write_text(
        "".join(f"2026-08-{i + 1:02d}\t{v}\n" for i, v in enumerate(values)),
        encoding="utf-8",
    )


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(sq, "IC_HISTORY_PATH", tmp_path / "ic_history.tsv")
    monkeypatch.setattr(sq, "previous_trading_day", lambda day: day)
    monkeypatch.setattr(sq, "compute_forward_returns", lambda *a, **k: pd.Series(dtype=float))
    monkeypatch.setattr(sq, "lagged_ic_for_date", lambda *a, **k: (0.10, 100))
    monkeypatch.delenv("ASHARE_SQ_DRYRUN", raising=False)
    pd.DataFrame({"instrument": ["SH600000"], "score": [1.0]}).to_parquet(
        tmp_path / "2026-08-06.parquet"
    )
    return tmp_path


def _ctx(tmp_path, trade_date="2026-08-06"):
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    ctx.pred_path = tmp_path / f"{trade_date}.parquet"
    return ctx


def _push(wired, trade_date="2026-08-06", **kwargs):
    with patch(
        "ashare_lab.paper.signal_quality.send_bridge_alert", **kwargs
    ) as alert:
        sq.signal_quality_ic_step(_ctx(wired, trade_date))
    return alert


def test_a_sustained_negative_edge_pushes_one_alert(wired):
    _seed_history(wired, NEGATIVE)

    alert = _push(wired, return_value=True)

    assert alert.call_count == 1
    body = alert.call_args.kwargs["body"]
    assert "均值" in body and "t(NW)" in body, body


def test_a_healthy_series_pushes_nothing(wired):
    _seed_history(wired, POSITIVE)

    alert = _push(wired, return_value=True)

    alert.assert_not_called()


def test_a_short_history_says_nothing(wired):
    _seed_history(wired, NEGATIVE[:3])

    alert = _push(wired, return_value=True)

    alert.assert_not_called()


def test_the_push_is_rate_limited_to_one_a_week(wired):
    _seed_history(wired, NEGATIVE)
    (wired / "2026-08-07.parquet").write_bytes((wired / "2026-08-06.parquet").read_bytes())

    with patch(
        "ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True
    ) as alert:
        sq.signal_quality_ic_step(_ctx(wired, "2026-08-06"))
        sq.signal_quality_ic_step(_ctx(wired, "2026-08-07"))

    assert alert.call_count == 1


def test_dry_run_computes_but_does_not_push(wired, monkeypatch):
    monkeypatch.setenv("ASHARE_SQ_DRYRUN", "1")
    _seed_history(wired, NEGATIVE)

    alert = _push(wired, return_value=True)

    alert.assert_not_called()


def test_a_failed_send_is_logged_not_raised(wired, caplog):
    _seed_history(wired, NEGATIVE)

    with caplog.at_level("ERROR"):
        alert = _push(wired, return_value=False)

    assert alert.call_count == 1
    assert "send failed" in caplog.text
