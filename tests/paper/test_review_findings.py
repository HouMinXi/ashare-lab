"""Regressions found by the six-lens review of a23aff6..612b8b9.

Each case names the failure it prevents, because all five were introduced by
changes that already had tests passing.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from ashare_lab.paper import signal_quality as sq
from ashare_lab.paper.ledger import init_schema
from ashare_lab.research.attribution import compute_attribution, main

NEGATIVE = [-0.05 + 0.01 * ((i % 3) - 1) for i in range(60)]


# --------------------------------------------------------------- IC alert retry
def _seed_history(tmp_path) -> None:
    (tmp_path / "ic_history.tsv").write_text(
        "".join(f"2026-08-{i + 1:02d}\t{v}\n" for i, v in enumerate(NEGATIVE)),
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
    _seed_history(tmp_path)
    return tmp_path


def _step(tmp_path, trade_date="2026-08-06"):
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.steps = None
    ctx.trade_date = trade_date
    ctx.predictions_date_str = trade_date
    ctx.pred_path = tmp_path / f"{trade_date}.parquet"
    pd.DataFrame({"instrument": ["SH600000"], "score": [1.0]}).to_parquet(ctx.pred_path)
    sq.signal_quality_ic_step(ctx)


def test_a_failed_delivery_is_retried_on_the_next_run(wired):
    """The stamp records a delivery, not an intention.

    Writing it before the send meant one failed bridge call silenced the alert
    for a week: the failure and the success left the same state behind.
    """
    with patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=False):
        _step(wired, "2026-08-06")

    assert not (wired / ".ic_alert_stamp").exists()

    with patch(
        "ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True
    ) as alert:
        _step(wired, "2026-08-07")

    assert alert.call_count == 1
    assert (wired / ".ic_alert_stamp").read_text() == "2026-08-07"


def test_a_delivered_alert_stamps_and_then_throttles(wired):
    with patch("ashare_lab.paper.signal_quality.send_bridge_alert", return_value=True) as alert:
        _step(wired, "2026-08-06")
        _step(wired, "2026-08-07")

    assert alert.call_count == 1


# ------------------------------------------------------------------- attribution
def _nav_row(date_str, nav, cash, bench=None, hedge=0.0):
    row = {"trade_date": date_str, "total_nav": nav, "cash": cash,
           "benchmark_csi1000": bench}
    if hedge is not None:
        row["hedge_value"] = hedge
    return row


def test_exposure_excludes_the_hedge_leg():
    """Cash and hedge are not stock exposure; counting them overstates it 5x."""
    rows = [
        _nav_row("2026-08-03", 100_000.0, 50_000.0, 1000.0, hedge=40_000.0),
        _nav_row("2026-08-04", 100_000.0, 50_000.0, 1100.0, hedge=40_000.0),
    ]

    stats = compute_attribution(rows)

    assert stats["avg_exposure"] == pytest.approx(0.10)


def test_a_zero_starting_nav_reports_nothing_instead_of_raising():
    rows = [
        _nav_row("2026-08-03", 0.0, 0.0, 1000.0),
        _nav_row("2026-08-04", 100.0, 0.0, 1100.0),
    ]

    stats = compute_attribution(rows)

    assert stats["book_return"] is None
    assert stats["excess_vs_passive_pp"] is None


def test_a_hash_in_the_path_still_reads_that_file(tmp_path, capsys):
    """`file:path?mode=ro` with a raw '#' opens the prefix path instead.

    The tool would then report numbers from a different database -- or from an
    empty one it had just created -- while looking entirely plausible.
    """
    db = tmp_path / "live.db#snapshot"
    conn = sqlite3.connect(db)
    init_schema(conn)
    for date_str, nav in (("2026-08-03", 100_000.0), ("2026-08-04", 110_000.0)):
        conn.execute(
            "INSERT INTO nav (trade_date, cash, market_value, total_nav, benchmark_csi1000) "
            "VALUES (?, 0, 0, ?, 1000)",
            (date_str, nav),
        )
    conn.commit()
    conn.close()

    rc = main(["--db", str(db), "--since", "2026-08-01", "--json"])
    stats = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert stats["days"] == 2
    assert not (tmp_path / "live.db").exists(), "the prefix path must not be created"


def test_a_malformed_since_is_rejected(tmp_path):
    db = tmp_path / "paper.db"
    sqlite3.connect(db).close()

    with pytest.raises(SystemExit) as excinfo:
        main(["--db", str(db), "--since", "20260811"])

    assert excinfo.value.code == 2


# ------------------------------------------------------------------- zero topk
def test_a_zero_topk_does_not_crash_the_buy_step(tmp_path):
    """place_buys takes equity_nav as an argument, so it is evaluated even when
    the topk branch empties the buy list and never assigns it."""
    from tests.paper.ctx_factory import make_daily_ctx
    from ashare_lab.paper.pipeline import _step10_signal_generation

    (tmp_path / "predictions").mkdir()
    pred = tmp_path / "predictions" / "2025-06-20.parquet"
    pred.touch()
    conn = sqlite3.connect(tmp_path / "paper.db")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    prices = {f"SH60000{i}": {"close": 10.0 + i} for i in range(1, 6)}
    signals = {f"SH60000{i}": 0.9 - i * 0.05 for i in range(1, 6)}
    ctx = make_daily_ctx(
        conn=conn, pred_path=pred,
        paper_cfg={"initial_cash": 300_000.0, "topk": 0, "n_drop": 1,
                   "listing_min_days": 0, "liquidity_min_turnover": 0,
                   "turnover_cap": 0.2, "turnover_ramp_invested_ratio": 0.0},
        config={"cost_model": {"risk_degree": 0.95},
                "universe": {"listing_min_days": 0, "min_avg_turnover_20d": 0.0,
                             "exclude_close_above_cny": 300}},
        prices=prices, universe=list(prices),
        market_data={s: {"close": p["close"], "listing_days": 999,
                         "avg_turnover_20d": 1e9} for s, p in prices.items()},
    )

    with patch("ashare_lab.paper.pipeline.generate_signals", return_value=signals):
        _step10_signal_generation(ctx)

    assert conn.execute("SELECT COUNT(*) FROM orders WHERE side='buy'").fetchone()[0] == 0
