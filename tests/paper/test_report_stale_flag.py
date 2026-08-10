"""_step13_report stale-flag parsing: explicit '0' is valid, not stale.

Regression: the report step used to warn "treating as stale" for
ASHARE_USE_STALE='0' even though every downstream consumer treats only
'1' as stale -- the warning lied on every normal run.
"""

import logging
import sqlite3

import pytest

from ashare_lab.paper.pipeline import DailyRunContext, _step13_report


def _ctx():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE nav (trade_date TEXT, total_nav REAL)")
    ctx = DailyRunContext(
        trade_date="2026-08-10",
        force=False,
        steps={"nav"},  # not 'report': step exits before the heavy import
        pred_path=None,
        start_time=0.0,
        predictions_date_str="2026-08-10",
    )
    ctx.conn = conn
    return ctx, conn


def _run(ctx, conn, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ASHARE_USE_STALE", raising=False)
    else:
        monkeypatch.setenv("ASHARE_USE_STALE", value)
    _step13_report(ctx)
    conn.close()


def test_stale_flag_zero_is_quiet(monkeypatch, caplog):
    ctx, conn = _ctx()
    with caplog.at_level(logging.INFO):
        _run(ctx, conn, monkeypatch, "0")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("stale" in r.getMessage().lower() for r in warnings), [
        r.getMessage() for r in warnings
    ]


def test_stale_flag_one_skips_report(monkeypatch, caplog):
    ctx, conn = _ctx()
    with caplog.at_level(logging.INFO):
        _run(ctx, conn, monkeypatch, "1")
    assert any("skipping: stale predictions" in r.getMessage() for r in caplog.records)
    assert not any("unrecognized" in r.getMessage() for r in caplog.records)


def test_stale_flag_junk_warns_not_stale(monkeypatch, caplog):
    ctx, conn = _ctx()
    with caplog.at_level(logging.WARNING):
        _run(ctx, conn, monkeypatch, "yes")
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("unrecognized" in m and "not-stale" in m for m in messages), messages
