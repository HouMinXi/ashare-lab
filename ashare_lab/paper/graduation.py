"""Graduation gate for the 30-trading-day observation period.

Queries pipeline_runs to determine if the deployment has achieved
sufficient reliability (>=95% success rate over 30 trading days
with <=3 stale days) to graduate from paper trading.

One-shot notification guard: graduation notification fires only
once per graduation event via the graduation_status table.
"""

from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)


def check_graduation(
    conn: sqlite3.Connection,
    min_days: int = 30,
    min_rate: float = 0.95,
    max_stale: int = 3,
) -> tuple[bool, dict]:
    """Check whether the graduation gate passes.

    Queries the most recent pipeline_runs rows using a windowed query
    (ORDER BY trade_date DESC LIMIT N) and accumulates stats until the
    denominator (success + error) reaches *min_days*.  Stale days are
    excluded from the denominator (pause, not fail).

    Returns (passed, stats) where stats contains success, error, stale,
    denominator, rate, days_remaining, max_stale, and should_notify.
    """
    # Deduplicate: one row per trade_date (latest attempt wins).
    # Matches get_report's precedent: ORDER BY created_at DESC, id DESC.
    # Windowing is day-based, not row-based, so retried days are not
    # double-counted in the denominator.
    rows = conn.execute(
        "SELECT trade_date, status FROM ("
        "  SELECT trade_date, status,"
        "    ROW_NUMBER() OVER ("
        "      PARTITION BY trade_date"
        "      ORDER BY created_at DESC, id DESC"
        "    ) AS rn"
        "  FROM pipeline_runs"
        ") WHERE rn = 1 "
        "ORDER BY trade_date DESC "
        "LIMIT ?",
        (min_days + max_stale + 10,),
    ).fetchall()

    success = 0
    error = 0
    stale = 0

    # Accumulate until denominator reaches min_days
    for row in rows:
        status = row["status"]
        if status == "stale":
            stale += 1
        elif status == "success":
            success += 1
        else:
            error += 1
        # Stop once we have enough non-stale days
        if success + error >= min_days:
            break

    denominator = success + error
    rate = success / denominator if denominator > 0 else 0.0
    passed = (
        denominator >= min_days
        and rate >= min_rate
        and stale <= max_stale
    )

    # One-shot notification guard (R4M2)
    gs_row = conn.execute(
        "SELECT graduated_at, notified_at FROM graduation_status "
        "WHERE id = 1",
    ).fetchone()

    should_notify = False
    if passed:
        # Notify only if we haven't already
        if gs_row is None or gs_row["notified_at"] is None:
            should_notify = True
    else:
        # Gate regressed -- clear both graduated_at and notified_at
        # so a genuine recovery after regression can re-notify.
        # Oscillation is prevented by the 30-day window: the gate
        # needs 30 consecutive good days, so casual oscillation
        # around the threshold doesn't trigger repeated notifications.
        if gs_row is not None and gs_row["graduated_at"] is not None:
            conn.execute(
                "UPDATE graduation_status "
                "SET graduated_at = NULL, notified_at = NULL "
                "WHERE id = 1",
            )
            conn.commit()

    stats = {
        "success": success,
        "error": error,
        "stale": stale,
        "denominator": denominator,
        "rate": round(rate, 6),
        "days_remaining": max(0, min_days - denominator),
        "max_stale": max_stale,
        "should_notify": should_notify,
    }
    return passed, stats


def notify_graduation(conn: sqlite3.Connection, stats: dict) -> bool:
    """Send a WeChat graduation notification via iLink.

    Writes graduated_at and notified_at to graduation_status on
    successful delivery.  On failure, does NOT write timestamps so
    the next run retries (should_notify remains True).

    Returns True on success, False on failure.
    """
    try:
        import asyncio
        import aiohttp
        from ashare_lab.paper.report import (
            send_text_ilink, _get_secret,
        )
    except Exception:
        log.warning("graduation: failed to import iLink dependencies")
        return False

    text = (
        f"[graduation gate PASSED]\n"
        f"success: {stats['success']}, error: {stats['error']}, "
        f"stale: {stats['stale']}\n"
        f"rate: {stats['rate']:.1%}, "
        f"denominator: {stats['denominator']}"
    )

    try:
        token = _get_secret("ashare/weixin-token")
        chat_id = _get_secret("ashare/weixin-chat-id")
    except Exception:
        log.warning("graduation: failed to read iLink secrets")
        return False

    async def _send() -> dict:
        async with aiohttp.ClientSession() as session:
            return await send_text_ilink(session, token, chat_id, text)

    try:
        asyncio.run(_send())
    except Exception:
        log.warning("graduation: iLink delivery failed", exc_info=True)
        return False

    # Delivery succeeded -- write one-shot guard
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO graduation_status "
        "(id, graduated_at, notified_at) VALUES (1, ?, ?)",
        (now, now),
    )
    conn.commit()
    return True
