"""The CSI1000 exit mechanism needs its marker to survive the insert.

_step7_csi1000_exits sells a holding that has dropped out of the index
universe, and it tags that order with cancel_reason='csi1000_exit' so the
next day can tell "I already queued this exit" from "this is a fresh
absence" (pipeline.py:1162 and :1174 both query by that value, and
engine_settle.py excludes orders carrying a reason from the normal forced
liquidation check).

insert_order never accepted the argument, so the call at pipeline.py:1185
raises TypeError the first time a held position leaves the universe, and
the idempotency queries could never have matched anything. The only test
covering that step patches it out, so nothing noticed.
"""

from __future__ import annotations

import sqlite3

import pytest

from ashare_lab.paper.ledger import get_connection, init_schema, insert_order

EXIT_REASON = "csi1000_exit"


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = get_connection(tmp_path / "paper.db")
    init_schema(connection)
    yield connection
    connection.close()


def _reason_of(connection: sqlite3.Connection, order_id: int) -> str | None:
    row = connection.execute(
        "SELECT cancel_reason FROM orders WHERE id=?", (order_id,)
    ).fetchone()
    assert row is not None, f"order {order_id} was not inserted"
    return row["cancel_reason"]


def test_an_exit_order_keeps_the_reason_it_was_created_with(
    conn: sqlite3.Connection,
) -> None:
    order_id = insert_order(
        conn,
        "2026-10-23",
        "SZ301029",
        "sell",
        100,
        None,
        "pending",
        0,
        "2026-10-22",
        cancel_reason=EXIT_REASON,
    )
    assert _reason_of(conn, order_id) == EXIT_REASON


def test_an_ordinary_order_carries_no_reason(conn: sqlite3.Connection) -> None:
    """engine_settle matches `cancel_reason IS NULL` to find ordinary orders.

    A default that wrote some placeholder string would quietly remove every
    signal-driven order from that check.
    """
    order_id = insert_order(
        conn, "2026-10-23", "SZ301029", "sell", 100, None, "pending", 0, "2026-10-22"
    )
    assert _reason_of(conn, order_id) is None


def test_the_exit_order_is_found_by_the_query_that_looks_for_it(
    conn: sqlite3.Connection,
) -> None:
    """The round trip the two-day absence rule depends on.

    Verbatim query from pipeline.py:1171-1176. Without it the step reads
    every absence as the first one and queues a duplicate exit sell daily.
    """
    insert_order(
        conn,
        "2026-10-23",
        "SZ301029",
        "sell",
        100,
        None,
        "pending",
        0,
        "2026-10-22",
        cancel_reason=EXIT_REASON,
    )
    found = conn.execute(
        "SELECT id FROM orders WHERE symbol=? AND side='sell' "
        "AND status IN ('pending','carry') "
        "AND cancel_reason = 'csi1000_exit'",
        ("SZ301029",),
    ).fetchone()
    assert found is not None, "the exit order cannot be found by its own marker"


def test_an_ordinary_order_does_not_answer_the_exit_query(
    conn: sqlite3.Connection,
) -> None:
    """Otherwise a signal-driven sell would be mistaken for a queued exit."""
    insert_order(
        conn, "2026-10-23", "SZ301029", "sell", 100, None, "pending", 0, "2026-10-22"
    )
    found = conn.execute(
        "SELECT id FROM orders WHERE symbol=? AND side='sell' "
        "AND status IN ('pending','carry') "
        "AND cancel_reason = 'csi1000_exit'",
        ("SZ301029",),
    ).fetchone()
    assert found is None


def test_cancel_reason_is_keyword_only(conn: sqlite3.Connection) -> None:
    """One argument too many must raise, not file the reason positionally.

    source="signal" is the optional positional slot just before
    cancel_reason, so a call with a single extra argument lands there and
    proves nothing. Two extras would fill cancel_reason if the keyword-only
    marker were removed; with the marker present the call is a TypeError.
    """
    with pytest.raises(TypeError):
        insert_order(
            conn,
            "2026-10-23",
            "SZ301029",
            "sell",
            100,
            None,
            "pending",
            0,
            "2026-10-22",
            "signal",
            EXIT_REASON,
        )
