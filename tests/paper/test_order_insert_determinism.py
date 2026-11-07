"""The ledger's sell rows must not depend on how a set happens to iterate.

Both sell loops build their symbol set from a Python set, and a set's
iteration order follows per-process string hashing.  What the fills do with
that order is nothing: each sell is sized against its own symbol's volume and
settlement re-sorts both sides by id before filling, so the same symbols fill
the same way whichever order they were inserted in.  The ids are a different
matter.  `orders.id` is AUTOINCREMENT, so a set-ordered loop writes a
different id-to-symbol mapping on every process, settlement walks in id order,
and a replay diffed against the recorded ledger then reports differences that
are not trading differences.

The pick itself is covered by test_sell_pick_determinism, which is where the
2026-10-22 incident actually lived: one seed sold SZ301029 and another sold
SZ301293 straight out of topk_dropout_orders, before any row was inserted.
That file never reaches the insert loops, so reverting either sort leaves it
green -- which is why this one drives the real steps.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ashare_lab.config import PROJECT_ROOT
from tests.paper.ctx_factory import make_daily_ctx

# Deliberately not alphabetical, so a set that iterates in hash or insertion
# order lands somewhere other than the sorted order this must produce.
SELLS = ["SZ000005", "SZ000003", "SZ000001", "SZ000004", "SZ000002"]
EXPECTED = sorted(SELLS)
SEEDS = ("1", "2", "3", "4", "5", "6")


def _sells_inserted_by_book_a(root: Path) -> list[str]:
    """Run the real step 10 and return sell symbols in the order they landed."""
    from unittest.mock import patch

    from ashare_lab.paper.ledger import get_connection, init_schema
    from ashare_lab.paper.pipeline import _step10_signal_generation

    os.chdir(root)
    # A prediction path inside the sandbox: with None the step resolves the
    # checkout's predictions directory, which a worktree symlinks to the live
    # one, and then reads -- and writes a sidecar into -- the real tree.
    pred = root / "predictions" / "2025-06-20.parquet"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.touch()

    conn = get_connection(root / "paper.db")
    init_schema(conn)

    config = {
        "paper": {
            "topk": 15, "n_drop": 1, "initial_cash": 300_000,
            "listing_min_days": 0, "liquidity_min_turnover": 0,
        },
        "cost_model": {"risk_degree": 0.95},
        "universe": {"exclude_close_above_cny": 300},
    }
    prices = {sym: {"close": 10.0, "change": 0.01, "volume": 1e6,
                    "factor": 1.0, "threshold": 0.099} for sym in SELLS}

    ctx = make_daily_ctx(
        trade_date="2025-06-20", conn=conn, config=config, pred_path=pred,
        prices=prices, positions={sym: {"qty": 100} for sym in SELLS},
        cash=300_000.0, total_nav=300_000.0, buying_halted=True,
        market_data={sym: {"close": 10.0, "listing_days": 999,
                           "avg_turnover_20d": 1e9} for sym in SELLS},
    )

    # Buying halted keeps this to the sell side; the sell list is fixed so the
    # only thing under test is the order the orders are inserted in.  Both
    # names are bound at the top of the pipeline module, so the patch has to
    # land there.
    with patch("ashare_lab.paper.pipeline.generate_signals", return_value={}), \
         patch("ashare_lab.paper.pipeline.topk_dropout_orders", return_value=(list(SELLS), [])):
        _step10_signal_generation(ctx)

    rows = conn.execute(
        "SELECT symbol FROM orders WHERE side='sell' ORDER BY id"
    ).fetchall()
    conn.close()
    return [r["symbol"] for r in rows]


def _sells_inserted_by_book_b(root: Path) -> list[str]:
    """Same, through the shadow book that feeds the G3 comparison."""
    from unittest.mock import patch

    from ashare_lab.paper.ledger import get_connection, init_schema
    from ashare_lab.paper.pipeline import _run_book_b

    os.chdir(root)
    prod = root / "paper.db"
    conn = get_connection(prod)
    init_schema(conn)
    conn.execute(
        "INSERT OR REPLACE INTO nav (trade_date, cash, market_value, total_nav) "
        "VALUES (?, ?, ?, ?)", ("2026-08-01", 1_000_000.0, 0.0, 1_000_000.0),
    )
    conn.commit()
    conn.close()

    pred = root / "predictions" / "2026-08-15.parquet"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.touch()

    ctx = make_daily_ctx(
        trade_date="2026-08-15", db_path=prod, pred_path=pred,
        paper_cfg={"initial_cash": 1_000_000.0, "topk": 15, "n_drop": 1,
                   "listing_min_days": 60, "turnover_cap": None},
        config={
            "cost_model": {"risk_degree": 0.95},
            "universe": {"listing_min_days": 60, "min_avg_turnover_20d": 0.0,
                         "exclude_close_above_cny": 300.0},
            "carry_days": 3, "slippage": 0.001, "volume_participation_pct": 0.05,
        },
        prices={sym: {"close": 10.0, "volume": 10_000_000.0} for sym in SELLS},
        universe=list(SELLS),
        market_data={sym: {"close": 10.0, "listing_days": 100,
                           "avg_turnover_20d": 1e8} for sym in SELLS},
        cash=1_000_000.0, total_nav=1_000_000.0, buying_halted=True,
        hedge_state=None,
    )

    # First run bootstraps the shadow book; the second is the one that inserts.
    # The multiplier is read through timing_multiplier_detail, not the
    # single-value wrapper.
    with patch("ashare_lab.paper.pipeline.timing_multiplier_detail",
               return_value=(1.0, False)), \
         patch("ashare_lab.paper.signal.generate_signals", return_value={}):
        _run_book_b(ctx, "none")

    book = get_connection(root / "paper_b_none.db")
    for sym in SELLS:
        book.execute(
            "INSERT INTO positions (trade_date, symbol, qty, avg_cost, market_value, buy_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-08-15", sym, 100, 10.0, 1000.0, "2026-08-15"),
        )
    book.commit()
    book.close()

    with patch("ashare_lab.paper.pipeline.timing_multiplier_detail",
               return_value=(1.0, False)), \
         patch("ashare_lab.paper.signal.generate_signals", return_value={}), \
         patch("ashare_lab.paper.signal.topk_dropout_orders", return_value=(list(SELLS), [])):
        _run_book_b(ctx, "none")

    book = get_connection(root / "paper_b_none.db")
    rows = book.execute(
        "SELECT symbol FROM orders WHERE side='sell' ORDER BY id"
    ).fetchall()
    book.close()
    return [r["symbol"] for r in rows]


def _run_across_seeds(driver, root: Path) -> dict[str, list[str]]:
    """Run one driver per hash seed, each in its own process.

    The directory belongs to this process, so the children leave nothing
    behind once it is removed; a set's iteration order is only observable
    across processes.
    """
    script = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_order_insert_determinism as m\n"
        "from pathlib import Path\n"
        f"print(json.dumps(m.{driver.__name__}(Path(os.environ['CASE_DIR']))))\n"
    )
    results: dict[str, list[str]] = {}
    try:
        for seed in SEEDS:
            child = root / f"seed-{seed}"
            child.mkdir()
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["CASE_DIR"] = str(child)
            done = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, cwd=str(child),
                timeout=300, check=False,
            )
            assert done.returncode == 0, f"seed {seed} failed: {done.stderr.strip()[-800:]}"
            results[seed] = json.loads(done.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return results


def _assert_only_one_order(results: dict[str, list[str]], what: str) -> None:
    orders = {seed: tuple(order) for seed, order in results.items()}
    assert len(set(orders.values())) == 1, f"{what} follows the hash seed: {results}"
    landed = next(iter(orders.values()))
    assert landed == tuple(EXPECTED), f"expected {EXPECTED}, got {list(landed)}"


def test_book_a_inserts_sells_in_symbol_order_across_seeds() -> None:
    _assert_only_one_order(
        _run_across_seeds(_sells_inserted_by_book_a,
                          Path(tempfile.mkdtemp(prefix="sell-order-a-"))),
        "book A's insert order",
    )


def test_book_b_inserts_sells_in_symbol_order_across_seeds() -> None:
    _assert_only_one_order(
        _run_across_seeds(_sells_inserted_by_book_b,
                          Path(tempfile.mkdtemp(prefix="sell-order-b-"))),
        "book B's insert order",
    )


def test_both_settlement_queries_ask_for_id_order() -> None:
    """Both books read pending orders and must ask for the id order they walk.

    Parsed rather than grepped: the query is assembled from adjacent string
    literals, so a text search both breaks on a harmless reflow and can be
    satisfied by a comment.  The clause has no observable effect today --
    SQLite scans the table and settlement sorts by id anyway -- which is why
    it needs a pin instead of a behavioural test.
    """
    source = (PROJECT_ROOT / "ashare_lab" / "paper" / "pipeline.py").read_text()
    queries: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "execute"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        sql = node.args[0].value
        if not isinstance(sql, str):
            continue
        flat = " ".join(sql.split())
        if "status IN ('pending','carry')" in flat and "trade_date <=" in flat:
            queries.append((node.lineno, flat))
    assert len(queries) == 2, (
        "expected the settlement query once per book, found "
        f"{[(line, q[:60]) for line, q in queries]}"
    )
    for line, sql in queries:
        assert "ORDER BY id" in sql, (
            f"settlement query at pipeline.py:{line} no longer asks for id order: {sql}"
        )
