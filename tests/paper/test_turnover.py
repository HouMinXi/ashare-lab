"""Tests for the daily turnover budget (L3).

T1: cap not binding -> orders unchanged
T2: cap binding -> uniform factor, total <= cap, WARNING for 0-lot
T3: forced sells / IPO / carried excluded from numerator and scaling
T4: ramp: day k caps follow 50,44,38,32,26,20 then normal cap
T5: ramp persists across a simulated restart (reload from ledger)
T6: Book B binds independently of Book A (different NAV)
T7: integration through _step10 path (direct ctx, cap on/off)
T8: bug-injection at the call site (bypass cap -> all orders inserted)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ashare_lab.paper.ledger import get_connection, init_schema
from ashare_lab.paper.turnover import (
    cap_rotation_buys,
    load_turnover_ramp,
    save_turnover_ramp,
    reset_turnover_ramp,
    check_ramp_activation,
)
from tests.paper.ctx_factory import make_daily_ctx


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cd_tmp_path(tmp_path, monkeypatch):
    """Keep artifact writes inside tmp_path.

    The _step10 integration tests reach the daily comparison artifact writer,
    which builds a relative path. Without this the suite drops fixture
    artifacts into the real experiments/control_books/ directory whenever it
    runs from a working checkout.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def turnover_conn(tmp_path: Path) -> sqlite3.Connection:
    """In-memory SQLite with paper_state table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def prices_5() -> dict[str, dict]:
    """5 stocks with known prices."""
    return {
        "SZ000001": {"close": 10.0},
        "SZ000002": {"close": 20.0},
        "SZ000003": {"close": 30.0},
        "SZ000004": {"close": 40.0},
        "SZ000005": {"close": 50.0},
    }


# -------------------------------------------------------------------
# T1: cap not binding -> orders unchanged
# -------------------------------------------------------------------


def test_cap_not_binding_unchanged(prices_5):
    """When total planned buy value < cap, return buy_syms unchanged."""
    buy_syms = ["SZ000001", "SZ000002"]
    target_value = 10000.0  # 2 * 10000 = 20000
    equity_nav = 300000.0
    cap = 0.20  # 20% of 300K = 60K; 20K < 60K -> not binding
    ramp_state = {"active": False, "k": 0}

    capped, _, effective_cap, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    assert capped == buy_syms
    assert effective_cap == cap


# -------------------------------------------------------------------
# T2: cap binding -> ranked prefix at full size
# -------------------------------------------------------------------


def test_cap_funds_a_ranked_prefix_at_full_size(prices_5):
    """The budget limits how many names move, not how large each one is."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0   # 5 * 20000 = 100000 planned
    equity_nav = 300000.0
    cap = 0.10               # 30K budget -> room for one full position

    capped, value, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, {"active": False, "k": 0},
    )

    assert capped == ["SZ000001"], f"expected the top-ranked name, got {capped}"
    assert value == target_value, f"a funded name keeps its full size, got {value}"


def test_an_empty_book_still_buys_when_the_cap_binds():
    """Twelve planned buys used to scale to a quarter position each and vanish.

    This is the 2026-08 ledger shape: three names held, twelve wanted, a 20%
    cap.  The book bought nothing for six weeks.
    """
    symbols = [f"SZ{i:06d}" for i in range(12)]
    prices = {s: {"close": 60.0} for s in symbols}
    nav = 275_000.0
    target_value = nav * 0.95 / 15       # 17,417
    capped, value, _, _ = cap_rotation_buys(
        symbols, target_value, prices, nav, 0.20, {"active": False, "k": 0},
    )

    assert len(capped) == 3, f"expected three full positions, got {capped}"
    assert capped == symbols[:3], "the prefix must follow rank order"
    assert value == target_value, f"positions must not be scaled down, got {value}"


def test_the_budget_contracts_the_trade_instead_of_cancelling_it(prices_5):
    """Less room than one position, more than one lot: buy the top name with it."""
    buy_syms = ["SZ000001", "SZ000002"]
    target_value = 20000.0
    equity_nav = 300000.0
    cap = 0.05               # 15K budget, a full position needs 20K

    capped, value, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, {"active": False, "k": 0},
    )

    assert capped == ["SZ000001"], f"expected the top name, got {capped}"
    assert value == 15000.0, f"the trade absorbs the whole budget, got {value}"


def test_a_name_that_cannot_buy_one_lot_is_skipped_for_a_cheaper_one():
    """Funding an expensive name would round to zero shares and waste the slot."""
    symbols = ["EXPENSIVE", "CHEAP1", "CHEAP2"]
    prices = {"EXPENSIVE": {"close": 500.0}, "CHEAP1": {"close": 10.0},
              "CHEAP2": {"close": 20.0}}
    nav = 1_000_000.0
    target_value = 10_000.0          # 500 yuan needs 50,000 for one lot
    capped, _, _, _ = cap_rotation_buys(
        symbols, target_value, prices, nav, 0.022, {"active": False, "k": 0},
    )
    assert capped == ["CHEAP1", "CHEAP2"], f"expected the lot-feasible names, got {capped}"


def test_the_fallback_takes_the_first_name_that_can_buy_a_lot():
    """The budget must not stop at a name whose price it cannot cover."""
    symbols = ["EXPENSIVE", "CHEAP"]
    prices = {"EXPENSIVE": {"close": 500.0}, "CHEAP": {"close": 10.0}}
    capped, value, _, _ = cap_rotation_buys(
        symbols, 100_000.0, prices, 1_000_000.0, 0.005, {"active": False, "k": 0},
    )
    assert capped == ["CHEAP"], f"expected the affordable name, got {capped}"
    assert value == 5_000.0


def test_the_funded_prefix_stays_inside_the_budget():
    """Aggregate spend is bounded by the cap, not just by the name count."""
    symbols = [f"S{i:03d}" for i in range(9)]
    prices = {s: {"close": 33.0} for s in symbols}
    nav = 300_000.0
    target_value = 19_000.0
    cap = 0.20
    capped, value, _, _ = cap_rotation_buys(
        symbols, target_value, prices, nav, cap, {"active": False, "k": 0},
    )
    from ashare_lab.paper.engine import round_lots
    spend = sum(round_lots(value / prices[s]["close"], "buy") * prices[s]["close"]
                for s in capped)
    assert spend <= cap * nav, f"spend {spend} exceeds cap {cap * nav}"
    assert len(capped) == 3


def test_a_zero_target_selects_nothing():
    capped, _, _, _ = cap_rotation_buys(
        ["SZ000001"], 0.0, {"SZ000001": {"close": 10.0}}, 300_000.0, 0.20,
        {"active": False, "k": 0},
    )
    assert capped == []


def test_an_exact_budget_ratio_is_not_lost_to_floating_point():
    """floor(cap/target) must not drop a name when the ratio is integral.

    At this size the ratio is exactly 3.0 under true division while Python's
    floor-division operator returns 2.0 for the same pair, so a rewrite to
    ``//`` would silently fund one name fewer than the budget allows.
    """
    nav = 1_000_000.0
    target_value = nav * 0.95 / 6
    prices = {f"S{i}": {"close": 1000.0} for i in range(6)}
    capped, _, _, _ = cap_rotation_buys(
        list(prices), target_value, prices, nav, 0.475, {"active": False, "k": 0},
    )
    assert len(capped) == 3, f"expected three funded names, got {len(capped)}"


def test_a_budget_below_one_lot_sends_nothing(prices_5):
    """Below a single lot there is no order to send; that is a no-op, not a bug."""
    buy_syms = ["SZ000001", "SZ000005"]
    target_value = 1000.0
    equity_nav = 300000.0
    cap = 0.001              # 300 yuan, the cheapest lot here costs 1000

    capped, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, {"active": False, "k": 0},
    )

    assert capped == []


# -------------------------------------------------------------------
# T3: forced sells / IPO / carried excluded
# -------------------------------------------------------------------


def test_forced_sells_excluded(prices_5):
    """Forced sells are not in buy_syms, so they're naturally excluded."""
    # The caller (pipeline) already excludes forced_sells from buy_syms.
    # This test verifies cap_rotation_buys doesn't touch sells at all.
    buy_syms = ["SZ000001"]  # Only discretionary buy
    target_value = 10000.0
    equity_nav = 300000.0
    cap = 0.20
    ramp_state = {"active": False, "k": 0}

    capped, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, cap, ramp_state,
    )
    assert capped == ["SZ000001"]


# -------------------------------------------------------------------
# T4: ramp day k caps follow 50,44,38,32,26,20
# -------------------------------------------------------------------


def test_ramp_day_caps():
    """Ramp cap decays linearly from 50% to 20% over 5 days."""
    ramp_cap = 0.50
    normal_cap = 0.20
    ramp_days = 5

    expected = [0.50, 0.44, 0.38, 0.32, 0.26, 0.20]
    for k, exp in enumerate(expected):
        if k < ramp_days:
            cap = ramp_cap - (ramp_cap - normal_cap) * k / ramp_days
        else:
            cap = normal_cap
        assert abs(cap - exp) < 1e-10, f"k={k}: expected {exp}, got {cap}"


def test_ramp_active_caps(prices_5):
    """When ramp is active, effective cap is ramp cap, not normal cap."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total
    equity_nav = 300000.0
    normal_cap = 0.10  # 30K
    ramp_state = {
        "active": True, "k": 0, "entry_date": "2026-07-28",
        "ramp_cap": 0.50, "ramp_days": 5,
    }
    # ramp cap at k=0 = 0.50 -> 150K; 100K < 150K -> not binding
    capped, _, effective_cap, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, normal_cap, ramp_state,
    )
    assert len(capped) == 5
    assert abs(effective_cap - 0.50) < 1e-10


def test_ramp_expires(prices_5):
    """After ramp_days, ramp deactivates and normal cap applies."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total
    equity_nav = 300000.0
    normal_cap = 0.10  # 30K
    ramp_state = {
        "active": True, "k": 5, "entry_date": "2026-07-28",
        "ramp_cap": 0.50, "ramp_days": 5,
    }
    # k=5 >= ramp_days=5 -> ramp expires, normal cap 0.10 -> 30K
    # 100K > 30K -> binding, factor = 0.3
    capped, _, effective_cap, state = cap_rotation_buys(
        buy_syms, target_value, prices_5, equity_nav, normal_cap, ramp_state,
    )
    assert effective_cap == normal_cap
    assert state["active"] is False


# -------------------------------------------------------------------
# T5: ramp persists across simulated restart
# -------------------------------------------------------------------


def test_ramp_persistence(turnover_conn):
    """Ramp state survives save + reload (simulates restart)."""
    save_turnover_ramp(turnover_conn, "2026-07-28", 3)
    turnover_conn.commit()

    # Simulate restart: load from DB
    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["entry_date"] == "2026-07-28"
    assert loaded["k"] == 3
    assert loaded["active"] is True


def test_ramp_reset(turnover_conn):
    """reset_turnover_ramp clears the ramp state."""
    save_turnover_ramp(turnover_conn, "2026-07-28", 2)
    turnover_conn.commit()

    reset_turnover_ramp(turnover_conn)
    turnover_conn.commit()

    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["active"] is False
    assert loaded["entry_date"] is None


def test_ramp_no_state(turnover_conn):
    """load_turnover_ramp returns inactive when no state exists."""
    loaded = load_turnover_ramp(turnover_conn)
    assert loaded["active"] is False
    assert loaded["k"] == 0


# -------------------------------------------------------------------
# T6: Book B binds independently
# -------------------------------------------------------------------


def test_book_b_independent(prices_5):
    """Book B with different NAV binds independently."""
    buy_syms = ["SZ000001", "SZ000002", "SZ000003", "SZ000004", "SZ000005"]
    target_value = 20000.0  # 100000 total

    # Book A: NAV=300K, cap=20% -> 60K; 100K > 60K -> binding
    cap_a, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, 300000.0, 0.20,
        {"active": False, "k": 0},
    )
    # Book B: NAV=600K, cap=20% -> 120K; 100K < 120K -> not binding
    cap_b, _, _, _ = cap_rotation_buys(
        buy_syms, target_value, prices_5, 600000.0, 0.20,
        {"active": False, "k": 0},
    )
    # A: 5 x 20,000 = 100,000 against a 60,000 budget funds three.
    # B: 100,000 against 120,000 does not bind, so all five go.
    assert len(cap_a) == 3, f"book A should fund three names, got {cap_a}"
    assert len(cap_b) == 5, f"book B should be uncapped, got {cap_b}"


# -------------------------------------------------------------------
# T7: check_ramp_activation
# -------------------------------------------------------------------


def test_ramp_activation_low_invested():
    """Ramp activates when invested ratio < threshold."""
    positions = {"SZ000001": {"market_value": 50000}}
    equity_nav = 300000.0
    # invested_ratio = 50K/300K = 0.167 < 0.30 -> activate
    assert check_ramp_activation(equity_nav, positions, 0.30) is True


def test_ramp_activation_high_invested():
    """Ramp does NOT activate when invested ratio >= threshold."""
    positions = {"SZ000001": {"market_value": 250000}}
    equity_nav = 300000.0
    # invested_ratio = 250K/300K = 0.833 >= 0.30 -> don't activate
    assert check_ramp_activation(equity_nav, positions, 0.30) is False


def test_ramp_activation_empty_positions():
    """Empty positions -> invested_ratio=0 -> ramp activates."""
    assert check_ramp_activation(300000.0, {}, 0.30) is True


# -------------------------------------------------------------------
# T7: Integration through _step10 path (direct ctx construction)
# -------------------------------------------------------------------


def _make_ctx(conn, signals, prices, config, *, positions=None, tmp_path=None):
    """Build a minimal DailyRunContext for _step10 testing."""
    pred_path = None
    if tmp_path:
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(exist_ok=True)
        pred_path = pred_dir / "2025-06-20.parquet"
        pred_path.touch()
    # 999 days and 1e9 turnover keep every symbol past the universe filters
    return make_daily_ctx(
        conn=conn, config=config, pred_path=pred_path, prices=prices,
        positions=positions, signals=signals,
        market_data={sym: {"close": p["close"], "listing_days": 999,
                           "avg_turnover_20d": 1e9} for sym, p in prices.items()},
    )


class TestTurnoverCapIntegration:
    """T7: Integration through the real _step10 path."""

    def test_cap_on_binding_day(self, turnover_conn, tmp_path):
        """Cap ON: fewer buy orders than uncapped."""
        from ashare_lab.paper.pipeline import _step10_signal_generation
        from ashare_lab.paper.ledger import init_schema
        from unittest.mock import patch as P

        init_schema(turnover_conn)
        config = {
            "paper": {
                "topk": 15, "n_drop": 1, "initial_cash": 300_000,
                "listing_min_days": 0, "liquidity_min_turnover": 0,
                "turnover_cap": 0.20, "turnover_ramp_cap": 0.50,
                "turnover_ramp_days": 5, "turnover_ramp_invested_ratio": 0.0,
            },
            "cost_model": {"risk_degree": 0.95},
            "universe": {"exclude_close_above_cny": 300},
        }
        prices = {f"SH60000{i}": {"close": 10.0 + i, "change": 0.01,
                                   "volume": 1e6, "factor": 1.0,
                                   "threshold": 0.099}
                  for i in range(1, 6)}
        signals = {f"SH60000{i}": 0.95 - i * 0.05 for i in range(1, 6)}

        ctx = _make_ctx(turnover_conn, signals, prices, config, tmp_path=tmp_path)
        # Inject: mock insert_order to capture calls
        insert_calls = []
        def mock_insert(conn, td, sym, side, qty, *a, **kw):
            insert_calls.append((sym, side, qty))
        with P("ashare_lab.paper.pipeline.insert_order", side_effect=mock_insert), \
             P("ashare_lab.paper.pipeline.generate_signals", return_value=signals):
            _step10_signal_generation(ctx)

        buy_calls = [c for c in insert_calls if c[1] == "buy"]
        # 300,000 * 0.20 = 60,000 of budget against a 19,000 target funds three
        # positions, each at full size.  The old contract scaled all five down
        # instead; that is what deleted buys once a scaled position fell below
        # one lot, so the expectation changes with the contract.
        assert len(buy_calls) == 3, f"expected three funded positions, got {buy_calls}"
        for sym, side, qty in buy_calls:
            price = prices[sym]["close"]
            full_qty = int(19000 / price // 100) * 100
            assert qty == full_qty, f"{sym}: {qty} is not the full position {full_qty}"

    def test_cap_off_full_orders(self, turnover_conn, tmp_path):
        """Cap OFF: all buy orders inserted."""
        from ashare_lab.paper.pipeline import _step10_signal_generation
        from ashare_lab.paper.ledger import init_schema
        from unittest.mock import patch as P

        init_schema(turnover_conn)
        config = {
            "paper": {
                "topk": 15, "n_drop": 1, "initial_cash": 300_000,
                "listing_min_days": 0, "liquidity_min_turnover": 0,
                # No turnover_cap key
            },
            "cost_model": {"risk_degree": 0.95},
            "universe": {"exclude_close_above_cny": 300},
        }
        prices = {f"SH60000{i}": {"close": 10.0 + i, "change": 0.01,
                                   "volume": 1e6, "factor": 1.0,
                                   "threshold": 0.099}
                  for i in range(1, 6)}
        signals = {f"SH60000{i}": 0.95 - i * 0.05 for i in range(1, 6)}

        ctx = _make_ctx(turnover_conn, signals, prices, config, tmp_path=tmp_path)
        insert_calls = []
        def mock_insert(conn, td, sym, side, qty, *a, **kw):
            insert_calls.append((sym, side, qty))
        with P("ashare_lab.paper.pipeline.insert_order", side_effect=mock_insert), \
             P("ashare_lab.paper.pipeline.generate_signals", return_value=signals):
            _step10_signal_generation(ctx)

        buy_calls = [c for c in insert_calls if c[1] == "buy"]
        assert len(buy_calls) == 5

    def test_ramp_k_persists(self, turnover_conn):
        """Ramp k advances and persists across simulated restarts."""
        from ashare_lab.paper.turnover import (
            load_turnover_ramp, save_turnover_ramp,
        )
        save_turnover_ramp(turnover_conn, "2025-06-20", 1)
        turnover_conn.commit()

        state = load_turnover_ramp(turnover_conn)
        assert state["k"] == 1
        assert state["entry_date"] == "2025-06-20"

        save_turnover_ramp(turnover_conn, "2025-06-20", state["k"] + 1)
        turnover_conn.commit()

        state2 = load_turnover_ramp(turnover_conn)
        assert state2["k"] == 2


# -------------------------------------------------------------------
# T8: bug-injection at call sites
# -------------------------------------------------------------------


class TestTurnoverBugInjection:
    """T8: Delete cap call -> integration test RED; restore -> GREEN."""

    def test_inject_step10_cap_bypass(self, turnover_conn, tmp_path):
        """Cap call bypassed -> all buy orders inserted."""
        from ashare_lab.paper.pipeline import _step10_signal_generation
        from ashare_lab.paper import turnover as turnover_mod
        from ashare_lab.paper.ledger import init_schema
        from unittest.mock import patch as P

        init_schema(turnover_conn)
        config = {
            "paper": {
                "topk": 15, "n_drop": 1, "initial_cash": 300_000,
                "listing_min_days": 0, "liquidity_min_turnover": 0,
                "turnover_cap": 0.20, "turnover_ramp_cap": 0.50,
                "turnover_ramp_days": 5, "turnover_ramp_invested_ratio": 0.0,
            },
            "cost_model": {"risk_degree": 0.95},
            "universe": {"exclude_close_above_cny": 300},
        }
        prices = {f"SH60000{i}": {"close": 10.0 + i, "change": 0.01,
                                   "volume": 1e6, "factor": 1.0,
                                   "threshold": 0.099}
                  for i in range(1, 6)}
        signals = {f"SH60000{i}": 0.95 - i * 0.05 for i in range(1, 6)}

        ctx = _make_ctx(turnover_conn, signals, prices, config, tmp_path=tmp_path)

        # Inject: bypass cap_rotation_buys
        def noop_cap(buy_syms, target_value, prices, equity_nav, cap, ramp_state):
            return buy_syms, target_value, cap, ramp_state

        insert_calls = []
        def mock_insert(conn, td, sym, side, qty, *a, **kw):
            insert_calls.append((sym, side, qty))

        with P("ashare_lab.paper.pipeline.insert_order", side_effect=mock_insert), \
             P("ashare_lab.paper.pipeline.generate_signals", return_value=signals), \
             P.object(turnover_mod, "cap_rotation_buys", side_effect=noop_cap):
            _step10_signal_generation(ctx)

        buy_calls = [c for c in insert_calls if c[1] == "buy"]
        # With cap bypassed: all 5 orders inserted
        assert len(buy_calls) == 5, (
            f"Cap bypassed: expected 5 buys, got {len(buy_calls)}"
        )


# -------------------------------------------------------------------
# Book B integration test
# -------------------------------------------------------------------


class TestBookBCapIntegration:
    """Book B turnover cap integration test."""

    def test_book_b_cap_independent(self, tmp_path):
        """Book B cap binds based on its own NAV, independent of Book A."""
        from ashare_lab.paper.pipeline import _run_book_b, DailyRunContext
        from ashare_lab.paper.ledger import init_schema, snapshot_positions, record_nav
        from unittest.mock import MagicMock, patch
        from ashare_lab.paper.risk import RiskCheckResult

        # Create prod DB with positions (invested_ratio > 0.30, no ramp)
        prod_db = tmp_path / "paper.db"
        prod_conn = get_connection(prod_db)
        init_schema(prod_conn)
        snapshot_positions(prod_conn, "2025-06-19", {
            "SH600001": {"qty": 1000, "avg_cost": 10.0, "market_value": 11000.0},
            "SH600002": {"qty": 2000, "avg_cost": 20.0, "market_value": 42000.0},
            "SH600003": {"qty": 3000, "avg_cost": 30.0, "market_value": 93000.0},
            "SH600004": {"qty": 1500, "avg_cost": 40.0, "market_value": 64000.0},
        })
        record_nav(prod_conn, "2025-06-19", 90000.0, 210000.0, 300000.0,
                   None, None, None, None)
        prod_conn.commit()
        prod_conn.close()

        config = {
            "paper": {
                "db_path": str(prod_db),
                "topk": 15, "n_drop": 1, "initial_cash": 300_000,
                "listing_min_days": 0, "liquidity_min_turnover": 0,
                "turnover_cap": 0.20,
                "turnover_ramp_invested_ratio": 0.0,
            },
            "cost_model": {"risk_degree": 0.95},
            "universe": {"exclude_close_above_cny": 300},
        }
        prices = {f"SH60000{i}": {"close": 10.0 + i, "change": 0.01,
                                   "volume": 1e6, "factor": 1.0,
                                   "threshold": 0.099}
                  for i in range(1, 6)}
        signals = {f"SH60000{i}": 0.95 - i * 0.05 for i in range(1, 6)}

        (tmp_path / "predictions").mkdir(exist_ok=True)
        (tmp_path / "predictions" / "2025-06-20.parquet").touch()

        ctx = DailyRunContext(
            trade_date="2025-06-20", force=False, steps=None,
            pred_path=tmp_path / "predictions" / "2025-06-20.parquet",
            start_time=0.0, predictions_date_str="2025-06-20",
        )
        ctx.conn = prod_conn
        ctx.config = config
        ctx.paper_cfg = config["paper"]
        ctx.db_path = prod_db
        ctx.prices = prices
        ctx.current_positions = {}
        ctx.cash = 300_000.0
        ctx.total_nav = 300_000.0
        ctx.risk_result = RiskCheckResult(
            buying_halted=False, forced_sells={},
            blocked_industries=set(), blocked_rebuys=set(),
            topk_override=None, cooldown_entries={},
            suspension_risk={}, shadow_log=None,
        )
        ctx.hedge_state = MagicMock(active=False)
        ctx.hedge_symbols = set()
        ctx.market_data = {s: {"close": prices[s]["close"],
                                "listing_days": 999,
                                "avg_turnover_20d": 1e9}
                           for s in prices}
        ctx.universe_symbols = list(prices.keys())
        ctx.ipo_listing_syms = set()
        ctx.st_names = set()
        ctx.benchmarks = {"csi300": 4000.0, "csi1000": 6000.0}

        with patch("ashare_lab.paper.pipeline.timing_multiplier", return_value=1.0), \
             patch("ashare_lab.paper.signal.generate_signals", return_value=signals):
            _run_book_b(ctx)

        # Query Book B DB for inserted orders
        book_b_path = tmp_path / "paper_b_none.db"
        assert book_b_path.exists(), "Book B DB not created"
        book_conn = get_connection(book_b_path)
        orders = book_conn.execute(
            "SELECT symbol, side, target_qty FROM orders WHERE side='buy'"
        ).fetchall()
        book_conn.close()

        # Budget 18,000 against a 5,700 target funds three positions at full
        # size.  The old contract scaled all five by 0.63 and kept five partial
        # orders -- the shape that let a three-name book stop buying entirely.
        assert len(orders) == 3, f"expected three funded positions, got {len(orders)}"
        expected = {"SH600001": 500, "SH600002": 400, "SH600003": 400}
        for row in orders:
            sym = row["symbol"]
            assert row["target_qty"] == expected[sym], (
                f"{sym}: expected full-size qty {expected[sym]}, "
                f"got {row['target_qty']}"
            )
