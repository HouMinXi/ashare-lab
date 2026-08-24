"""Track S dynamic-slippage offline replay (R6).

Replays Track A's actual fill sequence under dynamic slippage models,
recomputing only fill_price / fees / NAV.  Read-only on paper.db.
See .planning/r6_slippage_dual_track_20260815.md (FROZEN 2026-08-15).

Model registry:
  S0: fixed 0.001 -- replay validator, must reproduce Track A NAV within 1e-6
  S2: Amihud-style linear impact (ILLIQ * q * P * 1e4)
  S3: liquidity-banded fixed slippage (quintile bands)
  S1: Almgren-Chriss square-root impact (demoted to sensitivity)

Usage:
    python -m ashare_lab.research.track_s_replay \\
        --paper-db /path/to/paper.db \\
        --model S2 \\
        --calibration /path/to/calibration_illiq.json \\
        --shadow-dir /path/to/shadow_slippage
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ashare_lab.paper.fees import calculate_fees
from ashare_lab.paper.ledger import init_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PAPER_DB = DEFAULT_PROJECT_ROOT / "paper.db"
DEFAULT_SHADOW_DIR = DEFAULT_PROJECT_ROOT / "shadow_slippage"
DEFAULT_CALIBRATION = DEFAULT_SHADOW_DIR / "calibration_illiq.json"
DEFAULT_ARTIFACT_DIR = DEFAULT_PROJECT_ROOT / "experiments" / "shadow_slippage"

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackTrade:
    """A single fill from Track A, augmented with recomputed values."""

    order_id: int
    trade_date: str
    symbol: str
    side: str
    fill_qty: int
    # Track A original values
    orig_fill_price: float
    orig_commission: float
    orig_stamp: float
    orig_transfer_fee: float
    # Replay-computed values (filled by model)
    replay_fill_price: float = 0.0
    replay_commission: float = 0.0
    replay_stamp: float = 0.0
    replay_transfer_fee: float = 0.0
    # Market data (fetched from qlib)
    close: float = 0.0
    volume: float = 0.0
    change: float = 0.0
    factor: float = 1.0


@dataclass
class DayNav:
    """NAV snapshot for one trading day."""

    trade_date: str
    cash: float
    market_value: float
    total_nav: float
    pre_trade_nav: float
    post_trade_nav: float


@dataclass
class ReplayResult:
    """Complete replay result for one model."""

    model_name: str
    trades: list[TrackTrade] = field(default_factory=list)
    nav_series: list[DayNav] = field(default_factory=list)
    total_impact_cost: float = 0.0
    total_track_a_cost: float = 0.0
    dates_covered: list[str] = field(default_factory=list)

    @property
    def n_fills(self) -> int:
        return len(self.trades)

    @property
    def first_date(self) -> str | None:
        return self.dates_covered[0] if self.dates_covered else None

    @property
    def last_date(self) -> str | None:
        return self.dates_covered[-1] if self.dates_covered else None


# ---------------------------------------------------------------------------
# Slippage model base + implementations
# ---------------------------------------------------------------------------


class SlippageModel(abc.ABC):
    """Abstract base for a Track S slippage model."""

    name: str = ""

    @abc.abstractmethod
    def compute_fill_price(
        self, trade: TrackTrade, **kwargs: Any
    ) -> float:
        """Return fill_price given the trade and market data."""

    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description."""


class S0Fixed(SlippageModel):
    """S0: fixed 0.001 slippage -- Track A reproducer."""

    name = "S0"

    def __init__(self, slippage: float = 0.001) -> None:
        self._slippage = slippage

    def compute_fill_price(self, trade: TrackTrade, **kwargs: Any) -> float:
        if trade.side == "buy":
            return trade.close * (1.0 + self._slippage)
        return trade.close * (1.0 - self._slippage)

    def description(self) -> str:
        return f"Fixed slippage {self._slippage}"


def _validate_calibration_schema(calibration: dict | None, model_name: str) -> None:
    """Validate calibration schema for models requiring calibration data."""
    if not isinstance(calibration, dict):
        raise ValueError(
            f"Invalid calibration schema for model {model_name}: "
            f"expected dict, got {type(calibration).__name__}"
        )
    per_stock = calibration.get("per_stock")
    if not isinstance(per_stock, dict) or not per_stock:
        raise ValueError(
            f"Invalid calibration schema for model {model_name}: "
            "'per_stock' must be a non-empty dict"
        )


class S2Amihud(SlippageModel):
    """S2: Amihud-style linear impact.

    impact_bp = ILLIQ_frozen * q * P * 1e4
    fill_price = close * (1 +/- impact_bp / 1e4)
    """

    name = "S2"

    def __init__(self, calibration: dict) -> None:
        _validate_calibration_schema(calibration, "S2")
        self._per_stock: dict[str, dict] = calibration.get("per_stock", {})
        illiq_vals = [
            v.get("illiq", 0.0) for v in self._per_stock.values() if v.get("illiq", 0.0) > 0
        ]
        self._median_illiq = sorted(illiq_vals)[len(illiq_vals) // 2] if illiq_vals else 1e-8

    def compute_fill_price(self, trade: TrackTrade, **kwargs: Any) -> float:
        illiq_data = self._per_stock.get(trade.symbol, {})
        illiq = illiq_data.get("illiq", self._median_illiq)
        # impact_bp = ILLIQ * q * P * 1e4
        impact_bp = illiq * trade.fill_qty * trade.close * 1e4
        slip = impact_bp / 1e4
        if trade.side == "buy":
            return trade.close * (1.0 + slip)
        return trade.close * (1.0 - slip)

    def description(self) -> str:
        return "Amihud-style linear impact (ILLIQ * q * P * 1e4)"


class S3Banded(SlippageModel):
    """S3: liquidity-banded fixed slippage.

    Quintile bands by cross-sectional ILLIQ, each with a fixed rate.
    """

    name = "S3"

    def __init__(self, calibration: dict) -> None:
        _validate_calibration_schema(calibration, "S3")
        self._per_stock: dict[str, dict] = calibration.get("per_stock", {})
        self._rates: list[float] = calibration.get(
            "quintile_rates", [0.0005, 0.0008, 0.0010, 0.0015, 0.0025]
        )

    def compute_fill_price(self, trade: TrackTrade, **kwargs: Any) -> float:
        illiq_data = self._per_stock.get(trade.symbol, {})
        quintile = illiq_data.get("quintile", 2)  # default to middle band
        quintile = max(0, min(quintile, len(self._rates) - 1))
        rate = self._rates[quintile]

        if trade.side == "buy":
            return trade.close * (1.0 + rate)
        return trade.close * (1.0 - rate)

    def description(self) -> str:
        return f"Liquidity-banded fixed slippage rates={self._rates}"


class S1AlmgrenChriss(SlippageModel):
    """S1: Almgren-Chriss square-root temporary impact (sensitivity analysis only).

    impact_bp = k * sigma_d * sqrt(q / V_d) * 1e4
    """

    name = "S1"

    def __init__(self, k: float = 0.1, default_sigma: float = 0.02) -> None:
        self.k = k
        self.default_sigma = default_sigma

    def compute_fill_price(self, trade: TrackTrade, **kwargs: Any) -> float:
        import math

        vol = max(trade.volume, 1.0)
        part = min(trade.fill_qty / vol, 1.0)
        sigma = abs(trade.change) if trade.change != 0 else self.default_sigma
        impact_bp = self.k * sigma * math.sqrt(part) * 1e4
        slip = impact_bp / 1e4
        if trade.side == "buy":
            return trade.close * (1.0 + slip)
        return trade.close * (1.0 - slip)

    def description(self) -> str:
        return f"Almgren-Chriss square-root impact (k={self.k}) [UNRELIABLE / SENSITIVITY]"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[SlippageModel]] = {
    "S0": S0Fixed,
    "S2": S2Amihud,
    "S3": S3Banded,
    "S1": S1AlmgrenChriss,
}


def get_model(name: str, calibration: dict | None = None) -> SlippageModel:
    """Factory: return a model instance by name."""
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown model {name!r}; available: {list(_MODEL_REGISTRY)}"
        )
    if name == "S0":
        return cls()
    if name in ("S2", "S3"):
        _validate_calibration_schema(calibration, name)
        return cls(calibration)
    if name == "S1":
        return cls()
    return cls()


# ---------------------------------------------------------------------------
# Paper DB reader (read-only)
# ---------------------------------------------------------------------------


def _paper_conn(paper_db: str | Path) -> sqlite3.Connection:
    """Open paper.db in read-only mode."""
    db_path = Path(paper_db).resolve()
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_trades(paper_db: str | Path) -> list[TrackTrade]:
    """Read all trades from paper.db, ordered by trade_date, id."""
    conn = _paper_conn(paper_db)
    try:
        rows = conn.execute(
            """
            SELECT t.order_id, t.trade_date, t.symbol, t.side,
                   t.fill_qty, t.fill_price, t.commission,
                   t.stamp, t.transfer_fee
            FROM trades t
            ORDER BY t.trade_date, t.id
            """
        ).fetchall()

        trades = []
        for row in rows:
            trades.append(
                TrackTrade(
                    order_id=row[0],
                    trade_date=row[1],
                    symbol=row[2],
                    side=row[3],
                    fill_qty=row[4],
                    orig_fill_price=row[5],
                    orig_commission=row[6],
                    orig_stamp=row[7],
                    orig_transfer_fee=row[8],
                )
            )
        logger.info("Read %d trades from %s", len(trades), paper_db)
        return trades
    finally:
        conn.close()


def read_initial_cash(paper_db: str | Path) -> float:
    """Read initial cash from the first NAV entry."""
    conn = _paper_conn(paper_db)
    try:
        row = conn.execute(
            "SELECT cash FROM nav ORDER BY trade_date ASC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else 300000.0
    finally:
        conn.close()


def read_nav_series(paper_db: str | Path) -> list[dict[str, Any]]:
    """Read full NAV series from paper.db for cross-validation."""
    conn = _paper_conn(paper_db)
    try:
        rows = conn.execute(
            """
            SELECT trade_date, cash, market_value, total_nav,
                   pre_trade_nav, post_trade_nav
            FROM nav ORDER BY trade_date
            """
        ).fetchall()
        return [
            {
                "trade_date": r[0],
                "cash": r[1],
                "market_value": r[2],
                "total_nav": r[3],
                "pre_trade_nav": r[4] if r[4] is not None else 0.0,
                "post_trade_nav": r[5] if r[5] is not None else 0.0,
            }
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Qlib data fetcher
# ---------------------------------------------------------------------------


def fetch_qlib_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    provider_uri: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fetch daily prices/volume/returns from qlib for given symbols/date range.

    Returns {symbol: {date: {"close": float, "volume": float, "change": float, "factor": float}}}.
    """
    try:
        import qlib
        from qlib.data import D
    except ImportError as e:
        raise ImportError(f"Failed to import qlib for Track S replay: {e}") from e

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI

    uri = provider_uri or str(DEFAULT_PROVIDER_URI)
    try:
        qlib.init(provider_uri=uri)
    except Exception:
        pass

    raw = D.features(
        instruments=list(set(symbols)),
        fields=["$close", "$volume", "$change", "$factor"],
        start_time=start_date,
        end_time=end_date,
    )

    out: dict[str, dict[str, dict[str, float]]] = {}
    if raw is not None and not raw.empty:
        for idx, row in raw.iterrows():
            inst = idx[0] if isinstance(idx, tuple) else str(idx)
            dt = str(idx[1])[:10] if isinstance(idx, tuple) else ""
            close_raw = float(row.get("$close", 0.0))
            factor = float(row.get("$factor", 1.0))
            actual_close = close_raw / factor if factor > 0 else close_raw
            out.setdefault(str(inst), {})[dt] = {
                "close": actual_close,
                "volume": float(row.get("$volume", 0.0)),
                "change": float(row.get("$change", 0.0)),
                "factor": factor,
            }
    return out


def _load_calibration(calibration_path: str | Path) -> dict:
    """Load frozen calibration JSON."""
    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found at {path}. "
            "Run calibrate_illiq.py first."
        )
    cal = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cal, dict) or "per_stock" not in cal:
        raise ValueError(f"Invalid calibration artifact at {path}: missing 'per_stock'")
    return cal


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def run_model(
    trades: list[TrackTrade],
    nav_dates: list[str],
    prices: dict[str, dict[str, dict[str, float]]],
    model: SlippageModel,
    initial_cash: float,
) -> ReplayResult:
    """Run one model over all dates, reproducing trade fills and NAV series.

    Returns ReplayResult with recomputed trades and NAV series.
    """
    if trades and not prices:
        raise ValueError("Price data dictionary is empty while trades exist")

    result = ReplayResult(model_name=model.name)

    trades_by_date: dict[str, list[TrackTrade]] = {}
    for t in trades:
        trades_by_date.setdefault(t.trade_date, []).append(t)

    cash = initial_cash
    positions: dict[str, int] = {}  # symbol -> qty
    last_known_close: dict[str, float] = {}

    for dt in nav_dates:
        # Update last known close for currently held positions if present today
        for sym in positions:
            p = prices.get(sym, {}).get(dt, {}).get("close", 0.0)
            if p > 0.0:
                last_known_close[sym] = p

        # Pre-trade market value
        pre_market_value = 0.0
        for sym, qty in positions.items():
            p = prices.get(sym, {}).get(dt, {}).get("close", 0.0)
            if p <= 0.0:
                p = last_known_close.get(sym, 0.0)
            if p <= 0.0 and qty > 0:
                raise ValueError(
                    f"Missing valid close price for held symbol {sym} on date {dt}"
                )
            pre_market_value += qty * p
        pre_trade_nav = cash + pre_market_value

        day_trades = trades_by_date.get(dt, [])
        for t in day_trades:
            pdata = prices.get(t.symbol, {}).get(dt, {})
            close = pdata.get("close", 0.0)
            if close <= 0.0:
                close = last_known_close.get(t.symbol, 0.0)
            if close <= 0.0:
                raise ValueError(
                    f"Missing valid close price for trade on {t.symbol} on date {dt}"
                )
            last_known_close[t.symbol] = close
            t.close = close
            t.volume = pdata.get("volume", 0.0)
            t.change = pdata.get("change", 0.0)
            t.factor = pdata.get("factor", 1.0)

            # Compute new fill price
            t.replay_fill_price = model.compute_fill_price(t)
            notional = t.replay_fill_price * t.fill_qty
            fees = calculate_fees(notional, t.side)
            t.replay_commission = fees.commission
            t.replay_stamp = fees.stamp
            t.replay_transfer_fee = fees.transfer

            # Original vs replay costs
            orig_notional = t.orig_fill_price * t.fill_qty
            orig_total_fees = t.orig_commission + t.orig_stamp + t.orig_transfer_fee
            replay_total_fees = fees.total

            result.total_impact_cost += (
                notional + replay_total_fees - orig_notional - orig_total_fees
            )
            result.total_track_a_cost += orig_notional + orig_total_fees

            if t.side == "buy":
                cash -= notional + fees.total
                positions[t.symbol] = positions.get(t.symbol, 0) + t.fill_qty
            else:
                current_qty = positions.get(t.symbol, 0)
                if t.fill_qty > current_qty:
                    raise ValueError(
                        f"Oversell detected: symbol {t.symbol} held {current_qty}, "
                        f"attempted to sell {t.fill_qty}"
                    )
                cash += notional - fees.total
                positions[t.symbol] = current_qty - t.fill_qty
                if positions[t.symbol] <= 0:
                    positions.pop(t.symbol, None)

            result.trades.append(t)

        # Post-trade market value
        post_market_value = 0.0
        for sym, qty in positions.items():
            p = prices.get(sym, {}).get(dt, {}).get("close", 0.0)
            if p <= 0.0:
                p = last_known_close.get(sym, 0.0)
            if p <= 0.0 and qty > 0:
                raise ValueError(
                    f"Missing valid close price for held symbol {sym} on date {dt}"
                )
            post_market_value += qty * p
        total_nav = cash + post_market_value

        result.nav_series.append(
            DayNav(
                trade_date=dt,
                cash=cash,
                market_value=post_market_value,
                total_nav=total_nav,
                pre_trade_nav=pre_trade_nav,
                post_trade_nav=total_nav,
            )
        )
        result.dates_covered.append(dt)

    logger.info(
        "Model %s: %d fills over %d dates, total_impact_cost_diff=%.2f CNY",
        model.name, result.n_fills, len(result.dates_covered),
        result.total_impact_cost,
    )
    return result


# ---------------------------------------------------------------------------
# Shadow DB writer
# ---------------------------------------------------------------------------


def init_shadow_db(db_path: str | Path) -> sqlite3.Connection:
    """Create/initialize shadow DB with same schema as paper.db."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    init_schema(conn)
    return conn


def write_to_shadow_db(
    db_path: str | Path,
    model_name: str,
    trades: list[TrackTrade],
    nav_series: list[DayNav],
    initial_cash: float,
) -> None:
    """Write replay results to shadow DB."""
    conn = init_shadow_db(db_path)
    try:
        with conn:
            # Insert trades
            conn.executemany(
                """
                INSERT OR REPLACE INTO trades (order_id, trade_date, symbol, side,
                                               fill_price, fill_qty, commission,
                                               stamp, transfer_fee)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.order_id,
                        t.trade_date,
                        t.symbol,
                        t.side,
                        t.replay_fill_price,
                        t.fill_qty,
                        t.replay_commission,
                        t.replay_stamp,
                        t.replay_transfer_fee,
                    )
                    for t in trades
                ],
            )

            # Insert NAV
            conn.executemany(
                """
                INSERT OR REPLACE INTO nav (trade_date, cash, market_value, total_nav,
                                            pre_trade_nav, post_trade_nav)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        nav.trade_date,
                        nav.cash,
                        nav.market_value,
                        nav.total_nav,
                        nav.pre_trade_nav,
                        nav.post_trade_nav,
                    )
                    for nav in nav_series
                ],
            )

            # Insert run record
            conn.execute(
                "INSERT INTO runs (trade_date, status, started_at) VALUES (?, ?, ?)",
                (
                    nav_series[-1].trade_date if nav_series else "unknown",
                    "settled",
                    datetime.now().isoformat(),
                ),
            )

        logger.info(
            "Wrote %d trades, %d NAV rows to %s",
            len(trades), len(nav_series), db_path,
        )
    finally:
        conn.close()


def get_last_shadow_date(shadow_db: str | Path) -> str | None:
    """Return the last trade_date in the shadow DB, or None if empty."""
    path = Path(shadow_db)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM nav"
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------


def write_artifact(
    artifact_dir: str | Path,
    model_name: str,
    trade_date: str,
    day_trades: list[TrackTrade],
    nav: DayNav | None = None,
    gate1_diff: float | None = None,
) -> Path:
    """Write per-day artifact JSON."""
    path = Path(artifact_dir) / f"{trade_date}_{model_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    trades_data = []
    for t in day_trades:
        trades_data.append(
            {
                "order_id": t.order_id,
                "symbol": t.symbol,
                "side": t.side,
                "fill_qty": t.fill_qty,
                "orig_fill_price": t.orig_fill_price,
                "replay_fill_price": t.replay_fill_price,
                "orig_total_fees": (
                    t.orig_commission + t.orig_stamp + t.orig_transfer_fee
                ),
                "replay_total_fees": (
                    t.replay_commission + t.replay_stamp + t.replay_transfer_fee
                ),
                "close": t.close,
                "volume": t.volume,
            }
        )

    artifact: dict[str, Any] = {
        "model": model_name,
        "trade_date": trade_date,
        "n_fills": len(day_trades),
        "trades": trades_data,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if nav is not None:
        artifact["nav"] = {
            "cash": nav.cash,
            "market_value": nav.market_value,
            "total_nav": nav.total_nav,
            "pre_trade_nav": nav.pre_trade_nav,
            "post_trade_nav": nav.post_trade_nav,
        }

    if gate1_diff is not None:
        artifact["gate_1_diff"] = gate1_diff

    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True))
    return path


# ---------------------------------------------------------------------------
# Gate 1: S0 vs Track A NAV validation
# ---------------------------------------------------------------------------


def gate_1_check(
    s0_result: ReplayResult,
    track_a_nav: list[dict[str, Any]],
) -> float:
    """Gate 1: compare S0 replay NAV against Track A.

    Returns max relative diff.  Exits non-zero if > 1e-6.
    """
    if not s0_result.nav_series or not track_a_nav:
        logger.error("Gate 1: empty NAV series, cannot compare")
        return float("inf")

    s0_by_date = {n.trade_date: n.total_nav for n in s0_result.nav_series}
    a_by_date = {n["trade_date"]: n["total_nav"] for n in track_a_nav}

    max_diff = 0.0
    diffs = []
    for dt, s0_nav in s0_by_date.items():
        a_nav = a_by_date.get(dt)
        if a_nav is None or a_nav == 0.0:
            continue
        rel_diff = abs(s0_nav - a_nav) / abs(a_nav)
        diffs.append((dt, rel_diff, s0_nav, a_nav))
        if rel_diff > max_diff:
            max_diff = rel_diff

    logger.info(
        "Gate 1: max relative diff = %.2e (%d dates compared)",
        max_diff, len(diffs),
    )

    if max_diff > 1e-6:
        worst = max(diffs, key=lambda x: x[1])
        logger.error(
            "Gate 1 FAILED: %s S0=%.6f TrackA=%.6f diff=%.2e",
            worst[0], worst[2], worst[3], worst[1],
        )
        print(
            f"GATE 1 FAILED: max relative diff = {max_diff:.2e} > 1e-6. "
            f"Replay is broken. STOP."
        )
        sys.exit(1)

    return max_diff


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_replay(
    paper_db: str | Path = DEFAULT_PAPER_DB,
    model_name: str = "S0",
    calibration_path: str | Path | None = None,
    shadow_dir: str | Path = DEFAULT_SHADOW_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    provider_uri: str | None = None,
    incremental: bool = True,
    skip_gate1: bool = False,
) -> ReplayResult:
    """Run the full replay pipeline for one model.

    Returns ReplayResult.
    """
    paper = Path(paper_db)
    if not paper.exists():
        raise FileNotFoundError(f"Paper DB not found: {paper}")

    # Load calibration if needed
    calibration = None
    if model_name in ("S2", "S3"):
        cal_path = calibration_path or DEFAULT_CALIBRATION
        calibration = _load_calibration(cal_path)

    # Create model
    model = get_model(model_name, calibration)

    # Read trades and NAV series from paper.db
    all_trades = read_trades(paper)
    track_a_nav = read_nav_series(paper)

    if not track_a_nav:
        logger.warning("No NAV records found in paper.db")
        return ReplayResult(model_name=model_name)

    all_nav_dates = [n["trade_date"] for n in track_a_nav]
    start_date = all_nav_dates[0]
    end_date = all_nav_dates[-1]

    # Collect symbols needed for price lookup
    symbols = list({t.symbol for t in all_trades})

    # Fetch qlib data for all symbols across the full date range
    logger.info(
        "Fetching qlib prices for %d symbols (%s to %s)...",
        len(symbols), start_date, end_date,
    )
    prices = fetch_qlib_prices(symbols, start_date, end_date, provider_uri=provider_uri)
    if all_trades and not prices:
        raise ValueError(
            f"qlib returned no prices for {len(symbols)} symbols ({start_date} to {end_date})"
        )

    initial_cash = read_initial_cash(paper)

    # Check incremental state
    shadow_db = Path(shadow_dir) / f"paper_s_{model_name}.db"
    last_shadow_date = get_last_shadow_date(shadow_db) if incremental else None

    # Run replay across all dates
    result = run_model(all_trades, all_nav_dates, prices, model, initial_cash)

    # Gate 1 check for S0
    gate1_diff = None
    if model_name == "S0" and not skip_gate1:
        gate1_diff = gate_1_check(result, track_a_nav)

    # Write shadow DB
    write_to_shadow_db(
        shadow_db, model_name, result.trades, result.nav_series, initial_cash
    )

    # Write per-day artifacts (for all dates or new dates)
    for nav in result.nav_series:
        if last_shadow_date is None or nav.trade_date > last_shadow_date:
            day_trades = [t for t in result.trades if t.trade_date == nav.trade_date]
            write_artifact(
                artifact_dir, model_name, nav.trade_date, day_trades, nav,
                gate1_diff=gate1_diff if model_name == "S0" else None,
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track S dynamic-slippage offline replay"
    )
    parser.add_argument(
        "--paper-db", default=str(DEFAULT_PAPER_DB),
        help="Path to paper.db (default: %(default)s)",
    )
    parser.add_argument(
        "--model", "-m", default="S0",
        choices=list(_MODEL_REGISTRY),
        help="Slippage model (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration", "-c", default=None,
        help="Path to calibration JSON (required for S2/S3)",
    )
    parser.add_argument(
        "--shadow-dir", default=str(DEFAULT_SHADOW_DIR),
        help="Shadow DB directory (default: %(default)s)",
    )
    parser.add_argument(
        "--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR),
        help="Artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--provider-uri", default=None,
        help="Custom qlib provider URI",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full replay (ignore incremental)",
    )
    parser.add_argument(
        "--skip-gate1", action="store_true",
        help="Skip Gate 1 check (S0 only; for development)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_replay(
        paper_db=args.paper_db,
        model_name=args.model,
        calibration_path=args.calibration,
        shadow_dir=args.shadow_dir,
        artifact_dir=args.artifact_dir,
        provider_uri=args.provider_uri,
        incremental=not args.full,
        skip_gate1=args.skip_gate1,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
