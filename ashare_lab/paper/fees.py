"""A-share fee calculation for the paper trading engine.

Commission 0.025% both sides (min 5 CNY per order), stamp duty 0.05%
sell-only, transfer fee 0.001% both sides.  Rates as of Aug 2023.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeResult:
    """Immutable fee breakdown for a single order."""

    commission: float
    stamp: float
    transfer: float
    total: float


def calculate_fees(
    notional: float,
    side: str,
    commission_rate: float = 0.00025,
    min_commission: float = 5.0,
    stamp_rate: float = 0.0005,
    transfer_rate: float = 0.00001,
) -> FeeResult:
    """Compute fees for one order.

    Parameters
    ----------
    notional : float
        fill_price * fill_qty (absolute value).
    side : str
        ``"buy"`` or ``"sell"``.
    commission_rate : float
        Per-side commission rate (default 0.025 %).
    min_commission : float
        Floor per order in CNY (default 5.0).
    stamp_rate : float
        Stamp duty rate, sell-only (default 0.05 %).
    transfer_rate : float
        Transfer fee rate, both sides (default 0.001 %).
    """
    commission = max(notional * commission_rate, min_commission)
    stamp = notional * stamp_rate if side == "sell" else 0.0
    transfer = notional * transfer_rate
    return FeeResult(
        commission=commission,
        stamp=stamp,
        transfer=transfer,
        total=commission + stamp + transfer,
    )
