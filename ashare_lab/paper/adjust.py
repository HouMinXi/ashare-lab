"""Adjustfactor tracking for stock splits and bonus issues.

Detects corporate actions by comparing consecutive days' qlib $factor
values.  AUDIT-ONLY: qty and avg_cost are never mutated because NAV
is valued at the continuous qfq $close series -- mutating qty would
distort NAV (e.g. a 2:1 split doubles $factor while $close stays
continuous; applying qty * old/new would halve NAV).

The only position mutation is refreshing the stored factor field so
the next day's comparison detects correctly.
"""

from __future__ import annotations

import copy


def check_and_apply_adjustfactor(
    positions: dict[str, dict],
    previous_factors: dict[str, float],
    current_factors: dict[str, float],
) -> tuple[dict[str, dict], list[dict]]:
    """Detect factor changes and refresh stored factor (audit-only).

    Pure function -- deep-copies *positions* before any mutation.

    Parameters
    ----------
    positions : dict
        symbol -> {qty, avg_cost, market_value, buy_date, holding_high, factor}
    previous_factors : dict
        symbol -> factor from yesterday's position snapshot
    current_factors : dict
        symbol -> factor from today's qlib $factor field

    Returns
    -------
    (new_positions, adjustment_records)
        new_positions: deep copy with factor field refreshed where changed
        adjustment_records: list of audit dicts for the JSONL trail
    """
    new_positions = copy.deepcopy(positions)
    records: list[dict] = []

    for symbol in new_positions:
        old_factor = previous_factors.get(symbol)
        new_factor = current_factors.get(symbol)

        # Skip if either side is missing (new listing or delisted today)
        if old_factor is None or new_factor is None:
            continue

        if old_factor == new_factor:
            continue

        # Corporate action detected -- audit only, no qty/cost mutation.
        # Refresh stored factor for next-day change detection.
        new_positions[symbol]["factor"] = new_factor

        records.append({
            "symbol": symbol,
            "old_factor": old_factor,
            "new_factor": new_factor,
            "qty": new_positions[symbol]["qty"],
            "avg_cost": new_positions[symbol]["avg_cost"],
        })

    return new_positions, records
