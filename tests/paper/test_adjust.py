"""Tests for adjustfactor tracking (audit-only, no qty/cost mutation)."""

from ashare_lab.paper.adjust import check_and_apply_adjustfactor


def _make_pos(qty: int, avg_cost: float, factor: float) -> dict:
    """Build a minimal position dict for testing."""
    return {
        "qty": qty,
        "avg_cost": avg_cost,
        "market_value": qty * avg_cost,
        "buy_date": "2025-01-10",
        "holding_high": avg_cost * 1.1,
        "factor": factor,
    }


class TestSplitDetected:
    """Factor change (2:1 split): qty/cost unchanged, factor refreshed."""

    def test_split_no_qty_mutation(self):
        positions = {"SH600000": _make_pos(1000, 10.0, 0.14)}
        prev = {"SH600000": 0.14}
        curr = {"SH600000": 0.28}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        assert result["SH600000"]["qty"] == 1000
        assert result["SH600000"]["avg_cost"] == 10.0
        assert result["SH600000"]["factor"] == 0.28
        assert len(records) == 1
        assert records[0]["old_factor"] == 0.14
        assert records[0]["new_factor"] == 0.28

    def test_original_not_mutated(self):
        positions = {"SH600000": _make_pos(1000, 10.0, 0.14)}
        prev = {"SH600000": 0.14}
        curr = {"SH600000": 0.28}

        result, _ = check_and_apply_adjustfactor(positions, prev, curr)

        # Original must be untouched (deep copy)
        assert positions["SH600000"]["factor"] == 0.14
        assert result is not positions


class TestReverseSplit:
    """Reverse split (factor decreases): still audit-only."""

    def test_reverse_split_no_qty_mutation(self):
        positions = {"SZ300123": _make_pos(2000, 5.0, 0.28)}
        prev = {"SZ300123": 0.28}
        curr = {"SZ300123": 0.14}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        assert result["SZ300123"]["qty"] == 2000
        assert result["SZ300123"]["avg_cost"] == 5.0
        assert result["SZ300123"]["factor"] == 0.14
        assert len(records) == 1
        assert records[0]["old_factor"] == 0.28
        assert records[0]["new_factor"] == 0.14


class TestNoChange:
    """Same factor: no mutation, no audit record."""

    def test_no_factor_change(self):
        positions = {"SH600000": _make_pos(500, 20.0, 1.0)}
        prev = {"SH600000": 1.0}
        curr = {"SH600000": 1.0}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        assert result["SH600000"]["qty"] == 500
        assert result["SH600000"]["avg_cost"] == 20.0
        assert result["SH600000"]["factor"] == 1.0
        assert len(records) == 0


class TestNoneFactorGuard:
    """None factor (delisted/missing): skipped entirely."""

    def test_new_factor_none(self):
        positions = {"SH600999": _make_pos(300, 15.0, 0.5)}
        prev = {"SH600999": 0.5}
        curr = {"SH600999": None}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        assert result["SH600999"]["qty"] == 300
        assert result["SH600999"]["factor"] == 0.5
        assert len(records) == 0

    def test_old_factor_none(self):
        positions = {"SH601000": _make_pos(100, 30.0, 1.0)}
        prev = {}  # new listing, no previous factor
        curr = {"SH601000": 1.0}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        assert result["SH601000"]["factor"] == 1.0
        assert len(records) == 0


class TestMultiplePositions:
    """Only the changed symbol gets a refreshed factor."""

    def test_selective_refresh(self):
        positions = {
            "SH600000": _make_pos(1000, 10.0, 0.14),
            "SZ000001": _make_pos(500, 20.0, 1.0),
            "SH688001": _make_pos(200, 50.0, 0.5),
        }
        prev = {"SH600000": 0.14, "SZ000001": 1.0, "SH688001": 0.5}
        curr = {"SH600000": 0.28, "SZ000001": 1.0, "SH688001": 0.5}

        result, records = check_and_apply_adjustfactor(positions, prev, curr)

        # Only SH600000 changed
        assert result["SH600000"]["factor"] == 0.28
        assert result["SZ000001"]["factor"] == 1.0
        assert result["SH688001"]["factor"] == 0.5

        # Only SH600000 has an audit record
        assert len(records) == 1
        assert records[0]["symbol"] == "SH600000"

        # Other positions untouched in every field
        assert result["SZ000001"]["qty"] == 500
        assert result["SZ000001"]["avg_cost"] == 20.0
        assert result["SH688001"]["qty"] == 200
        assert result["SH688001"]["avg_cost"] == 50.0
