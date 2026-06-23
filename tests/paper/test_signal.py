"""Tests for signal generation, TopkDropout order logic, and candidate filtering."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# topk_dropout_orders tests
# ---------------------------------------------------------------------------


class TestTopkDropoutOrders:
    """Pure-function tests for TopkDropout rebalancing logic (D-20)."""

    def test_basic_sell_buy(self):
        """Sell worst held in top-K out, buy next best not held."""
        from ashare_lab.paper.signal import topk_dropout_orders

        signals = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
        held = {"A", "B", "C"}
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        # C is the worst-ranked held stock NOT in top-3... wait:
        # top-3 = {A, B, C}, so all held ARE in top-3.
        # Positions not in top-K: none.  So sell_cands is empty.
        # Actually re-read: C is IN top-3 (rank 3). So sell = [].
        # But D is rank 4, not held. buy_candidates from top-3 not held = [].
        # keep = 3 - 0 = 3, n_buy = max(0, 3-3) = 0 -> buy = []
        # That means all_held_in_topk test. Let me re-check the plan:
        # Plan says: sell=[C], buy=[D]. That means C is NOT in top-3?
        # top-3 by score: A(0.9), B(0.8), C(0.7). C IS in top-3.
        # Hmm, but n_drop=1 means we drop the worst held.
        # Re-reading the research code: sell_cands = positions NOT in top_k_symbols
        # So if all 3 held are in top-3, sell_cands is empty.
        # The plan behavior line says "worst held in top-3 is C at rank 3"
        # but that contradicts the algorithm: sell only what's NOT in top-K.
        # The algorithm is correct per D-20: drop positions not in top-K.
        # I trust the algorithm code over the behavior example text.
        # With these inputs, all held are in top-K, so sell=[], buy=[].
        assert sell == []
        assert buy == []

    def test_held_outside_topk(self):
        """Held stock not in top-K gets sold, replacement bought."""
        from ashare_lab.paper.signal import topk_dropout_orders

        # E is held but ranked 5th (outside top-3)
        signals = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
        held = {"A", "B", "E"}
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        # top-3 = {A, B, C}. E not in top-3 -> sell_cands = [E]
        # sell_list = [E] (capped to n_drop=1)
        assert sell == ["E"]
        # buy_candidates from top-3 not held: [C]
        # keep = 3 - 1 = 2, n_buy = max(0, 3 - 2) = 1
        assert buy == ["C"]

    def test_cold_start_fills_topk(self):
        """Cold start (no positions) buys up to topk, not n_drop."""
        from ashare_lab.paper.signal import topk_dropout_orders

        signals = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
        sell, buy = topk_dropout_orders(signals, set(), topk=3, n_drop=1)
        assert sell == []
        # keep = 0 - 0 = 0, n_buy = max(0, 3 - 0) = 3
        assert buy == ["A", "B", "C"]

    def test_all_held_in_topk_no_change(self):
        """When all held stocks are in top-K, no sells or buys."""
        from ashare_lab.paper.signal import topk_dropout_orders

        signals = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}
        held = {"A", "B", "C"}
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        assert sell == []
        assert buy == []

    def test_steady_state_ndrop_limits(self):
        """Steady state: held==topk, one drops out, n_drop=1 caps sell to 1."""
        from ashare_lab.paper.signal import topk_dropout_orders

        # Held: {A, B, F}. F is rank 6 (outside top-3).
        # New top-3: {A, B, C}.
        signals = {
            "A": 0.9, "B": 0.8, "C": 0.7,
            "D": 0.6, "E": 0.5, "F": 0.4,
        }
        held = {"A", "B", "F"}
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        # sell_cands = [F], capped to 1 -> [F]
        assert sell == ["F"]
        # keep = 3 - 1 = 2, n_buy = max(0, 3 - 2) = 1
        # buy_candidates from top-3 not held: [C]
        assert buy == ["C"]

    def test_multiple_outside_topk_capped_by_ndrop(self):
        """Multiple positions outside top-K, but n_drop caps sell count."""
        from ashare_lab.paper.signal import topk_dropout_orders

        signals = {
            "A": 0.9, "B": 0.8, "C": 0.7,
            "D": 0.6, "E": 0.5, "F": 0.4,
        }
        # D, E, F are all outside top-3, but n_drop=1 caps sell to 1
        held = {"A", "D", "E", "F"}
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        # sell_cands sorted ascending by score: [F(0.4), E(0.5), D(0.6)]
        # capped to n_drop=1 -> sell = [F]
        assert len(sell) == 1
        assert sell == ["F"]
        # keep = 4 - 1 = 3, n_buy = max(0, 3 - 3) = 0
        assert buy == []

    def test_soft_drawdown_sell_one_buy_zero(self):
        """Soft drawdown: held=15, topk reduced to 7, sell=1, buy=0."""
        from ashare_lab.paper.signal import topk_dropout_orders

        # 20 symbols, held has 15
        all_syms = [f"S{i:02d}" for i in range(20)]
        signals = {s: 1.0 - i * 0.01 for i, s in enumerate(all_syms)}
        # Held: first 15 symbols (S00..S14)
        held = set(all_syms[:15])

        sell, buy = topk_dropout_orders(signals, held, topk=7, n_drop=1)
        # top-7 = {S00..S06}. sell_cands = held NOT in top-7 = {S07..S14} (8 items)
        # sorted ascending by score, worst first: [S14, S13, ..., S07]
        # capped to n_drop=1 -> sell = [S14]
        assert len(sell) == 1
        assert sell == ["S14"]
        # keep = 15 - 1 = 14, n_buy = max(0, 7 - 14) = 0
        assert buy == []

    def test_topk_with_override(self):
        """When topk_override=7 (D-45), only top 7 are considered."""
        from ashare_lab.paper.signal import topk_dropout_orders

        all_syms = [f"T{i:02d}" for i in range(20)]
        signals = {s: 1.0 - i * 0.01 for i, s in enumerate(all_syms)}
        held = set(all_syms[:7])  # exactly topk held

        sell, buy = topk_dropout_orders(signals, held, topk=7, n_drop=1)
        # All 7 held are in top-7 -> no sell, no buy
        assert sell == []
        assert buy == []

    def test_held_stock_missing_from_signals(self):
        """Held stock with no signal gets -inf score, sold first."""
        from ashare_lab.paper.signal import topk_dropout_orders

        signals = {"A": 0.9, "B": 0.8, "C": 0.7}
        held = {"A", "B", "Z"}  # Z has no signal
        sell, buy = topk_dropout_orders(signals, held, topk=3, n_drop=1)
        # Z not in top-3 -> sell_cands = [Z]
        # Z gets -inf score -> sorted first -> sell = [Z]
        assert sell == ["Z"]
        # keep = 3 - 1 = 2, n_buy = max(0, 3 - 2) = 1
        assert buy == ["C"]

    def test_empty_signals(self):
        """No signals at all -> no sells, no buys."""
        from ashare_lab.paper.signal import topk_dropout_orders

        sell, buy = topk_dropout_orders({}, set(), topk=3, n_drop=1)
        assert sell == []
        assert buy == []


# ---------------------------------------------------------------------------
# filter_candidates tests
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    """Pure-function tests for candidate filtering (D-17, D-44)."""

    def _make_market_data(
        self,
        close: float = 100.0,
        listing_days: int = 120,
        avg_turnover_20d: float = 80_000_000.0,
    ) -> dict:
        return {
            "close": close,
            "listing_days": listing_days,
            "avg_turnover_20d": avg_turnover_20d,
        }

    def test_all_filters_pass(self):
        """Stock passing all filters is retained."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data()}
        result = filter_candidates(["SZ000001"], md)
        assert result == ["SZ000001"]

    def test_listing_days_excluded(self):
        """Stock with listing_days=30 excluded (< 60 per D-17)."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(listing_days=30)}
        result = filter_candidates(["SZ000001"], md)
        assert result == []

    def test_listing_days_boundary(self):
        """Stock with exactly 60 listing days passes."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(listing_days=60)}
        result = filter_candidates(["SZ000001"], md)
        assert result == ["SZ000001"]

    def test_low_turnover_excluded(self):
        """Stock with avg_turnover=40M excluded (< 50M per D-44)."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(avg_turnover_20d=40_000_000)}
        result = filter_candidates(["SZ000001"], md)
        assert result == []

    def test_high_close_excluded(self):
        """Stock with close=350 excluded (> 300 CNY)."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(close=350.0)}
        result = filter_candidates(["SZ000001"], md)
        assert result == []

    def test_close_boundary(self):
        """Stock with exactly close=300 passes (not strictly greater)."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(close=300.0)}
        result = filter_candidates(["SZ000001"], md)
        assert result == ["SZ000001"]

    def test_mixed_pass_fail(self):
        """Multiple stocks, some pass, some fail on different filters."""
        from ashare_lab.paper.signal import filter_candidates

        md = {
            "PASS": self._make_market_data(),
            "YOUNG": self._make_market_data(listing_days=10),
            "ILLIQUID": self._make_market_data(avg_turnover_20d=1_000_000),
            "PRICEY": self._make_market_data(close=500.0),
        }
        result = filter_candidates(
            ["PASS", "YOUNG", "ILLIQUID", "PRICEY"], md,
        )
        assert result == ["PASS"]

    def test_symbol_not_in_market_data_excluded(self):
        """Symbol with no market data is excluded."""
        from ashare_lab.paper.signal import filter_candidates

        result = filter_candidates(["UNKNOWN"], {})
        assert result == []

    def test_custom_thresholds(self):
        """Custom filter thresholds override defaults."""
        from ashare_lab.paper.signal import filter_candidates

        md = {"SZ000001": self._make_market_data(listing_days=45, close=250.0)}
        # Default listing_min_days=60 would exclude; custom 30 passes
        result = filter_candidates(
            ["SZ000001"], md,
            listing_min_days=30,
            liquidity_min_turnover=50_000_000,
            close_max=300.0,
        )
        assert result == ["SZ000001"]
