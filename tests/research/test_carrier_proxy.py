"""The carrier gate has to be able to say no.

Fixture with a known answer: every industry earns a fixed daily return, so the
trailing and forward windows are the same numbers and rank IC must be exactly
one. A panel with no relationship must sit near zero, and the floor must reject
a variant that is statistically clear but economically tiny.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_lab.research.carrier_proxy import (
    IC_FLOOR,
    T_FLOOR,
    ic_series,
    passing_variants,
    quintile_spread,
    summarise,
)


def _panel(n_dates: int = 60, n_industries: int = 40, daily=None) -> pd.DataFrame:
    dates = pd.period_range("2024-01-01", periods=n_dates, freq="D").astype(str)
    if daily is None:
        daily = np.linspace(-0.01, 0.01, n_industries)
    frame = pd.DataFrame(
        np.tile(np.asarray(daily, dtype=float), (n_dates, 1)),
        index=dates,
        columns=[f"ind{i:02d}" for i in range(n_industries)],
    )
    return frame


def test_constant_returns_give_a_perfect_rank_ic():
    wide = _panel()
    for horizon in (1, 5):
        ics = ic_series(wide, horizon)
        assert not ics.empty
        assert ics.mean() == pytest.approx(1.0), "the same ranking both sides must be +1"


def test_noise_does_not_clear_the_floor():
    wide = _noise_panel(n_dates=80)
    table = summarise(wide, horizons=(1,))
    assert abs(table.loc[0, "mean_ic"]) < 0.2


def test_the_spread_follows_the_signal():
    wide = _panel()
    spread = quintile_spread(wide, 5)
    assert (spread > 0).all(), "the best industries must also have the best forward returns"


def test_a_tiny_but_certain_ic_is_rejected():
    table = pd.DataFrame(
        {
            "horizon": [1, 5],
            "mean_ic": [IC_FLOOR / 2, 0.05],
            "t": [50.0, T_FLOOR / 2],
        }
    )
    assert passing_variants(table).empty, "neither row clears both floors"


def test_a_variant_clearing_both_floors_passes():
    table = pd.DataFrame({"horizon": [1], "mean_ic": [IC_FLOOR * 1.2], "t": [T_FLOOR * 1.5]})
    passed = passing_variants(table)
    assert len(passed) == 1
    assert int(passed.iloc[0]["horizon"]) == 1


def _noise_panel(n_dates: int = 90, n_industries: int = 40, seed: int = 20261110) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n_dates, n_industries)),
        index=pd.period_range("2024-01-01", periods=n_dates, freq="D").astype(str),
        columns=[f"ind{i:02d}" for i in range(n_industries)],
    )


def test_t_accounts_for_overlapping_windows():
    wide = _noise_panel()
    table = summarise(wide, horizons=(5,))
    ics = ic_series(wide, 5)
    assert ics.std(ddof=1) > 0, "this fixture must produce a varying IC series"
    expected = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics) / 5))
    assert table.loc[0, "t"] == pytest.approx(expected), "n_eff must be n/h, not n"
