"""Tests for the industry return panel."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_lab.research.industry_panel import (
    MIN_MEMBERS,
    build_panel,
    load_industry_map,
    summary,
    to_wide,
)


def _store(tmp_path: Path, dates: list[str], closes: dict[str, list[float]]) -> Path:
    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")
    for symbol, values in closes.items():
        d = root / "features" / symbol
        d.mkdir(parents=True)
        body = struct.pack("<f", 0.0) + b"".join(struct.pack("<f", float(v)) for v in values)
        (d / "close.day.bin").write_bytes(body)
        (d / "factor.day.bin").write_bytes(body)
    return root


def test_the_map_speaks_the_stores_spelling(tmp_path):
    csv = tmp_path / "industry.csv"
    csv.write_text(
        "date,code,industry\n2026-07-10,sh.600000,J66银行\n2026-07-10,SZ.000001,J66银行\n",
        encoding="utf-8",
    )
    assert load_industry_map(csv) == {"sh600000": "J66银行", "sz000001": "J66银行"}


def test_a_missing_column_is_refused(tmp_path):
    csv = tmp_path / "industry.csv"
    csv.write_text("date,code\n2026-07-10,sh.600000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_industry_map(csv)


def test_mean_median_and_member_count(tmp_path):
    dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
    closes = {f"sh60000{i}": [10.0, 10.0 + 0.1 * i, 10.0 + 0.15 * i] for i in range(6)}
    root = _store(tmp_path, dates, closes)
    mapping = {sym: "A" for sym in closes}

    panel = build_panel(root, mapping)

    first = panel[(panel["date"] == "2026-11-03") & (panel["industry"] == "A")].iloc[0]
    returns = [float(pd.Series(v).pct_change().iloc[1]) for v in closes.values()]
    assert first["n_members"] == 6
    assert first["mean_ret"] == pytest.approx(np.mean(returns))
    assert first["median_ret"] == pytest.approx(np.median(returns))
    assert 0 < first["up_share"] <= 1


def test_an_industry_below_the_floor_is_dropped(tmp_path):
    dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
    closes = {f"sh60000{i}": [10.0, 10.5, 10.6] for i in range(MIN_MEMBERS - 1)}
    root = _store(tmp_path, dates, closes)

    panel = build_panel(root, {sym: "A" for sym in closes})

    assert panel.empty, "four names wearing a label are not an industry mean"


def test_an_impossible_move_is_excluded_and_counted(tmp_path):
    dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
    closes = {f"sh60000{i}": [10.0, 10.2, 10.3] for i in range(MIN_MEMBERS)}
    closes["sh600009"] = [10.0, 15.0, 15.0]  # +50%: a broken row, not a return
    root = _store(tmp_path, dates, closes)

    panel = build_panel(root, {sym: "A" for sym in closes})

    day = panel[panel["date"] == "2026-11-03"].iloc[0]
    assert day["n_suspect"] == 1
    assert day["n_members"] == MIN_MEMBERS
    assert day["mean_ret"] < 0.10, "the broken row must not drag the mean"


def test_a_symbol_with_one_close_is_skipped(tmp_path):
    dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
    closes = {f"sh60000{i}": [10.0, 10.2, 10.3] for i in range(MIN_MEMBERS)}
    closes["sh600009"] = [10.0, float("nan"), float("nan")]
    root = _store(tmp_path, dates, closes)

    panel = build_panel(root, {sym: "A" for sym in closes})

    assert not panel.empty
    assert panel["n_members"].max() == MIN_MEMBERS


def test_the_summary_reports_shape_and_dispersion(tmp_path):
    dates = [f"2026-11-{d:02d}" for d in range(2, 12)]
    closes = {}
    for i in range(MIN_MEMBERS):
        closes[f"sh60000{i}"] = [10.0 + (i * 0.1) * (d % 3) for d in range(len(dates))]
    root = _store(tmp_path, dates, closes)
    panel = build_panel(root, {sym: "A" for sym in closes})

    stats = summary(panel, horizons=(1, 5))

    assert stats["sessions"] == len(dates) - 1
    assert stats["industries"] == 1
    assert stats["dispersion"] >= 0
    assert set(stats) > {"momentum_autocorr_1d", "momentum_autocorr_5d"}


def test_wide_shape_is_date_by_industry(tmp_path):
    dates = ["2026-11-02", "2026-11-03"]
    closes = {f"sh60000{i}": [10.0, 10.5] for i in range(MIN_MEMBERS)}
    root = _store(tmp_path, dates, closes)
    panel = build_panel(root, {sym: "A" for sym in closes})

    wide = to_wide(panel)

    assert list(wide.index) == ["2026-11-03"]
    assert list(wide.columns) == ["A"]


def test_a_calendar_date_snaps_to_the_next_session(tmp_path):
    dates = ["2026-11-02", "2026-11-03", "2026-11-04"]
    closes = {f"sh60000{i}": [10.0, 10.2, 10.3] for i in range(MIN_MEMBERS)}
    root = _store(tmp_path, dates, closes)

    panel = build_panel(root, {sym: "A" for sym in closes}, start="2026-11-01")

    assert panel["date"].min() == "2026-11-03", "the first session with a return"


def test_autocorrelation_reads_continuation_as_positive(tmp_path):
    """An AR(1) return series must show positive next-session correlation."""
    rng = np.random.default_rng(7)
    n = 60
    series = {}
    for name, seed in (("A", 0.0), ("B", 1.0)):
        noise = rng.normal(0, 0.01, n)
        r = np.zeros(n)
        r[0] = noise[0]
        for i in range(1, n):
            r[i] = 0.9 * r[i - 1] + noise[i]
        series[name] = r
    index = [f"2026-{m:02d}-{d:02d}" for m, d in ((1 + i // 28, 1 + i % 28) for i in range(n))]
    wide = pd.DataFrame(series, index=index)
    panel = (
        wide.reset_index()
        .melt(id_vars="index", var_name="industry", value_name="mean_ret")
        .rename(columns={"index": "date"})
    )

    stats = summary(panel, horizons=(1,))

    assert stats["momentum_autocorr_1d"] > 0.5
