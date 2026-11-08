"""Tests for the IC monitor.

The monitor replaces a retrain gate, so its job is narrow: report the mean,
ICIR and a HAC t-statistic over a long window, and stay silent unless the
negative edge is large relative to its own autocorrelated noise.
"""

from __future__ import annotations

import math

import pytest

from ashare_lab.paper.ic_monitor import (
    format_line,
    load_series,
    newey_west_t,
    should_alert,
    summarize,
)


def _pairs(values, start_day=1):
    return [(f"2026-09-{start_day + i:02d}", v) for i, v in enumerate(values)]


def test_load_series_skips_malformed_rows(tmp_path):
    path = tmp_path / "ic_history.tsv"
    path.write_text(
        "2026-09-01\t0.10\n"
        "\n"
        "2026-09-02\t\n"
        "2026-09-03\tnan\n"
        "2026-09-04\t-0.25\n",
        encoding="utf-8",
    )

    pairs = load_series(path)

    assert pairs == [("2026-09-01", 0.10), ("2026-09-04", -0.25)]


def test_load_series_of_a_missing_file_is_empty(tmp_path):
    assert load_series(tmp_path / "absent.tsv") == []


def test_summarize_reports_mean_icir_and_t():
    values = [0.10, 0.20, 0.05, 0.15, 0.00]
    stats = summarize(_pairs(values), window=5)

    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(sum(values) / 5)
    sd = statistics_stdev(values)
    assert stats["sd"] == pytest.approx(sd)
    assert stats["icir"] == pytest.approx((sum(values) / 5) / sd)
    assert stats["t_iid"] == pytest.approx((sum(values) / 5) / (sd / math.sqrt(5)))


def statistics_stdev(values):
    import statistics

    return statistics.stdev(values)


def test_summarize_only_looks_at_the_trailing_window():
    old = _pairs([0.50] * 10, start_day=1)
    recent = _pairs([-0.05, -0.06, -0.04], start_day=20)
    stats = summarize(old + recent, window=3)

    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(-0.05)
    assert stats["first_date"] == "2026-09-20"


def test_a_constant_series_reports_no_t_and_no_alert():
    stats = summarize(_pairs([0.0] * 10), window=10)

    assert stats["sd"] == 0
    assert stats["t_hac"] is None
    assert stats["alert"] is False


def test_autocorrelation_shrinks_the_t_statistic():
    """Same mean, but a series that drifts must not claim the same evidence."""
    drifting = []
    for i in range(60):
        base = -0.05 if i < 30 else -0.03
        drifting.append(base + 0.002 * ((i % 5) - 2))

    t_iid = summarize(_pairs(drifting), window=60)["t_iid"]
    t_hac = summarize(_pairs(drifting), window=60)["t_hac"]

    assert t_hac is not None
    assert abs(t_hac) < abs(t_iid), (t_hac, t_iid)


def test_a_sustained_negative_edge_alerts():
    """Mean -0.05 with sd 0.01 over 60 sessions: unmistakable."""
    values = [-0.05 + 0.01 * ((i % 3) - 1) for i in range(60)]
    stats = summarize(_pairs(values), window=60)

    assert stats["mean"] < 0
    assert stats["t_hac"] < -2
    assert stats["alert"] is True
    assert format_line(stats).startswith("⚠️")


def test_a_negative_mean_inside_the_noise_does_not_alert():
    """This book's actual shape: mean slightly negative, sd 0.23."""
    values = [(-0.001) + 0.2324 * math.sin(2 * math.pi * i / 12) for i in range(60)]
    stats = summarize(_pairs(values), window=60)

    assert stats["mean"] < 0
    assert abs(stats["t_hac"]) < 2
    assert stats["alert"] is False
    assert "⚠️" not in format_line(stats)


def test_a_positive_edge_never_alerts_however_strong():
    values = [0.05 + 0.01 * ((i % 3) - 1) for i in range(60)]
    stats = summarize(_pairs(values), window=60)

    assert stats["t_hac"] > 2
    assert stats["alert"] is False


def test_should_alert_needs_both_conditions():
    assert should_alert({"mean": -0.05, "t_hac": -3.0}) is True
    assert should_alert({"mean": -0.05, "t_hac": -1.0}) is False
    assert should_alert({"mean": 0.05, "t_hac": 3.0}) is False
    assert should_alert({"mean": None, "t_hac": -3.0}) is False


def test_format_line_is_empty_without_history():
    assert format_line(summarize([], window=60)) == ""
    assert format_line(summarize([("2026-09-01", 0.1)], window=60)) == ""


def test_newey_west_matches_the_iid_t_for_white_noise():
    import random

    rng = random.Random(7)
    values = [0.10 + rng.gauss(0, 0.05) for _ in range(200)]

    t_iid = summarize(_pairs(values), window=200)["t_iid"]
    t_hac = newey_west_t(values, bandwidth=5)

    assert t_hac == pytest.approx(t_iid, rel=0.5)
