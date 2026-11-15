"""The momentum factor must be computable on the day it is scored.

The expression was once Ref($close, -20)/$close - 1, which qlib evaluates
against the close twenty sessions ahead. Live, that close does not exist yet:
the column came back all-NaN, the z-score turned it into zeros, and a factor
quietly left the regression. These checks run against the real store in a child
process, because the property is about qlib's own Ref semantics.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.stack import require

PROVIDER = Path(os.environ.get("ASHARE_TEST_PROVIDER", "~/.qlib/qlib_data/cn_data")).expanduser()


def _probe():
    import pandas as pd
    import qlib

    qlib.init(provider_uri=str(PROVIDER), region="cn", kernels=1)
    from qlib.data import D

    from ashare_lab.research.neutralize import compute_style_factors

    instruments = ["SH600000", "SH600009", "SH600016", "SH600028", "SH600030",
                   "SH600036", "SH600048", "SH600050", "SH600104", "SH600276",
                   "SH600309", "SH600585", "SH600690", "SH600745", "SH600809"]
    raw = D.features(instruments, ["$close"], "2026-06-01", "2026-12-31")
    close = raw["$close"].unstack("instrument")
    if len(close) < 50:
        raise SystemExit("provider too short for this check")

    pos = len(close) - 25
    day = pd.Timestamp(close.index[pos])
    factors = compute_style_factors(instruments, str(day.date()), str(day.date()))
    momentum = factors.xs(day, level="datetime")["momentum"]

    past = close.iloc[pos] / close.iloc[pos - 20] - 1
    future = close.iloc[pos + 20] / close.iloc[pos] - 1
    frame = pd.concat({"z": momentum, "past": past, "future": future}, axis=1).dropna()

    assert len(frame) >= 10, frame
    assert momentum.notna().all(), "momentum must be populated on the live day"
    assert float(frame["z"].corr(frame["past"])) == pytest.approx(1.0, abs=0.02), frame
    assert float(frame["z"].corr(frame["future"])) < 0.9, frame


def test_momentum_is_the_past_return_on_the_real_store():
    require("qlib", "torch")
    if not PROVIDER.exists():
        pytest.skip(f"no qlib provider at {PROVIDER}")

    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--probe"],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__" and "--probe" in sys.argv:
    _probe()
