"""The live window and the routing that sends future dates to it."""

from __future__ import annotations

import json

from unittest.mock import patch


def test_live_window_ends_on_the_last_trading_day():
    """Three rolling years ending on the last trading day.

    The walk-forward windows cannot do this: a window's training data has to
    end before its own validation and test spans, so the freshest one is months
    behind. The live window exists to be current. Its validation span overlaps
    training on purpose -- it only guides early stopping, and the test segment
    mirrors it so the trainer has samples to score.
    """
    from ashare_lab.research.smoke_test import get_live_window, latest_trading_day

    w = get_live_window()
    latest = latest_trading_day()

    assert w["window_id"] == "live"  # the window; the artifact is wlive.pt
    assert w["train_end"] == str(latest)
    assert w["valid_end"] == w["train_end"]
    assert (w["test_start"], w["test_end"]) == (w["valid_start"], w["valid_end"])
    assert w["is_complete"] is False
    assert w["train_start"] > "2020-01-01", "three rolling years, not expanding"


def test_predict_prefers_the_live_model_after_the_last_window(tmp_path):
    """A date past every walk-forward test span loads wlive.pt, not latest.pt."""
    from ashare_lab.research.predict import _resolve_model_path

    models = tmp_path / "models"
    models.mkdir()
    (models / "wlive.pt").write_bytes(b"live model")
    (models / "latest.pt").write_bytes(b"some other model")

    windows = [{
        "window_id": 11,
        "train_start": "2022-01-01", "train_end": "2025-12-31",
        "valid_start": "2026-01-01", "valid_end": "2026-06-30",
        "test_start": "2026-07-01", "test_end": "2026-11-06",
    }]

    with patch("ashare_lab.research.predict._send_alert"):
        path, window = _resolve_model_path(
            "2026-11-09", None, windows, {"research": {}}, models
        )

    assert path.name == "wlive.pt"
    assert window["window_id"] == "wlive"


def test_predict_falls_back_to_latest_without_a_live_model(tmp_path):
    from ashare_lab.research.predict import _resolve_model_path

    models = tmp_path / "models"
    models.mkdir()
    (models / "latest.pt").write_bytes(b"latest model")

    windows = [{
        "window_id": 11,
        "train_start": "2022-01-01", "train_end": "2025-12-31",
        "valid_start": "2026-01-01", "valid_end": "2026-06-30",
        "test_start": "2026-07-01", "test_end": "2026-11-06",
    }]

    with patch("ashare_lab.research.predict._send_alert"):
        path, window = _resolve_model_path(
            "2026-11-09", None, windows, {"research": {}}, models
        )

    assert path.name == "latest.pt"
    assert window["window_id"] == 11


def test_a_named_window_is_recorded_in_the_model_meta(tmp_path):
    """window_id is no longer an int: the live artifact records "wlive"."""
    from ashare_lab.research.model_meta import write_model_meta

    model = tmp_path / "wlive.pt"
    model.write_bytes(b"x")
    write_model_meta(model, "2026-11-06", "wlive")

    meta = json.loads((tmp_path / "wlive.meta.json").read_text())
    assert meta["window_id"] == "wlive"
    assert meta["train_date"] == "2026-11-06"
    assert not (tmp_path / "meta.json").exists(), (
        "the live artifact must not overwrite the walk-forward descriptor"
    )


def test_the_production_date_uses_the_live_model(tmp_path):
    """The newest window's test_end is clamped to the latest trading day.

    So "past every window" never holds on a production run, and a rule keyed on
    that never selects the live artifact. The day being predicted is the
    criterion; historical dates keep their window models.
    """
    from ashare_lab.research.predict import _resolve_model_path
    from ashare_lab.research.smoke_test import latest_trading_day

    models = tmp_path / "models"
    models.mkdir()
    (models / "wlive.pt").write_bytes(b"live model")
    for n in (10, 11):
        (models / f"w{n}.pt").write_bytes(f"window {n}".encode())

    today = str(latest_trading_day())
    windows = [{
        "window_id": 11,
        "train_start": "2022-01-01", "train_end": "2025-12-31",
        "valid_start": "2026-01-01", "valid_end": "2026-06-30",
        "test_start": "2026-07-01", "test_end": today,
    }]

    with patch("ashare_lab.research.predict._send_alert"):
        path, window = _resolve_model_path(today, None, windows, {"research": {}}, models)

    assert path.name == "wlive.pt"
    assert window["window_id"] == "wlive"


def test_a_historical_date_still_uses_its_window_model(tmp_path):
    from ashare_lab.research.predict import _resolve_model_path

    models = tmp_path / "models"
    models.mkdir()
    (models / "wlive.pt").write_bytes(b"live model")
    (models / "w11.pt").write_bytes(b"window 11")

    windows = [{
        "window_id": 11,
        "train_start": "2022-01-01", "train_end": "2025-12-31",
        "valid_start": "2026-01-01", "valid_end": "2026-06-30",
        "test_start": "2026-07-01", "test_end": "2026-11-06",
    }]

    with patch("ashare_lab.research.predict._send_alert"):
        path, _ = _resolve_model_path("2026-08-03", None, windows, {"research": {}}, models)

    assert path.name == "w11.pt"
