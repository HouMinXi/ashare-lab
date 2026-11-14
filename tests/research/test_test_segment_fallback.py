"""A window whose test period has not started must not train on an empty segment."""

from __future__ import annotations

from ashare_lab.research.train import resolve_test_segment


def _window(test_start: str, test_end: str) -> dict:
    return {
        "train_start": "2023-06-30",
        "train_end": "2026-06-30",
        "valid_start": "2026-07-01",
        "valid_end": "2026-12-31",
        "test_start": test_start,
        "test_end": test_end,
    }


def test_a_window_whose_test_period_is_in_the_future_evaluates_on_validation():
    """Window 11's test starts 2027-01-01, after its own end; qlib needs samples.

    An empty test segment makes pytorch_tra read metrics["MSE"] off an empty
    frame and raise KeyError after the first epoch -- which is how the
    2026-11-07 retrain died after 38 minutes of training.
    """
    window = _window("2027-01-01", "2026-11-06")
    assert resolve_test_segment(window) == ("2026-07-01", "2026-12-31")


def test_a_complete_window_keeps_its_test_span():
    window = _window("2026-07-01", "2026-11-06")
    assert resolve_test_segment(window) == ("2026-07-01", "2026-11-06")
