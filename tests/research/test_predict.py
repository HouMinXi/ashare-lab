"""Unit tests for ashare_lab.research.predict (mocked, no qlib/torch).

The predict module uses DEFERRED imports (torch, qlib, MTSDatasetH, etc.)
inside predict_for_date(). Those qlib submodules import torch at MODULE
level, so monkeypatch.setattr("qlib.contrib.data.dataset.MTSDatasetH", ...)
would trigger the real import chain and fail without torch installed.

Solution: inject mock modules into sys.modules BEFORE predict_for_date
runs, so its "from qlib.contrib.data.dataset import MTSDatasetH" resolves
to a pre-planted mock without touching the real qlib files.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# sys.modules stubs for torch / qlib (installed before each test)
# ---------------------------------------------------------------------------

def _install_fake_modules():
    """Populate sys.modules with stubs so deferred imports never hit torch.

    Returns a cleanup function that removes the injected modules.
    """
    injected = []

    def _ensure(name, attrs=None):
        if name not in sys.modules:
            mod = ModuleType(name)
            if attrs:
                for k, v in attrs.items():
                    setattr(mod, k, v)
            sys.modules[name] = mod
            injected.append(name)
        return sys.modules[name]

    # torch
    torch_mod = _ensure("torch")
    torch_mod.load = MagicMock()

    # qlib chain
    _ensure("qlib")
    qlib_config = _ensure("qlib.config")
    mock_c = MagicMock()
    mock_c.registered = True
    qlib_config.C = mock_c
    qlib_config.REG_CN = "cn"

    _ensure("qlib.contrib")
    _ensure("qlib.contrib.data")
    qlib_handler = _ensure("qlib.contrib.data.handler")
    qlib_handler.Alpha158 = MagicMock()

    qlib_dataset = _ensure("qlib.contrib.data.dataset")
    qlib_dataset.MTSDatasetH = MagicMock()

    # qlib.init
    sys.modules["qlib"].init = MagicMock()

    def cleanup():
        for name in reversed(injected):
            sys.modules.pop(name, None)

    return cleanup


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_windows():
    """Two walk-forward windows for testing."""
    return [
        {
            "window_id": 1,
            "train_start": "2018-01-01",
            "train_end": "2020-12-31",
            "valid_start": "2021-01-01",
            "valid_end": "2021-06-30",
            "test_start": "2021-07-01",
            "test_end": "2021-12-31",
        },
        {
            "window_id": 2,
            "train_start": "2018-01-01",
            "train_end": "2021-06-30",
            "valid_start": "2021-07-01",
            "valid_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2022-06-30",
        },
    ]


def _make_pred_df(trade_date, instruments=None, include_nan=False):
    """Build a mock TRAModel.predict() DataFrame.

    Returns a DataFrame with columns [score, label] indexed by
    (datetime, instrument) MultiIndex, covering trade_date and one
    prior day (to verify date slicing works).
    """
    if instruments is None:
        instruments = ["SH600000", "SH600001", "SH600002"]
    dates = [pd.Timestamp("2021-11-29"), pd.Timestamp(trade_date)]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    scores = np.arange(len(idx), dtype=float) * 0.1 + 0.5
    if include_nan:
        # Make the first instrument on trade_date NaN.
        scores[len(instruments)] = np.nan
    return pd.DataFrame(
        {"score": scores, "label": np.zeros(len(idx))},
        index=idx,
    )


def _stub_model(pred_df):
    """Return a mock model whose .predict() returns pred_df."""
    model = MagicMock()
    model.predict.return_value = pred_df
    # Deliberately do NOT set _writer so the shim test can verify it.
    if hasattr(model, "_writer"):
        del model._writer
    return model


def _apply_patches(monkeypatch, tmp_path, windows=None, pred_df=None):
    """Wire up all patches for predict_for_date.

    Patches ashare_lab.config constants and stubs get_all_windows,
    load_config, blend_tra_ntra, and the torch.load call (via
    sys.modules["torch"]).

    Returns (model_stub, blend_calls).
    """
    if windows is None:
        windows = _make_windows()
    trade_date = "2021-12-01"
    if pred_df is None:
        pred_df = _make_pred_df(trade_date)

    model = _stub_model(pred_df)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    preds_dir = tmp_path / "predictions"

    # Create model files that the window resolver expects.
    for w in windows:
        (models_dir / f"w{w['window_id']}.pt").write_bytes(b"fake")
    (models_dir / "latest.pt").write_bytes(b"fake")

    # Patch config-level constants.
    monkeypatch.setattr("ashare_lab.config.MODELS_DIR", models_dir)
    monkeypatch.setattr("ashare_lab.config.PREDICTIONS_DIR", preds_dir)

    # Patch get_all_windows.
    monkeypatch.setattr(
        "ashare_lab.research.smoke_test.get_all_windows",
        lambda: windows,
    )

    # Patch load_config.
    monkeypatch.setattr(
        "ashare_lab.config.load_config",
        lambda: {
            "universe": {"primary": "csi1000"},
            "model": {
                "step_len": 20,
                "routing": {"num_states": 3},
            },
        },
    )

    # Patch ALPHA158_WARMUP_START.
    monkeypatch.setattr(
        "ashare_lab.research.train.ALPHA158_WARMUP_START", "2017-01-01"
    )

    # Patch DEFAULT_PROVIDER_URI.
    monkeypatch.setattr(
        "ashare_lab.data.update.DEFAULT_PROVIDER_URI",
        tmp_path / "qlib_data",
    )

    # Wire torch.load to return our stub model (via sys.modules stub).
    sys.modules["torch"].load = MagicMock(
        side_effect=lambda path, **kw: model
    )

    # Patch blend_tra_ntra: identity (returns pred unchanged).
    blend_calls = []

    def fake_blend(pred, window, **kwargs):
        blend_calls.append((pred, window, kwargs))
        result = pred.copy()
        result.name = "score"
        return result

    monkeypatch.setattr(
        "ashare_lab.research.blend.blend_tra_ntra", fake_blend
    )

    return model, blend_calls


@pytest.fixture(autouse=True)
def _fake_heavy_modules():
    """Install fake torch/qlib modules for all tests in this module."""
    cleanup = _install_fake_modules()
    yield
    cleanup()
    # Also clear any cached predict module so patches don't leak.
    sys.modules.pop("ashare_lab.research.predict", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictForDate:
    """Core predict_for_date behavior."""

    def test_writes_parquet_with_correct_columns(self, monkeypatch, tmp_path):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        out = predict_for_date(trade_date)

        assert out.exists()
        assert out.name == f"{trade_date}.parquet"

        df = pd.read_parquet(out)
        assert list(df.columns) == ["instrument", "score"]
        assert df["instrument"].dtype == object  # str
        assert df["score"].dtype == float

    def test_row_count_matches_trade_date_instruments(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2021-12-01"
        instruments = ["SH600000", "SH600001", "SH600002"]
        pred_df = _make_pred_df(trade_date, instruments=instruments)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        out = predict_for_date(trade_date)
        df = pd.read_parquet(out)

        # Only rows for trade_date, not the extra prior day.
        assert len(df) == len(instruments)

    def test_writer_shim_applied(self, monkeypatch, tmp_path):
        """model._writer must be set to None unconditionally."""
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        model, _ = _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        assert model._writer is None

    def test_blend_tra_ntra_called_with_correct_args(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _, blend_calls = _apply_patches(
            monkeypatch, tmp_path, pred_df=pred_df
        )

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        assert len(blend_calls) == 1
        _, window_arg, kwargs = blend_calls[0]
        # Window must be one of the test windows (W1 contains 2021-12-01).
        assert window_arg["window_id"] == 1
        # No tra_weight override passed -- uses the locked default.
        assert "tra_weight" not in kwargs

    def test_meta_json_written(self, monkeypatch, tmp_path):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        meta_path = tmp_path / "predictions" / f"{trade_date}.meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["universe"] == "csi1000"
        assert meta["window_id"] == 1
        assert meta["n_instruments"] == 3
        assert "produced_at" in meta
        assert "model" in meta


class TestWindowSelection:
    """Window-to-model mapping."""

    def test_backfill_date_in_window1_resolves_w1(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _, blend_calls = _apply_patches(
            monkeypatch, tmp_path, pred_df=pred_df
        )

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        assert blend_calls[0][1]["window_id"] == 1

    def test_backfill_date_in_window2_resolves_w2(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2022-03-15"
        pred_df = _make_pred_df(trade_date)
        _, blend_calls = _apply_patches(
            monkeypatch, tmp_path, pred_df=pred_df
        )

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        assert blend_calls[0][1]["window_id"] == 2

    def test_future_date_resolves_latest_pt(self, monkeypatch, tmp_path):
        """A date after the last window's test_end uses latest.pt."""
        trade_date = "2023-06-01"
        pred_df = _make_pred_df(trade_date)
        model, blend_calls = _apply_patches(
            monkeypatch, tmp_path, pred_df=pred_df
        )

        # Track which path torch.load receives.
        load_calls = []

        def capturing_load(path, **kw):
            load_calls.append(path)
            return model

        sys.modules["torch"].load = MagicMock(side_effect=capturing_load)

        from ashare_lab.research.predict import predict_for_date

        predict_for_date(trade_date)

        assert any("latest.pt" in str(c) for c in load_calls)
        # Window should be the last one (W2).
        assert blend_calls[0][1]["window_id"] == 2

    def test_pre_first_window_raises_valueerror(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2020-01-01"
        pred_df = _make_pred_df(trade_date)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        with pytest.raises(ValueError, match="no trained model"):
            predict_for_date(trade_date)

    def test_missing_model_file_raises_filenotfounderror(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        # Delete the expected model file.
        (tmp_path / "models" / "w1.pt").unlink()

        from ashare_lab.research.predict import predict_for_date

        with pytest.raises(FileNotFoundError, match="does not exist"):
            predict_for_date(trade_date)


class TestMissingDate:
    """Missing trade_date in prediction index."""

    def test_no_rows_for_trade_date_raises_valueerror(
        self, monkeypatch, tmp_path
    ):
        """When model returns no rows for trade_date, raise ValueError."""
        trade_date = "2021-12-01"
        # Build a pred_df that has rows for a DIFFERENT date only.
        other_date = "2021-11-29"
        instruments = ["SH600000", "SH600001"]
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp(other_date)], instruments],
            names=["datetime", "instrument"],
        )
        pred_df = pd.DataFrame(
            {"score": [0.5, 0.6], "label": [0.0, 0.0]}, index=idx
        )

        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        with pytest.raises(ValueError, match="no predictions"):
            predict_for_date(trade_date)


class TestNonFiniteDrop:
    """Non-finite score handling."""

    def test_nan_scores_dropped_from_output(self, monkeypatch, tmp_path):
        """NaN scores are dropped; remaining finite rows are kept."""
        trade_date = "2021-12-01"
        instruments = ["SH600000", "SH600001", "SH600002"]
        pred_df = _make_pred_df(
            trade_date, instruments=instruments, include_nan=True
        )
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        out = predict_for_date(trade_date)
        df = pd.read_parquet(out)

        # First instrument on trade_date was NaN -> dropped.
        assert len(df) == len(instruments) - 1
        assert "SH600000" not in df["instrument"].values
        assert all(np.isfinite(df["score"].values))

    def test_all_nan_scores_raises_valueerror(self, monkeypatch, tmp_path):
        """If ALL trade_date scores are NaN, raise ValueError."""
        trade_date = "2021-12-01"
        instruments = ["SH600000", "SH600001"]
        dates = [pd.Timestamp("2021-11-29"), pd.Timestamp(trade_date)]
        idx = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        scores = np.array([0.5, 0.6, np.nan, np.nan])
        pred_df = pd.DataFrame(
            {"score": scores, "label": np.zeros(4)}, index=idx
        )

        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        from ashare_lab.research.predict import predict_for_date

        with pytest.raises(ValueError, match="non-finite"):
            predict_for_date(trade_date)


class TestBlendExceptionPropagates:
    """Blend failure must propagate -- no raw-fallback file written."""

    def test_blend_exception_propagates_no_file_written(
        self, monkeypatch, tmp_path
    ):
        trade_date = "2021-12-01"
        pred_df = _make_pred_df(trade_date)
        _apply_patches(monkeypatch, tmp_path, pred_df=pred_df)

        def exploding_blend(pred, window, **kwargs):
            raise RuntimeError("neutralization failed")

        monkeypatch.setattr(
            "ashare_lab.research.blend.blend_tra_ntra", exploding_blend
        )

        from ashare_lab.research.predict import predict_for_date

        with pytest.raises(RuntimeError, match="neutralization failed"):
            predict_for_date(trade_date)

        # No prediction file should have been written.
        preds_dir = tmp_path / "predictions"
        assert not (preds_dir / f"{trade_date}.parquet").exists()
