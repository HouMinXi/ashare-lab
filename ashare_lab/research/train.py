"""Walk-forward window training: Alpha158 + LGBModel on CSI500/CSI300.

Provides apply_price_filter() and train_window(). All qlib imports are
deferred inside function bodies so this module is importable without a
qlib runtime (required for unit test isolation).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ashare_lab.config import load_config

log = logging.getLogger(__name__)

# Alpha158 requires ~1 year of price history before the first train_start.
# This constant is shared with smoke_test.py; both must use the same value.
ALPHA158_WARMUP_START = "2017-01-01"


def apply_price_filter(
    pred,
    test_start: str,
    test_end: str,
    universe: str,
):
    """Remove predictions for stocks with close above the configured threshold.

    Fetches daily close prices for all instruments in pred over the test
    period via qlib D.features, then drops (date, instrument) pairs where
    close > config["universe"]["exclude_close_above_cny"].

    Args:
        pred: MultiIndex Series (datetime, instrument) -> float. Model
            prediction scores. Returned as-is if empty.
        test_start: Test period start, YYYY-MM-DD.
        test_end: Test period end, YYYY-MM-DD.
        universe: Universe label (e.g. "csi500"). Used for logging only.

    Returns:
        Filtered Series with the same MultiIndex structure as pred.
    """
    from qlib.data import D  # noqa: PLC0415

    if pred.empty:
        log.debug("price filter: empty pred; returning as-is")
        return pred

    config = load_config()
    threshold: float = config["universe"]["exclude_close_above_cny"]

    instruments = pred.index.get_level_values("instrument").unique().tolist()
    if not instruments:
        log.debug("price filter: no instruments in pred; returning as-is")
        return pred

    close_df = D.features(
        instruments=instruments,
        fields=["$close"],
        start_time=test_start,
        end_time=test_end,
    )

    if close_df is None or close_df.empty:
        log.warning(
            "price filter [%s]: no close data returned; returning pred unfiltered",
            universe,
        )
        return pred

    close_series = close_df["$close"]

    # Align to pred index before masking (pred may have dates not in close_df).
    aligned_close = close_series.reindex(pred.index)
    mask = aligned_close.notna() & (aligned_close <= threshold)

    excluded = int((~mask).sum())
    log.info(
        "price filter [%s]: excluded %d (date, instrument) pairs with close > %g CNY",
        universe,
        excluded,
        threshold,
    )

    return pred[mask]


def train_window(
    window: dict,
    exp_dir: Path,
    universe: str,
) -> tuple:
    """Train an LGBModel on one walk-forward window and return predictions.

    Initialises qlib, builds Alpha158 handler with the given universe and
    warmup start ALPHA158_WARMUP_START, wraps it in DatasetH with train/valid/test
    segments from window, trains LGBModel(random_state=42, verbose=-1),
    persists the model to exp_dir/models/w{window_id}.pkl, generates
    predictions on the test set, applies price filter, and extracts labels.

    Args:
        window: WindowDict with keys: window_id, train_start, train_end,
            valid_start, valid_end, test_start, test_end.
        exp_dir: Experiment output directory. The model is saved under
            exp_dir/models/w{window_id}.pkl (directory created if needed).
        universe: Qlib universe string (e.g. "csi500", "csi300").

    Returns:
        Tuple (model_path, pred, label):
            model_path: Path where the model was persisted.
            pred: MultiIndex Series (datetime, instrument) -> float,
                prediction scores AFTER price filter.
            label: MultiIndex Series (datetime, instrument) -> float,
                next-period return labels for the test set.
    """
    import qlib  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
    from qlib.contrib.model.gbdt import LGBModel  # noqa: PLC0415
    from qlib.data.dataset import DatasetH  # noqa: PLC0415

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415

    window_id: int = window["window_id"]

    log.info(
        "W%d: train %s..%s | valid %s..%s | test %s..%s [%s]",
        window_id,
        window["train_start"],
        window["train_end"],
        window["valid_start"],
        window["valid_end"],
        window["test_start"],
        window["test_end"],
        universe,
    )

    qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI), region=REG_CN)

    # Alpha158 is a DataHandlerLP (NOT a Dataset); wrap it in DatasetH.
    # Warmup start ALPHA158_WARMUP_START provides ~1 year of history for lookback
    # features before train_start 2018-01-01.
    handler = Alpha158(
        instruments=universe,
        start_time=ALPHA158_WARMUP_START,
        end_time=window["test_end"],
        fit_start_time=window["train_start"],
        fit_end_time=window["train_end"],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (window["train_start"], window["train_end"]),
            "valid": (window["valid_start"], window["valid_end"]),
            "test": (window["test_start"], window["test_end"]),
        },
    )

    model = LGBModel(random_state=42, verbose=-1)
    model.fit(dataset)

    # Persist model.
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"w{window_id}.pkl"
    model.to_pickle(path=str(model_path))
    log.info("W%d: model saved -> %s", window_id, model_path)

    # Predict on test set; returns MultiIndex Series (datetime, instrument).
    pred = model.predict(dataset, segment="test")

    # Apply price filter (modifies pred in-place logically; returns new Series).
    pred = apply_price_filter(pred, window["test_start"], window["test_end"], universe)

    # Extract test labels: first column of the label DataFrame (typically "LABEL0").
    label = dataset.prepare("test", col_set="label").iloc[:, 0]

    log.info(
        "W%d: %d predictions after price filter, %d labels",
        window_id,
        len(pred),
        len(label),
    )

    return model_path, pred, label
