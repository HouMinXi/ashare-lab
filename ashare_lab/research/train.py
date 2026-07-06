"""Walk-forward window training: multi-model on CSI1000.

Provides apply_price_filter() and train_window(). All qlib imports are
deferred inside function bodies so this module is importable without a
qlib runtime (required for unit test isolation).

Supported model types (config model.type):
  lgbm      -- LightGBM + DatasetH (tabular, GPU OpenCL)
  alstm     -- Attention-LSTM + TSDatasetH (time-series, CUDA)
  tra       -- Temporal Routing Adaptor + MTSDatasetH (multi-pattern, CUDA)
  densemble -- DoubleEnsemble + DatasetH (sample reweighting + feature selection)
"""

from __future__ import annotations

import logging
import os
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

    # qlib D.features returns MultiIndex (instrument, datetime);
    # pred has (datetime, instrument).  Swap levels so reindex aligns correctly.
    if close_series.index.names[0] != pred.index.names[0]:
        close_series = close_series.swaplevel().sort_index()

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
    seed: int | None = None,
) -> tuple:
    """Train a model on one walk-forward window and return predictions.

    Supports four model types selected via config model.type:
      lgbm      -- LightGBM + DatasetH; saves w{id}.pkl via to_pickle()
      alstm     -- Attention-LSTM + TSDatasetH (or DatasetH for alpha360); saves w{id}.pt via torch.save()
      tra       -- Temporal Routing Adaptor + MTSDatasetH; saves w{id}.pt via torch.save()
      densemble -- DoubleEnsemble + DatasetH; saves w{id}.pkl via to_pickle()

    Initialises qlib once per process (guarded), builds the handler for the
    configured feature set (Alpha158 or Alpha360), wraps it in the appropriate
    dataset class, trains the model, persists it to exp_dir/models/, generates
    predictions on the test set, applies price filter, and extracts labels.

    TRA note: TRAModel.predict() returns a DataFrame with columns
    [score, label, score_0..N].  The 'score' column is extracted as the
    prediction Series; labels are taken from the 'label' column (processed
    by learn_procs; rank-IC-safe since rank correlation is scale-invariant).

    Args:
        window: WindowDict with keys: window_id, train_start, train_end,
            valid_start, valid_end, test_start, test_end.
        exp_dir: Experiment output directory. The model is saved under
            exp_dir/models/w{window_id}.{pkl|pt} (directory created if needed).
        universe: Qlib universe string (e.g. "csi1000", "csi500").

    Returns:
        Tuple (model_path, pred, label):
            model_path: Path where the model was persisted.
            pred: MultiIndex Series (datetime, instrument) -> float,
                prediction scores AFTER price filter.
            label: MultiIndex Series (datetime, instrument) -> float,
                next-period return labels for the test set.
    """
    # mlflow >=2.x deprecated the filesystem tracking backend used by qlib's
    # experiment recorder.  Set the opt-out flag before any qlib/mlflow import
    # so the file store remains usable until a tracking backend migration is done.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import qlib  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.contrib.data.handler import Alpha158, Alpha360  # noqa: PLC0415
    from qlib.contrib.model.gbdt import LGBModel  # noqa: PLC0415
    from qlib.data.dataset import DatasetH, TSDatasetH  # noqa: PLC0415
    from qlib.contrib.data.dataset import MTSDatasetH  # noqa: PLC0415

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

    # qlib.init() must only be called once per process; subsequent calls while
    # a QlibRecorder is active raise RecorderInitializationError.  Guard here
    # so walk-forward windows W2..W6 don't re-initialize.
    from qlib.config import C as _QlibC  # noqa: PLC0415

    if not getattr(_QlibC, "registered", False):
        qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI), region=REG_CN)

    # Select data handler from config (alpha158 or alpha360).
    cfg_handler = load_config().get("model", {}).get("handler", "alpha158").lower()
    HandlerClass = Alpha360 if cfg_handler == "alpha360" else Alpha158  # noqa: N806
    log.info("W%d: handler=%s", window_id, cfg_handler)

    cfg_model = load_config().get("model", {})
    model_type = cfg_model.get("type", "lgbm").lower()

    VALID_MODEL_TYPES = {"lgbm", "alstm", "tra", "densemble"}
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"unknown model.type={model_type!r}, expected one of {sorted(VALID_MODEL_TYPES)}"
        )

    effective_seed = seed if seed is not None else 42

    # -----------------------------------------------------------------------
    # Build data handler + dataset.
    # Neural-network models need feature normalisation.  Dataset class differs:
    #   lgbm  -> DatasetH (tabular, no time-axis batching)
    #   alstm -> TSDatasetH (fixed-length rolling windows)
    #   tra   -> MTSDatasetH (per-stock memory, required by TRAModel)
    # -----------------------------------------------------------------------
    if model_type in ("alstm", "tra"):
        learn_procs = [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            # Scope Fillna to features only (NOT labels):
            #   - Labels are already clean after DropnaLabel + CSZScoreNorm.
            #   - Unscoped Fillna would silently replace NaN labels with 0
            #     (the median after z-score), corrupting degenerate training
            #     samples instead of surfacing the error.
            # ALSTM fix: NaN features -> NaN predictions -> NaN IC ->
            #   best_param never assigned -> UnboundLocalError at model load.
        ]
        # infer_processors is intentionally omitted for neural models:
        #   TRA:   MTSDatasetH always uses _learn data (source FIXME: cannot switch to _infer).
        #   ALSTM: qlib 0.9.7 bug -- passing infer_processors triggers TypeError
        #          (LocalDatasetProvider.dataset() gets inst_processors twice).
        #   Feature NaN is handled by Fillna in learn_procs (applied to all splits)
        #   and by np.nan_to_num() inside MTSDatasetH.setup_data() for TRA.
    else:
        learn_procs = [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ]

    # HandlerLP wraps around DatasetH/TSDatasetH/MTSDatasetH; warmup provides lookback history.
    handler = HandlerClass(
        instruments=universe,
        start_time=ALPHA158_WARMUP_START,
        end_time=window["test_end"],
        fit_start_time=window["train_start"],
        fit_end_time=window["train_end"],
        learn_processors=learn_procs,
    )

    segs = {
        "train": (window["train_start"], window["train_end"]),
        "valid": (window["valid_start"], window["valid_end"]),
        "test": (window["test_start"], window["test_end"]),
    }

    if model_type == "alstm":
        step_len = cfg_model.get("step_len", 20)
        if cfg_handler == "alpha360":
            # Alpha360 produces 360 flat features (6 fields x 60 lags).
            # ALSTM.forward reshapes (N, 360) -> (N, 6, 60) -> (N, 60, 6)
            # internally via view + permute.  Use flat DatasetH to avoid
            # double-windowing (TSDatasetH would yield (N, step_len, 360)).
            dataset = DatasetH(handler=handler, segments=segs)
            log.info("W%d: ALSTM alpha360 flat path (DatasetH)", window_id)
        else:
            dataset = TSDatasetH(handler=handler, segments=segs, step_len=step_len)
    elif model_type == "tra":
        routing_cfg: dict = dict(cfg_model.get("routing", {}))
        num_states: int = routing_cfg.get("num_states", 0)
        if num_states < 1:
            raise ValueError(
                "model.routing.num_states must be >= 1 for TRA "
                "(got %r); add routing.num_states to configs/baseline.yaml" % num_states
            )
        tra_batch_size: int = cfg_model.get("batch_size", -1)
        if tra_batch_size == 0:
            raise ValueError(
                "model.batch_size must not be 0 (use -1 for daily mode, >0 for sample mode)"
            )
        step_len = cfg_model.get("step_len", 60)
        if cfg_handler == "alpha360":
            # Alpha360 + MTSDatasetH is incompatible at input_size=6:
            # Alpha360's 360 cols = 6 fields x 60 lags (already windowed).
            # MTSDatasetH with seq_len=60 double-windows to (N, 60, 360).
            # seq_len=1 breaks TRA's routing memory (hist_loss length 0).
            # TRA with Alpha360 requires input_size=360 or a custom handler.
            raise ValueError(
                "TRA with handler=alpha360 and input_size=6 is not supported: "
                "Alpha360 produces 360 flat features (6 fields x 60 lags) "
                "which conflicts with MTSDatasetH windowing. Use handler=alpha158 "
                "with input_size=158, or set backbone.input_size=360."
            )
        dataset = MTSDatasetH(
            handler=handler,
            segments=segs,
            seq_len=step_len,
            num_states=num_states,
            memory_mode="sample",
            batch_size=tra_batch_size,
            shuffle=True,
        )
        log.info("W%d: MTSDatasetH seq_len=%d num_states=%d", window_id, step_len, num_states)
        # ponytail: pin_memory on MTSDatasetH iter; upgrade to DataLoader if
        # profiling shows host-to-device transfer is still the bottleneck.
        if cfg_model.get("use_amp", False):
            import torch as _torch  # noqa: PLC0415

            if _torch.cuda.is_available():
                _orig_iter = dataset.__class__.__iter__

                def _pinned_iter(self):
                    for batch in _orig_iter(self):
                        yield {
                            k: v.pin_memory()
                            if hasattr(v, "pin_memory") and not v.is_cuda
                            else v
                            for k, v in batch.items()
                        }

                dataset.__class__ = type(
                    dataset.__class__.__name__ + "Pinned",
                    (dataset.__class__,),
                    {"__iter__": _pinned_iter},
                )
    else:
        # lgbm and densemble both use DatasetH (tabular, no time-axis batching).
        dataset = DatasetH(handler=handler, segments=segs)

    # -----------------------------------------------------------------------
    # Build model.
    # -----------------------------------------------------------------------
    if model_type == "alstm":
        d_feat = cfg_model.get("d_feat", 360)
        if cfg_handler == "alpha360":
            # Flat variant: takes (N, 360) and reshapes to (N, 60, 6) internally.
            from qlib.contrib.model.pytorch_alstm import ALSTM  # noqa: PLC0415
        else:
            # TS variant: takes TSDatasetH output directly.
            from qlib.contrib.model.pytorch_alstm_ts import ALSTM  # noqa: PLC0415
        log.info("W%d: ALSTM d_feat=%d step_len=%d GPU=%s", window_id, d_feat, step_len, cfg_model.get("GPU", 0))
        model = ALSTM(
            d_feat=d_feat,
            hidden_size=cfg_model.get("hidden_size", 64),
            num_layers=cfg_model.get("num_layers", 2),
            dropout=cfg_model.get("dropout", 0.3),
            n_epochs=cfg_model.get("n_epochs", 100),
            lr=cfg_model.get("lr", 1e-3),
            batch_size=cfg_model.get("batch_size", 800),
            early_stop=cfg_model.get("early_stop", 20),
            GPU=cfg_model.get("GPU", 0),
            seed=effective_seed,
        )
        _original_fit_alstm = None
        if cfg_model.get("use_amp", False):
            import torch as _torch  # noqa: PLC0415

            _original_fit_alstm = model.fit

            def _amp_fit(dataset, *a, **kw):
                with _torch.amp.autocast(device_type="cuda", dtype=_torch.bfloat16):
                    return _original_fit_alstm(dataset, *a, **kw)

            model.fit = _amp_fit
            log.info("W%d: AMP enabled (bfloat16)", window_id)
        try:
            model.fit(dataset)
        finally:
            if _original_fit_alstm is not None:
                model.fit = _original_fit_alstm  # restore for pickling
    elif model_type == "tra":
        # TRAModel: Temporal Routing Adaptor (KDD 2021).
        # GPU device is auto-detected via torch.cuda.is_available() at module level;
        # no explicit GPU parameter in TRAModel constructor.
        from qlib.contrib.model.pytorch_tra import TRAModel  # noqa: PLC0415

        backbone_cfg: dict = dict(cfg_model.get("backbone", {}))
        # routing_cfg is already validated and fetched in the MTSDatasetH block above;
        # reuse it here so num_states is guaranteed consistent between dataset and model.
        log.info(
            "W%d: TRAModel backbone=%s routing=%s transport=%s pretrain=%s n_epochs=%d",
            window_id,
            backbone_cfg,
            routing_cfg,
            cfg_model.get("transport_method", "router"),
            cfg_model.get("pretrain", True),
            cfg_model.get("n_epochs", 200),
        )
        model = TRAModel(
            model_config=backbone_cfg,
            tra_config=routing_cfg,
            transport_method=cfg_model.get("transport_method", "router"),
            pretrain=cfg_model.get("pretrain", True),
            n_epochs=cfg_model.get("n_epochs", 200),
            early_stop=cfg_model.get("early_stop", 30),
            lr=cfg_model.get("lr", 1e-3),
            seed=effective_seed,
        )
        _original_fit_tra = None
        if cfg_model.get("use_amp", False):
            import torch as _torch  # noqa: PLC0415

            # bf16 tensor .numpy() crashes -- no numpy bf16 dtype.
            # Cast to fp32 before the original assign_data writes back.
            _orig_assign = dataset.assign_data

            def _safe_assign(index, vals):
                if isinstance(vals, _torch.Tensor):
                    vals = vals.float()
                return _orig_assign(index, vals)

            dataset.assign_data = _safe_assign

            _original_fit_tra = model.fit

            def _amp_fit(dataset, *a, **kw):
                with _torch.amp.autocast(device_type="cuda", dtype=_torch.bfloat16):
                    return _original_fit_tra(dataset, *a, **kw)

            model.fit = _amp_fit
            log.info("W%d: AMP enabled (bfloat16) + assign_data fp32 cast", window_id)
        try:
            model.fit(dataset)
        finally:
            if _original_fit_tra is not None:
                model.fit = _original_fit_tra  # restore for pickling
    elif model_type == "densemble":
        from qlib.contrib.model.double_ensemble import DEnsembleModel  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        # DEnsembleModel has no seed param; best-effort via global np.random.seed.
        # Safe only under subprocess isolation (one train_window call per process).
        np.random.seed(effective_seed)

        log.info("W%d: DEnsembleModel num_models=%d epochs=%d", window_id,
                 cfg_model.get("num_models", 6), cfg_model.get("epochs", 28))
        model = DEnsembleModel(
            base_model="gbm",
            num_models=cfg_model.get("num_models", 6),
            epochs=cfg_model.get("epochs", 28),
            decay=cfg_model.get("decay", 0.5),
            early_stopping_rounds=cfg_model.get("early_stopping_rounds", 10),
            enable_sr=cfg_model.get("enable_sr", True),
            enable_fs=cfg_model.get("enable_fs", True),
            learning_rate=cfg_model.get("learning_rate", 0.2),
            colsample_bytree=cfg_model.get("colsample_bytree", 0.8879),
            subsample=cfg_model.get("subsample", 0.8789),
            lambda_l1=cfg_model.get("lambda_l1", 205.6999),
            lambda_l2=cfg_model.get("lambda_l2", 580.9768),
            max_depth=cfg_model.get("max_depth", 8),
            num_leaves=cfg_model.get("num_leaves", 210),
        )
        model.fit(dataset)
    else:
        lgb_device = cfg_model.get("device", "cpu")
        lgb_kwargs: dict = {
            "loss": "mse",
            "learning_rate": cfg_model.get("learning_rate", 0.0421),
            "colsample_bytree": cfg_model.get("colsample_bytree", 0.8879),
            "subsample": cfg_model.get("subsample", 0.8789),
            "lambda_l1": cfg_model.get("lambda_l1", 205.6999),
            "lambda_l2": cfg_model.get("lambda_l2", 580.9768),
            "max_depth": cfg_model.get("max_depth", 8),
            "num_leaves": cfg_model.get("num_leaves", 210),
            "device": lgb_device,
            "random_state": effective_seed,
            "verbose": -1,
        }
        if lgb_device == "gpu":
            lgb_kwargs["gpu_platform_id"] = cfg_model.get("gpu_platform_id", 0)
            lgb_kwargs["gpu_device_id"] = cfg_model.get("gpu_device_id", 0)
        num_boost_round = cfg_model.get("num_boost_round", 500)
        log.info("W%d: LGB device=%s num_boost_round=%d", window_id, lgb_device, num_boost_round)
        model = LGBModel(**lgb_kwargs)
        model.fit(dataset, num_boost_round=num_boost_round)

    # Persist model.  TRA and ALSTM use torch.save; LGB uses pickle.
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if model_type in ("alstm", "tra"):
        import torch  # noqa: PLC0415

        model_path = models_dir / f"w{window_id}.pt"
        torch.save(model, str(model_path))
    else:
        model_path = models_dir / f"w{window_id}.pkl"
        model.to_pickle(path=str(model_path))
    log.info("W%d: model saved -> %s", window_id, model_path)

    # -----------------------------------------------------------------------
    # Predict on test set and extract labels.
    # TRAModel.predict() returns a DataFrame with columns:
    #   score       -- routing-weighted prediction (the signal we want)
    #   label       -- target return (from learn_procs; CSZScoreNorm applied)
    #   score_0..N  -- per-state raw predictions
    # Extract 'score' Series for downstream use; labels from 'label' column
    # are rank-IC-safe since rank correlation is scale-invariant.
    # Other models return a MultiIndex Series directly.
    # -----------------------------------------------------------------------
    if model_type == "tra":
        pred_df = model.predict(dataset, segment="test")
        pred = pred_df["score"]
        tra_label = pred_df["label"]
    else:
        pred = model.predict(dataset, segment="test")
        tra_label = None

    # Apply price filter (returns new Series with same MultiIndex).
    pred = apply_price_filter(pred, window["test_start"], window["test_end"], universe)

    # Extract test labels.
    if model_type == "tra":
        # Align TRA label to the (possibly reduced) filtered pred index.
        label = tra_label.reindex(pred.index)
    else:
        # DatasetH (LGB) prepare() returns a DataFrame with .iloc.
        # TSDatasetH (ALSTM) prepare() returns TSDataSampler -- no .iloc.
        label_data = dataset.prepare("test", col_set="label")
        if hasattr(label_data, "iloc"):
            # LGB / DatasetH: standard pandas access.
            label = label_data.iloc[:, 0]
        else:
            # ALSTM / TSDatasetH: TSDataSampler has no pandas interface.
            # Fetch raw labels from the underlying handler; Spearman rank-IC
            # is scale-invariant so raw (unprocessed) labels give identical
            # IC to z-score normalised learn-set labels.
            from qlib.data.dataset.handler import DataHandlerLP  # noqa: PLC0415

            raw = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
            # Handler MultiIndex order may be (instrument, datetime);
            # pred uses (datetime, instrument) -- swap if needed.
            if raw.index.names[0] != "datetime":
                raw = raw.swaplevel().sort_index()
            label = raw.loc[segs["test"][0] : segs["test"][1]].iloc[:, 0]

    log.info(
        "W%d: %d predictions after price filter, %d labels",
        window_id,
        len(pred),
        len(label),
    )

    return model_path, pred, label
