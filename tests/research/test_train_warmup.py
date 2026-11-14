"""Test scoring must select held-out dates without dropping feature history."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest


class _DatasetObserved(Exception):
    pass


def _real_dataset_case(tmp_path, model_type):
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.contrib.data.dataset import MTSDatasetH
    from qlib.data.dataset import TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    import ashare_lab.research.train as train_mod
    dates = pd.bdate_range("2026-06-01", periods=30)
    test_start, test_end = dates[20], dates[23]
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {
            ("feature", "x"): np.repeat(np.arange(len(dates), dtype=float), 2),
            ("label", "return"): np.tile([0.01, -0.01], len(dates)),
        }, index=index,
    )
    qlib.init(provider_uri=str(tmp_path), region="cn", kernels=1)
    def handler_factory(**kwargs):
        assert kwargs["fit_start_time"] == "2026-06-01"
        assert kwargs["fit_end_time"] == "2026-06-12"
        assert kwargs["instruments"] == "csi1000"
        # Honor the real constructor's bounds so dropping history at that call
        # cannot be hidden by a pre-populated handler returned unconditionally.
        start = pd.Timestamp(kwargs["start_time"])
        stop = pd.Timestamp(kwargs["end_time"])
        selected = frame.loc[(frame.index.get_level_values("datetime") >= start) &
                             (frame.index.get_level_values("datetime") <= stop)]
        handler = DataHandlerLP.from_df(selected)
        handler.data_loader.fields = {"label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])}
        return handler

    cfg = {"model": {"type": model_type, "step_len": 20, "routing": {"num_states": 3}}}
    train_mod.load_config = lambda: cfg
    seen = {}

    def observe(*args, **kwargs):
        if model_type == "tra":
            dataset = MTSDatasetH(*args, **kwargs)
            selected = dataset.prepare("test")
            selected.eval()
            batches = list(selected)
            scored = dataset.restore_index(np.concatenate([batch["index"] for batch in batches]))
            seen["dates"] = sorted(set(scored.get_level_values(1)))
            seen["first_features"] = batches[0]["data"][0, :, 0]
        else:
            dataset = TSDatasetH(*args, **kwargs)
            selected = dataset.prepare("test", col_set="feature")
            seen["dates"] = sorted(set(selected.get_index().get_level_values(0)))
            seen["first_features"] = selected[0][:, 0]
        raise _DatasetObserved

    window = {
        "window_id": 11,
        "train_start": dates[0].date().isoformat(), "train_end": dates[9].date().isoformat(),
        "valid_start": dates[10].date().isoformat(), "valid_end": dates[19].date().isoformat(),
        "test_start": test_start.date().isoformat(), "test_end": test_end.date().isoformat(),
    }
    with mock.patch("qlib.contrib.data.handler.Alpha158", side_effect=handler_factory), mock.patch(
        "qlib.data.D.calendar", return_value=dates,
    ), mock.patch("qlib.contrib.data.dataset.MTSDatasetH", side_effect=observe), mock.patch(
        "qlib.data.dataset.TSDatasetH", side_effect=observe,
    ):
        with pytest.raises(_DatasetObserved):
            train_mod.train_window(window, tmp_path, "csi1000")

    assert seen["dates"] == list(dates[20:24]), "validation/warmup rows entered the score population"
    np.testing.assert_array_equal(seen["first_features"], np.arange(1, 21, dtype=float))


def test_cli_forwards_zero_validation_to_the_validator(tmp_path, monkeypatch):
    import runpy
    import ashare_lab.research.train as train_mod
    from ashare_lab.research import smoke_test
    from ashare_lab.research import windows

    window = {
        "window_id": 11, "train_start": "2022-12-31", "train_end": "2025-12-31",
        "valid_start": "2026-01-01", "valid_end": "2026-06-30",
        "test_start": "2026-07-01", "test_end": "2026-11-06",
    }
    monkeypatch.setattr(smoke_test, "get_all_windows", lambda: [window])
    monkeypatch.setattr(windows, "latest_trading_day", lambda: dt.date(2026, 11, 6))
    monkeypatch.setattr(sys, "argv", [
        "train", "--windows", "11", "--valid-months", "0", "--exp-dir", str(tmp_path),
    ])
    # A fake existing model prevents a broken CLI guard from starting a fit.
    (tmp_path / "models").mkdir()
    (tmp_path / "models/w11.pt").write_bytes(b"do not load")
    with pytest.raises(ValueError, match="validation set"):
        runpy.run_path(str(Path(train_mod.__file__)), run_name="__main__")


def _run_dataset_probe(tmp_path, *args, timeout=60):
    # A fresh process uses the installed Qlib, not stubs from test collection.
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", MLFLOW_ALLOW_FILE_STORE="true",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_CACHE_DIR=str(tmp_path / "numba"))
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), str(tmp_path), *args],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("model_type", ["tra", "alstm"])
def test_time_series_scores_test_dates_but_keeps_preceding_features(tmp_path, model_type):
    _run_dataset_probe(tmp_path, model_type)


def _label_population_case(tmp_path, model_type, available, label_case=None):
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    import ashare_lab.research.train as train_mod

    qlib.init(provider_uri=str(tmp_path), region="cn", kernels=1)
    dates = pd.bdate_range(end="2026-11-10", periods=45)
    index = pd.MultiIndex.from_product([dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"])
    labels = np.tile([0.01, -0.01], len(dates)).astype(float)
    labels[-4:] = np.nan
    if available:
        labels[-2:] = [0.01, -0.01]
    features = np.repeat(np.arange(len(dates), dtype=float), 2)
    infer_processors = []
    learn_processors: list[dict[str, str | dict[str, str]]] = [{"class": "DropnaLabel"}]
    if label_case in {"imputed", "disjoint"}:
        infer_processors = [{"class": "Fillna", "kwargs": {"fields_group": "label"}}]
    if label_case == "disjoint":
        labels[-2] = 0.01
        features[-2] = np.nan
        learn_processors.append({"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}})
    elif label_case == "infinite":
        labels[-2:] = [np.inf, -np.inf]
    frame = pd.DataFrame({("feature", "x"): features, ("label", "LABEL0"): labels}, index=index)
    window = {"window_id": 11, "train_start": "2026-09-09", "train_end": "2026-10-09",
              "valid_start": "2026-10-10", "valid_end": "2026-11-09",
              "test_start": "2026-11-10", "test_end": "2026-11-10"}
    config = {"model": {"type": model_type, "handler": "alpha158", "step_len": 20,
                        "routing": {"num_states": 3}}}
    handlers = []

    def handler_factory(**kwargs):
        handler = DataHandlerLP(data_loader=StaticDataLoader(frame), infer_processors=infer_processors,
                                learn_processors=learn_processors)
        handler.data_loader.fields = {"label": (["Ref($close,-2)/Ref($close,-1)-1"], ["LABEL0"])}
        handlers.append(handler)
        return handler

    class ReachedModel(Exception):
        pass

    with mock.patch.object(train_mod, "load_config", return_value=config), mock.patch(
        "qlib.contrib.data.handler.Alpha158", side_effect=handler_factory,
    ), mock.patch("qlib.data.D.calendar", return_value=dates), mock.patch(
        "qlib.contrib.model.pytorch_tra.TRAModel", side_effect=ReachedModel,
    ), mock.patch("qlib.contrib.model.pytorch_alstm_ts.ALSTM", side_effect=ReachedModel), mock.patch(
        "qlib.contrib.model.gbdt.LGBModel", side_effect=ReachedModel,
    ), mock.patch("qlib.contrib.model.double_ensemble.DEnsembleModel", side_effect=ReachedModel):
        if available:
            with pytest.raises(ReachedModel):
                train_mod.train_window(window, tmp_path, "csi1000")
        else:
            with pytest.raises(ValueError, match="test segment has no labelled samples"):
                train_mod.train_window(window, tmp_path, "csi1000")
    if label_case is not None:
        day = slice(window["test_start"], window["test_end"])
        raw = handlers[0].fetch(day, col_set="label", data_key="raw").iloc[:, 0]
        learned = handlers[0].fetch(day, col_set="label", data_key="learn").iloc[:, 0]
        if label_case == "imputed":
            assert raw.isna().all() and learned.tolist() == [0.0, 0.0]
        elif label_case == "disjoint":
            assert raw.dropna().index.get_level_values("instrument").tolist() == ["SH600000"]
            assert learned.index.get_level_values("instrument").tolist() == ["SH600001"]
            assert learned.tolist() == [0.0]
        else:
            assert np.isinf(raw).all() and np.isinf(learned).all()
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("model_type", ["tra", "alstm", "lgbm", "densemble"])
@pytest.mark.parametrize("available", [False, True])
def test_missing_forward_labels_refuse_before_building_a_model(tmp_path, model_type, available):
    _run_dataset_probe(tmp_path, model_type, str(int(available)))


@pytest.mark.parametrize("label_case", ["imputed", "disjoint", "infinite"])
def test_finite_labels_must_match_on_same_sample(tmp_path, label_case):
    _run_dataset_probe(tmp_path, "lgbm", "0", label_case)


def _file_backed_score_case(tmp_path, model_type, handler_name, population):
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.contrib.data import handler as handlers
    from qlib.contrib.model import double_ensemble, gbdt, pytorch_alstm, pytorch_alstm_ts, pytorch_tra
    import ashare_lab.research.train as train_mod

    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    dates = pd.bdate_range("2024-09-02", periods=100).strftime("%Y-%m-%d").tolist()
    (provider / "calendars/day.txt").write_text("\n".join(dates) + "\n")
    codes = ["SH600000", "SH600001", "SH600002"]
    instruments = "".join(f"{code}\t{dates[0]}\t{dates[-1]}\n" for code in codes)
    (provider / "instruments/csi1000.txt").write_text(instruments)
    for i, code in enumerate(codes):
        directory = provider / "features" / code.lower()
        directory.mkdir(parents=True)
        step = np.arange(len(dates), dtype=np.float32)
        close = 10 + i * 2 + step * .02 + np.sin(step / (3 + i)) * .2
        # The two future closes of the interior flat-return day are equal.
        close[91:93] = 10 + i * 2
        fields = {"close": close, "open": close - .02, "high": close + .1,
                  "low": close - .1, "vwap": close + .01, "volume": 10000 + step * 100}
        for field, values in fields.items():
            np.r_[np.float32(0), values].astype("<f4").tofile(directory / f"{field}.day.bin")

    qlib.init(provider_uri=str(provider), region="cn", kernels=1, expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (tmp_path / "mlruns").as_uri(), "default_exp_name": "score-fixture"}})
    assert Path(train_mod.__file__).resolve().parents[2] == Path(__file__).resolve().parents[2]
    bounds = {"tail": (dates[99], dates[99]), "flat": (dates[90], dates[90]),
              "mixed": (dates[96], dates[99]), "serve-now": ("2025-01-20", dates[99])}
    start, end = bounds[population]
    window = {"window_id": 11, "train_start": dates[20], "train_end": dates[59],
              "valid_start": dates[60], "valid_end": dates[79], "test_start": start, "test_end": end}
    config = {"model": {"type": model_type, "handler": handler_name, "step_len": 7,
                        "routing": {"num_states": 3}, "d_feat": 6}}
    model_boundary = mock.Mock(side_effect=_ModelObserved)
    name = {"alpha158": "Alpha158", "alpha360": "Alpha360"}[handler_name]
    handler_class = getattr(handlers, name)
    captured = []

    def capture_handler(**kwargs):
        handler = handler_class(**kwargs)
        captured.append(handler)
        return handler

    with mock.patch.object(train_mod, "load_config", return_value=config), mock.patch.object(
        handlers, name, side_effect=capture_handler,
    ), mock.patch.object(pytorch_tra, "TRAModel", model_boundary), mock.patch.object(
        pytorch_alstm, "ALSTM", model_boundary,
    ), mock.patch.object(pytorch_alstm_ts, "ALSTM", model_boundary), mock.patch.object(
        gbdt, "LGBModel", model_boundary,
    ), mock.patch.object(double_ensemble, "DEnsembleModel", model_boundary):
        if population in {"tail", "flat"}:
            with pytest.raises(ValueError, match="test segment has no labelled samples"):
                train_mod.train_window(window, tmp_path / "output", "csi1000")
            model_boundary.assert_not_called()
        else:
            with pytest.raises(_ModelObserved):
                train_mod.train_window(window, tmp_path / "output", "csi1000")
            model_boundary.assert_called_once()

    selection = slice(dates[60], dates[79]) if population == "serve-now" else slice(start, end)
    raw = captured[0].fetch(selection, col_set="label", data_key="raw").iloc[:, 0]
    learned = captured[0].fetch(selection, col_set="label", data_key="learn").iloc[:, 0]
    if population == "tail":
        assert len(raw) == len(learned) == 3 and raw.isna().all() and learned.isna().all()
    elif population == "flat":
        assert len(raw) == len(learned) == 3 and (raw == 0).all() and learned.isna().all()
    else:
        expected = 6 if population == "mixed" else 60
        assert np.isfinite(raw).sum() == expected
        assert (np.isfinite(learned) & np.isfinite(raw.reindex(learned.index))).sum() == expected
    assert not (tmp_path / "output").exists()


class _ModelObserved(Exception):
    pass


@pytest.mark.parametrize("model_type,handler_name,population", [
    ("alstm", "alpha360", "tail"),
    ("alstm", "alpha360", "flat"),
    ("alstm", "alpha360", "mixed"),
    ("alstm", "alpha360", "serve-now"),
    ("alstm", "alpha158", "flat"),
    ("tra", "alpha158", "flat"),
    ("lgbm", "alpha360", "tail"),
    ("densemble", "alpha360", "tail"),
])
def test_file_backed_score_requires_finite_labels(tmp_path, model_type, handler_name, population):
    _run_dataset_probe(tmp_path, "provider", model_type, handler_name, population, timeout=90)


if __name__ == "__main__":
    if sys.argv[2] == "provider":
        _file_backed_score_case(Path(sys.argv[1]), *sys.argv[3:])
    elif len(sys.argv) in (4, 5):
        _label_population_case(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])), *sys.argv[4:])
    else:
        _real_dataset_case(Path(sys.argv[1]), sys.argv[2])
