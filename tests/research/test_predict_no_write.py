"""The live inference entry runs over an admitted view without a memory write.

The dataset is built here rather than imported from the adapter's test module:
that module imports qlib for real, which fails at collection when an earlier
test left a bare ``qlib`` stub in ``sys.modules``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from tests.stack import require

torch = require("torch")
DataHandlerLP = require("qlib.data.dataset.handler").DataHandlerLP

from ashare_lab.research.predict import predict_without_memory_write  # noqa: E402
from ashare_lab.research.sequence_dataset import (  # noqa: E402
    ContainedMTSDatasetH,
    SegmentAdmission,
)

DATES = pd.to_datetime(
    [
        "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
        "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14",
    ]
)
LABEL = ("label", "LABEL0")


def _frame(dates):
    pairs = [(dates[session], "A") for session in range(len(dates))]
    index = pd.MultiIndex.from_tuples(pairs, names=["datetime", "instrument"])
    columns = pd.MultiIndex.from_tuples(
        [("feature", "f0"), ("feature", "f1"), LABEL]
    )
    values = [[100.0 + s, 10.0 + s, 1.0 + s] for s in range(len(pairs))]
    return pd.DataFrame(values, index=index, columns=columns, dtype=np.float32)


def _labels(dates):
    index = pd.MultiIndex.from_tuples(
        [("A", dates[s]) for s in range(len(dates))],
        names=["instrument", "datetime"],
    )
    return pd.Series([1.0 + s for s in range(len(dates))], index=index, dtype="float64")


def _live_view():
    handler = DataHandlerLP.from_df(_frame(DATES))
    root = ContainedMTSDatasetH(
        handler,
        {
            "test": SegmentAdmission(
                start=DATES[8], end=DATES[8], purpose="live", outcome_cutoff=DATES[8]
            )
        },
        calendar=DATES,
        raw_labels=_labels(DATES),
        seq_len=3,
        num_states=2,
    )
    return root, root.prepare("test")


class _Backbone(torch.nn.Module):
    def forward(self, data):
        # Carry the last timestep of the first feature through, so a score can
        # only match its own instrument's row.
        level = data[:, -1, 0].unsqueeze(1)
        return level.repeat(1, 4)


class _Routing(torch.nn.Module):
    def forward(self, hidden, state):
        scores = hidden.sum(dim=1, keepdim=True)
        return torch.cat([scores, scores + 1.0], dim=1), None, None


def _model(transport=None):
    model = type("StubTRA", (), {})()
    model.fitted = True
    model.alpha = 0.5
    model.model = _Backbone()
    model.tra = _Routing()
    if transport is None:
        model.transport_method = "none"
    else:
        model.transport_method = "router"
        model.transport_fn = transport
    return model


def test_live_inference_returns_rows_without_touching_the_memory():
    root, view = _live_view()
    before = copy.deepcopy(root._memory)

    frame = predict_without_memory_write(_model(), view)

    assert list(frame.columns)[:2] == ["score", "label"]
    assert frame.index.names == ["datetime", "instrument"]
    assert set(frame.index.get_level_values("datetime")) == {DATES[8]}
    assert np.array_equal(root._memory, before)


def test_router_transport_is_used_when_configured():
    root, view = _live_view()
    seen = {}

    def transport(all_preds, label, choice, prob, state_mean, count, method, alpha, training):
        seen["method"] = method
        seen["training"] = training
        return None, all_preds.mean(dim=1) + 5.0, None, None

    frame = predict_without_memory_write(_model(transport), view)

    assert seen == {"method": "router", "training": False}
    # The backbone carries the row's own level (108 on the live session) into
    # all four hidden columns; the routing stub sums them (4 x 108) and emits
    # [4L, 4L + 1]; the transport adds 5 to their mean.
    assert float(frame["score"].iloc[0]) == pytest.approx(4 * 108.0 + 0.5 + 5.0)


def test_an_unfitted_model_is_refused():
    root, view = _live_view()
    model = _model()
    model.fitted = False

    with pytest.raises(ValueError, match="not fitted"):
        predict_without_memory_write(model, view)


def _multi_frame(dates, full, short):
    rows, idx = [], []
    for rank, inst in enumerate(full):
        for s in range(len(dates)):
            idx.append((dates[s], inst))
            # Each name carries its own level so a permuted score is visible.
            rows.append([100.0 * (rank + 1) + s, 10.0 + s, 1.0 + s])
    for inst in short:
        for s in range(len(dates) - 2, len(dates)):
            idx.append((dates[s], inst))
            rows.append([50.0 + s, 5.0 + s, 0.5 + s])
    columns = pd.MultiIndex.from_tuples([("feature", "f0"), ("feature", "f1"), LABEL])
    index = pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"])
    return pd.DataFrame(rows, index=index, columns=columns, dtype=np.float32).sort_index()


def _multi_labels(dates, instruments):
    idx = [(inst, dates[s]) for inst in instruments for s in range(len(dates))]
    return pd.Series(
        [1.0] * len(idx),
        index=pd.MultiIndex.from_tuples(idx, names=["instrument", "datetime"]),
        dtype="float64",
    )


def test_every_admitted_instrument_is_scored_exactly_once():
    instruments = ["A", "B", "C", "D"]
    handler = DataHandlerLP.from_df(_multi_frame(DATES, ("A", "B", "C"), ("D",)))
    root = ContainedMTSDatasetH(
        handler,
        {
            "test": SegmentAdmission(
                start=DATES[8], end=DATES[8], purpose="live", outcome_cutoff=DATES[8]
            )
        },
        calendar=DATES,
        raw_labels=_multi_labels(DATES, instruments),
        seq_len=3,
        num_states=2,
    )
    view = root.prepare("test")

    frame = predict_without_memory_write(_model(), view)

    # D has two sessions of history and cannot fill a three-session window, so
    # the view admits three and the score must describe exactly those three.
    assert view._diagnostics["admitted"] == 3
    assert view._diagnostics["insufficient_own_history"] == 1
    assert len(frame) == view._diagnostics["admitted"]
    assert set(frame.index.get_level_values("instrument")) == {"A", "B", "C"}
    # The rows must stay paired with their instruments: A < B < C by level.
    # Sorting first would assert only that sorting sorts.
    scores = frame["score"].droplevel("datetime")
    assert scores.loc["A"] < scores.loc["B"] < scores.loc["C"]
