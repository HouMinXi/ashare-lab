"""Behavioral tests for the contained sequence dataset adapter.

Every fixture is a real DataHandlerLP built from a literal synthetic frame and
the real installed qlib classes; expected positions, labels and counts are
written out independently of the adapter.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from tests.stack import require

torch = require("torch")
handler_module = pytest.importorskip(
    "qlib.data.dataset.handler", reason="qlib not installed on this host"
)
DataHandler = handler_module.DataHandler
DataHandlerLP = handler_module.DataHandlerLP
pytest.importorskip("qlib.contrib.data.dataset", reason="qlib not installed on this host")
from qlib.contrib.data.dataset import MTSDatasetH  # noqa: E402

from ashare_lab.research.sequence_dataset import (  # noqa: E402
    ContainedMTSDatasetH,
    SegmentAdmission,
)

DATES = pd.bdate_range("2020-03-02", periods=9)
LABEL = ("label", "LABEL0")


def frame(sessions_by_name, dates=DATES):
    """Build a handler frame indexed by datetime, instrument with vector features."""
    pairs = [(dates[session], name) for name, sessions in sessions_by_name.items() for session in sessions]
    index = pd.MultiIndex.from_tuples(pairs, names=["datetime", "instrument"])
    built = []
    for date, name in pairs:
        session = int(dates.get_loc(date))
        base = 100.0 * list(sessions_by_name).index(name)
        built.append([base + session, 10.0 + session, 1.0 + session])
    columns = pd.MultiIndex.from_tuples([("feature", "f0"), ("feature", "f1"), LABEL])
    return pd.DataFrame(built, index=index, columns=columns, dtype=np.float32).sort_index()


def raw_labels(sessions_by_name, dates=DATES):
    pairs = [(name, dates[session]) for name, sessions in sessions_by_name.items() for session in sessions]
    index = pd.MultiIndex.from_tuples(pairs, names=["instrument", "datetime"])
    values = [1.0 + dates.get_loc(date) for _, date in pairs]
    return pd.Series(values, index=index, dtype="float64")


def sessions(count, name="A", start=0):
    return {name: list(range(start, start + count))}


def admission(start, end, purpose="retrospective", cutoff=None, dates=DATES):
    start_at = dates[start] if isinstance(start, int) else start
    end_at = dates[end] if isinstance(end, int) else end
    if cutoff is None:
        cutoff_at = end_at
    else:
        cutoff_at = dates[cutoff] if isinstance(cutoff, int) else cutoff
    return SegmentAdmission(pd.Timestamp(start_at), pd.Timestamp(end_at), purpose, pd.Timestamp(cutoff_at))


def build(sessions_by_name, segments, *, dates=DATES, calendar=None, labels=None, seq_len=3, num_states=2,
          horizon=2, batch_size=-1, shuffle=False, drop_last=False, handler=None):
    handler = handler if handler is not None else DataHandlerLP.from_df(frame(sessions_by_name, dates))
    return ContainedMTSDatasetH(
        handler,
        segments,
        calendar=DATES if calendar is None else calendar,
        raw_labels=raw_labels(sessions_by_name, dates) if labels is None else labels,
        seq_len=seq_len,
        num_states=num_states,
        horizon=horizon,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def handler_state(handler):
    """Everything a caller hands over, in a form an in-place edit cannot hide."""
    learned = handler._learn
    return (
        list(learned.index),
        learned.to_numpy().tobytes(),
        handler._data.to_numpy().tobytes(),
        handler._data.index.tolist(),
        None if getattr(handler, "_infer", None) is None else handler._infer.to_numpy().tobytes(),
    )


def memory_bytes(obj):
    return obj._memory.tobytes()


def view_snapshot(obj):
    return (
        memory_bytes(obj),
        obj._data.tobytes(),
        obj._label.tobytes(),
        list(obj.params),
        (obj.batch_size, obj.n_samples, obj.drop_last, obj.shuffle),
        [item.stop - 1 for item in obj._batch_slices],
        list(obj._daily_index.values),
        sorted(int(value) for value in obj._admitted),
        sorted(int(value) for value in obj._write_permitted),
        id(obj._batch_slices),
        id(obj._daily_slices),
        id(obj._daily_index),
    )


@pytest.fixture
def setup_counter(monkeypatch):
    counter = []
    original = MTSDatasetH.setup_data

    def recorder(self, *args, **kwargs):
        counter.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MTSDatasetH, "setup_data", recorder)
    return counter


@pytest.mark.parametrize("rows", [0, 1])
def test_refuses_unsupported_master_size_before_upstream(setup_counter, rows):
    sessions_by_name = sessions(4)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    handler._learn = handler._learn.iloc[:rows]
    before = handler_state(handler)
    with pytest.raises(ValueError):
        build(sessions_by_name, {"train": admission(0, 1, "train")}, handler=handler)
    assert setup_counter == []
    assert len(handler._learn) == rows
    assert handler_state(handler) == before


@pytest.mark.parametrize("rows", [2, 7])
def test_two_and_seven_row_masters_reach_real_setup(setup_counter, rows):
    handler = DataHandlerLP.from_df(frame(sessions(rows)))
    before = handler_state(handler)
    root = build(sessions(rows), {"train": admission(0, rows - 1, "train")}, handler=handler)
    assert handler_state(handler) == before
    assert setup_counter == [1]
    assert root._data.shape == (rows, 2)
    assert root._label.shape == (rows,)
    assert root._memory.shape == (rows, 2)
    assert root._data.dtype == root._label.dtype == root._memory.dtype == np.float32
    assert list(root._index) == [("A", DATES[i]) for i in range(rows)]
    assert root._data[:, 0].tolist() == [float(i) for i in range(rows)]
    assert root._label.tolist() == [float(i + 1) for i in range(rows)]
    expected_stops = [(max(stop - 3, 0), stop) for stop in range(1, rows + 1)]
    assert [(item.start, item.stop) for item in root._master_batch_slices] == expected_stops


def test_master_guard_counts_learned_rows_not_a_superset(setup_counter):
    sessions_by_name = sessions(5)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    handler._learn = handler._learn.iloc[:2]
    root = build(sessions_by_name, {"train": admission(0, 1, "train")}, handler=handler)
    assert setup_counter == [1]
    assert len(root._index) == 2
    handler._learn = handler._learn.iloc[:1]
    with pytest.raises(ValueError):
        build(sessions_by_name, {"train": admission(0, 1, "train")}, handler=handler)
    assert setup_counter == [1]


def test_single_row_instrument_does_not_refuse_global_master(setup_counter):
    root = build({"A": [0], "B": [0, 1, 2, 3, 4, 5]}, {"train": admission(0, 5, "train")})
    assert setup_counter == [1]
    assert len(root._index) == 7
    assert list(root._index.get_level_values(0)).count("A") == 1


def test_seven_row_master_yields_one_target_view(setup_counter):
    root = build(sessions(7), {"window": admission(2, 2, cutoff=4)}, batch_size=1)
    view = root.prepare("window")
    assert isinstance(view, ContainedMTSDatasetH)
    assert view._diagnostics == {
        "input_in_range": 1,
        "insufficient_own_history": 0,
        "calendar_gap": 0,
        "label-excluded": 0,
        "admitted": 1,
        "raw_extra_keys": 0,
    }
    assert len(view) == 1
    batch = next(iter(view))
    assert batch["index"].tolist() == [2]
    assert batch["data"].shape == (1, 3, 2)
    assert batch["data"][0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert batch["label"].tolist() == [3.0]
    view.eval()
    daily = list(view)
    assert len(daily) == 1
    assert daily[0]["index"].tolist() == [2]
    assert daily[0]["daily_count"].tolist() == [1]
    assert daily[0]["daily_index"].tolist() == [2]


def test_root_is_a_preparation_container(setup_counter):
    root = build(sessions(7), {"window": admission(2, 4, cutoff=6)})
    with pytest.raises(ValueError):
        len(root)
    with pytest.raises(ValueError):
        next(iter(root))
    with pytest.raises(ValueError):
        root.assign_data(np.array([2]), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        root.prepare(slice(DATES[2], DATES[4]))
    with pytest.raises(ValueError):
        root.prepare((DATES[2], DATES[4]))
    with pytest.raises(ValueError):
        root.setup_data()
    with pytest.raises(ValueError):
        root.config()
    assert root._bound is False


def test_named_selection_and_selector_refusals(setup_counter):
    root = build(sessions(8), {"first": admission(2, 4, cutoff=6), "second": admission(5, 7, cutoff=8)})
    views = root.prepare(["first", "second"])
    assert [view._admission_name for view in views] == ["first", "second"]
    collected = [[int(value) for batch in view for value in batch["index"]] for view in views]
    assert collected == [[2, 3, 4], [5, 6]]
    assert views[1]._diagnostics["label-excluded"] == 1
    single = root.prepare(["first"])
    assert isinstance(single, list) and len(single) == 1 and isinstance(single[0], ContainedMTSDatasetH)
    with pytest.raises(ValueError):
        root.prepare("missing")
    with pytest.raises(ValueError):
        root.prepare(["first", "first"])
    with pytest.raises(ValueError):
        root.prepare([])
    view = root.prepare("first")
    with pytest.raises(ValueError):
        view.prepare("second")
    with pytest.raises(ValueError):
        view.prepare(slice(DATES[1], DATES[3]))
    with pytest.raises(ValueError):
        view.prepare(slice(DATES[3], DATES[6]))
    with pytest.raises(ValueError):
        view.prepare(slice(DATES[2], DATES[4], 1))
    nested = view.prepare(slice(DATES[3], DATES[4]))
    assert [int(value) for batch in nested for value in batch["index"]] == [3, 4]
    assert nested._admission_name == "first"
    again = view.prepare("first")
    assert [int(value) for batch in again for value in batch["index"]] == [2, 3, 4]


def test_insufficient_own_history_is_excluded(setup_counter):
    root = build(sessions(5), {"window": admission(0, 4, cutoff=6)})
    view = root.prepare("window")
    assert [int(value) for batch in view for value in batch["index"]] == [2, 3, 4]
    assert view._diagnostics == {
        "input_in_range": 5,
        "insufficient_own_history": 2,
        "calendar_gap": 0,
        "label-excluded": 0,
        "admitted": 3,
        "raw_extra_keys": 0,
    }


def test_calendar_gap_after_sufficient_history_is_excluded(setup_counter):
    root = build({"A": [0, 1, 2, 4, 5]}, {"window": admission(0, 5, cutoff=7)})
    view = root.prepare("window")
    assert next(iter(view))["index"].tolist() == [2]
    assert view._diagnostics == {
        "input_in_range": 5,
        "insufficient_own_history": 2,
        "calendar_gap": 2,
        "label-excluded": 0,
        "admitted": 1,
        "raw_extra_keys": 0,
    }


def test_immature_endpoints_are_excluded_by_the_cutoff(setup_counter):
    root = build(sessions(7), {"window": admission(2, 4, cutoff=4)})
    view = root.prepare("window")
    assert next(iter(view))["index"].tolist() == [2]
    assert view._diagnostics == {
        "input_in_range": 3,
        "insufficient_own_history": 0,
        "calendar_gap": 0,
        "label-excluded": 2,
        "admitted": 1,
        "raw_extra_keys": 0,
    }


def test_nonfinite_raw_and_learned_labels_are_excluded(setup_counter):
    sessions_by_name = sessions(8)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    handler._learn.iloc[4, handler._learn.columns.get_loc(LABEL)] = np.nan
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[5])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
    view = root.prepare("window")
    assert [int(value) for batch in view for value in batch["index"]] == [2, 3, 6]
    assert view._diagnostics["label-excluded"] == 2


def test_zero_admitted_targets_refuse_preparation(setup_counter):
    root = build(sessions(7), {"window": admission(3, 4, cutoff=4)})
    with pytest.raises(ValueError, match="no-admitted-targets"):
        root.prepare("window")


def test_live_view_needs_no_future_label_and_never_writes(setup_counter):
    sessions_by_name = sessions(8)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    handler._learn.iloc[4, handler._learn.columns.get_loc(LABEL)] = np.nan
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[4])] = np.nan
    root = build(sessions_by_name, {"live": admission(4, 4, "live")}, labels=labels, handler=handler)
    view = root.prepare("live")
    assert next(iter(view))["index"].tolist() == [4]
    assert not np.isfinite(root._raw_values[4])
    before = memory_bytes(root)
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.ones((1, 2)))
    assert memory_bytes(root) == before


def test_live_with_finite_labels_still_refuses_writes(setup_counter):
    root = build(sessions(8), {"live": admission(4, 4, "live")})
    view = root.prepare("live")
    assert np.isfinite(root._raw_values[4]) and np.isfinite(root._label[4])
    before = memory_bytes(root)
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.ones((1, 2)))
    assert memory_bytes(root) == before


def test_live_cannot_share_a_root(setup_counter):
    with pytest.raises(ValueError):
        build(sessions(8), {"live": admission(4, 4, "live"), "train": admission(2, 4, "train")})


def test_copy_shares_master_arrays_and_owns_tables(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    clone = copy.copy(view)
    assert np.shares_memory(clone._data, view._data)
    assert np.shares_memory(clone._memory, view._memory)
    assert np.shares_memory(clone._label, view._label)
    assert clone._batch_slices is not view._batch_slices
    assert clone._daily_slices is not view._daily_slices
    assert clone._daily_index is not view._daily_index
    original = view.batch_size
    clone.eval()
    assert view.batch_size == original
    assert clone.batch_size == -1
    assert next(iter(clone))["index"].tolist() == [2]


def test_sample_eval_train_round_trip_restores_flags(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)}, batch_size=2, drop_last=False)
    view = root.prepare("window")
    assert len(view) == 3
    assert [batch["index"].size for batch in view] == [2, 2, 1]
    view.eval()
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (-1, None, False, False)
    daily = list(view)
    assert len(daily) == 5
    assert sorted(int(value) for batch in daily for value in batch["index"]) == [2, 3, 4, 5, 6]
    view.train()
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (2, None, False, False)
    assert [batch["index"].size for batch in view] == [2, 2, 1]
    assert list(view.params) == [2, None, False, False]


def test_drop_last_and_shuffle_emit_only_full_batches(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)}, batch_size=2, drop_last=True, shuffle=True)
    view = root.prepare("window")
    np.random.seed(0)
    batches = list(view)
    assert len(batches) == 2
    emitted = [int(value) for batch in batches for value in batch["index"]]
    assert len(emitted) == len(set(emitted)) == 4
    assert set(emitted) <= {2, 3, 4, 5, 6}
    assert view.drop_last is True and view.shuffle is True
    view.eval()
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (-1, None, False, False)
    view.train()
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (2, None, True, True)
    assert len(view) == 2


@pytest.mark.parametrize("batch_size", [3, -3])
def test_zero_executable_batches_refuse_preparation(setup_counter, batch_size):
    root = build(sessions(8), {"small": admission(4, 5, cutoff=8)}, batch_size=batch_size, drop_last=True)
    with pytest.raises(ValueError, match="no-executable-batches"):
        root.prepare("small")


def test_partial_batch_is_admitted_when_not_dropped(setup_counter):
    root = build(sessions(8), {"small": admission(4, 5, cutoff=8)}, batch_size=3, drop_last=False)
    view = root.prepare("small")
    assert len(view) == 1
    assert next(iter(view))["index"].tolist() == [4, 5]


def test_three_endpoints_fill_one_batch(setup_counter):
    root = build(sessions(9), {"third": admission(4, 6, cutoff=8)}, batch_size=3, drop_last=True)
    view = root.prepare("third")
    assert len(view) == 1
    assert next(iter(view))["index"].tolist() == [4, 5, 6]


@pytest.mark.parametrize("batch_size", [3, -3])
def test_mode_transition_refusal_preserves_the_view(setup_counter, batch_size):
    root = build(sessions(9), {"small": admission(4, 5, cutoff=8)}, batch_size=batch_size, drop_last=True)
    root.eval()
    view = root.prepare("small")
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (-1, None, False, False)
    assert len(view) == 2
    before = view_snapshot(view)
    with pytest.raises(ValueError, match="no-executable-batches"):
        view.train()
    assert view_snapshot(view) == before
    assert (view.batch_size, view.n_samples, view.drop_last, view.shuffle) == (-1, None, False, False)
    assert list(view.params)[0] == batch_size
    assert [batch["index"].tolist() for batch in view] == [[4], [5]]
    view.assign_data(np.array([4]), np.ones((1, 2)))
    assert view._memory[4].tolist() == [1.0, 1.0]


def test_guarded_write_matches_global_positions(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    view.assign_data(np.array([2, 5]), np.array([[1.5, 2.5], [3.0, 4.0]]))
    assert root._memory[2].tolist() == [1.5, 2.5]
    assert root._memory[5].tolist() == [3.0, 4.0]
    assert root._memory[3].tolist() == [0.0, 0.0]
    view.assign_data(torch.tensor([6]), torch.tensor([[7.25, 8.5]], dtype=torch.float64))
    assert root._memory[6].tolist() == [7.25, 8.5]
    assert view._memory.dtype == np.float32


def test_write_refusals_preserve_the_full_memory(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    view.assign_data(np.array([3]), np.array([[1.0, 2.0]]))
    before = memory_bytes(root)
    rejects = [
        (np.array([0]), np.ones((1, 2))),
        (np.array([2, 2]), np.ones((2, 2))),
        (np.array([2, 7]), np.ones((2, 2))),
        (np.array([8]), np.ones((1, 2))),
        (np.array([True]), np.ones((1, 2))),
        (np.array([2]), np.ones((2, 2))),
        (np.array([2, 3]), np.ones((1, 2))),
        (np.array([2, 3]), np.ones((2, 1))),
        (np.array([2]), np.array([[np.nan, 1.0]])),
        (np.array([2]), np.array([[np.inf, 1.0]])),
        (np.array([2]), np.array([[1e300, 1.0]])),
        (np.array([2]), np.array([[1 + 2j, 1.0]])),
        (np.array([2]), np.array([[True, False]])),
        (np.array([2]), np.array([["a", "b"]])),
        (np.array([2]), np.array([[1.0, 2.0]], dtype=object)),
    ]
    for bad_index, bad_values in rejects:
        with pytest.raises(ValueError):
            view.assign_data(bad_index, bad_values)
        assert memory_bytes(root) == before


def test_float64_values_round_to_the_storage_dtype(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    view.assign_data(np.array([4]), np.array([[0.1, 1e-45]]))
    assert root._memory[4].tolist() == np.array([[0.1, 1e-45]], dtype="float64").astype("float32")[0].tolist()


def test_clear_memory_is_shared_across_views(setup_counter):
    root = build(sessions(8), {"first": admission(2, 4, cutoff=8), "second": admission(5, 7, cutoff=8)})
    one = root.prepare("first")
    two = root.prepare("second")
    one.assign_data(np.array([2]), np.ones((1, 2)))
    two.assign_data(np.array([5]), np.ones((1, 2)))
    assert root._memory[2].tolist() == [1.0, 1.0] and root._memory[5].tolist() == [1.0, 1.0]
    one.clear_memory()
    assert not root._memory.any()


def test_serialization_and_reconfiguration_are_refused(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    with pytest.raises(TypeError):
        view.__getstate__()
    with pytest.raises(TypeError):
        pickle.dumps(view)
    with pytest.raises(TypeError):
        copy.deepcopy(view)
    with pytest.raises(ValueError):
        view.config()
    with pytest.raises(ValueError):
        view.setup_data()


def test_caller_containers_cannot_change_the_adapter(setup_counter):
    segments = {"window": admission(2, 6, cutoff=8)}
    buffer = DATES.to_numpy().copy()
    calendar = pd.DatetimeIndex(buffer)
    labels = raw_labels(sessions(8))
    root = build(sessions(8), segments, calendar=calendar, labels=labels)
    segments["window"] = admission(0, 1, cutoff=8)
    buffer[:] = buffer[0]
    labels.iloc[:] = 999.0
    assert root._admissions["window"].start == DATES[2]
    assert root._calendar.tolist() == DATES.tolist()
    assert root._raw_values[0] == 1.0
    view = root.prepare("window")
    assert [int(value) for batch in view for value in batch["index"]] == [2, 3, 4, 5, 6]


def test_admission_and_setting_validation(setup_counter):
    with pytest.raises(ValueError):
        build(sessions(8), {})
    with pytest.raises(ValueError):
        build(sessions(8), {"": admission(2, 4, cutoff=6)})
    with pytest.raises(ValueError):
        build(sessions(8), {"train": admission(4, 2, "train")})
    with pytest.raises(ValueError):
        build(sessions(8), {"train": admission(2, 4, "train", cutoff=6)})
    with pytest.raises(ValueError):
        build(sessions(8), {"live": admission(2, 4, "live")})
    with pytest.raises(ValueError):
        build(sessions(8), {"retro": admission(2, 6, cutoff=pd.Timestamp("2030-01-01"))})
    with pytest.raises(ValueError):
        build(sessions(8), {"retro": admission(2, 6, cutoff=6)}, seq_len=2)
    with pytest.raises(ValueError):
        build(sessions(8), {"retro": admission(2, 6, cutoff=6)}, batch_size=0)
    with pytest.raises(ValueError):
        build(sessions(8), {"retro": admission(2, 6, cutoff=6)}, num_states=0)
    with pytest.raises(ValueError):
        build(sessions(8), {"retro": admission(2, 6, cutoff=6)}, seq_len=True)


def test_segment_admission_is_frozen():
    value = admission(2, 4, cutoff=6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.start = DATES[0]

def test_prepare_seg_routes_through_admission(setup_counter):
    sessions_by_name = sessions(9)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[6])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
    view = root.prepare("window")
    with pytest.raises(ValueError):
        root._prepare_seg((DATES[2], DATES[6]))
    nested = view._prepare_seg((DATES[5], DATES[6]))
    assert [int(value) for batch in nested for value in batch["index"]] == [5]
    assert nested._admission is view._admission
    sliced = view._prepare_seg(slice(DATES[3], DATES[5]))
    assert [int(value) for batch in sliced for value in batch["index"]] == [3, 4, 5]
    with pytest.raises(ValueError):
        view._prepare_seg(slice(DATES[3], DATES[5], 1))
    with pytest.raises(ValueError, match="nested range"):
        view._prepare_seg((DATES[1], DATES[5]))


def test_nested_empty_target_refuses_late_and_preserves_parents(setup_counter):
    sessions_by_name = sessions(9)
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[6])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8), "sibling": admission(3, 5, cutoff=8)},
                 labels=labels, handler=handler)
    view = root.prepare("window")
    sibling = root.prepare("sibling")
    before = [view_snapshot(root), view_snapshot(view), view_snapshot(sibling)]
    with pytest.raises(ValueError, match="no-admitted-targets"):
        view._prepare_seg((DATES[6], DATES[6]))
    assert [view_snapshot(root), view_snapshot(view), view_snapshot(sibling)] == before
    assert [int(value) for batch in view for value in batch["index"]] == [2, 3, 4, 5]
    view.assign_data(np.array([5]), np.ones((1, 2)))
    assert view._memory[5].tolist() == [1.0, 1.0]


def test_zero_batch_refusal_leaves_root_and_sibling_untouched(setup_counter):
    root = build(sessions(9), {"small": admission(4, 5, cutoff=8), "big": admission(2, 6, cutoff=8)},
                 batch_size=3, drop_last=True)
    sibling = root.prepare("big")
    before = [view_snapshot(root), view_snapshot(sibling)]
    with pytest.raises(ValueError, match="no-executable-batches"):
        root.prepare("small")
    assert [view_snapshot(root), view_snapshot(sibling)] == before
    assert len(sibling) == 1
    sibling.assign_data(np.array([4]), np.ones((1, 2)))
    assert sibling._memory[4].tolist() == [1.0, 1.0]


def test_missing_raw_label_key_is_an_error(setup_counter):
    labels = raw_labels(sessions(8)).drop(("A", DATES[3]))
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labels)


def test_non_midnight_range_is_refused(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    shifted = DATES[3] + pd.Timedelta(hours=9)
    with pytest.raises(ValueError, match="midnight"):
        view.prepare(slice(shifted, DATES[5]))


def test_unknown_purpose_is_refused(setup_counter):
    with pytest.raises(ValueError, match="purpose"):
        build(sessions(9), {"window": SegmentAdmission(DATES[2], DATES[6], "validaton", DATES[6])})


def test_constructor_type_and_shape_checks_are_pinned(setup_counter):
    with pytest.raises(ValueError, match="positive integer"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, num_states=True)
    with pytest.raises(ValueError, match="nonzero integer"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, batch_size=2.5)
    with pytest.raises(ValueError, match="booleans"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, shuffle=1)
    with pytest.raises(ValueError, match="DataHandlerLP"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, handler=frame(sessions(8)))


def test_unknown_name_and_reversed_range_are_refused(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    with pytest.raises(ValueError, match="unknown segment name"):
        root.prepare(["window", "missing"])
    view = root.prepare("window")
    with pytest.raises(ValueError, match="follow its end"):
        view._prepare_seg(slice(DATES[5], DATES[3]))


def test_unsupported_horizon_is_refused(setup_counter):
    for horizon in (1, 3):
        with pytest.raises(ValueError):
            build(sessions(8), {"window": admission(2, 6, cutoff=8)}, horizon=horizon)


def test_non_default_fetch_settings_are_refused(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    with pytest.raises(ValueError):
        root.prepare("window", data_key=DataHandlerLP.DK_L)
    with pytest.raises(ValueError):
        root.prepare("window", col_set=DataHandler.CS_RAW)
    with pytest.raises(ValueError):
        root.prepare("window", extra_argument=1)


def test_write_controls_hold_under_raising_error_state(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    baseline = np.geterr().copy()
    with np.errstate(over="raise"):
        view.assign_data(np.array([4]), np.array([[1.0, 2.0]]))
        with pytest.raises(ValueError):
            view.assign_data(np.array([4]), np.array([[1e300, 0.0]]))
        assert np.geterr()["over"] == "raise"
    assert np.geterr() == baseline
    assert root._memory[4].tolist() == [1.0, 2.0]


def test_integer_values_are_accepted(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    view.assign_data(np.array([4]), np.array([[3, 4]], dtype=np.int64))
    assert root._memory[4].tolist() == [3.0, 4.0]


def test_state_visibility_follows_the_upstream_offset(setup_counter):
    dates = pd.bdate_range("2021-01-04", periods=12)
    sessions_by_name = {"A": list(range(11))}
    root = build(
        sessions_by_name,
        {"window": admission(3, 9, cutoff=11, dates=dates)},
        dates=dates,
        calendar=dates,
        labels=raw_labels(sessions_by_name, dates),
        seq_len=4,
        batch_size=1,
    )
    view = root.prepare("window")
    view.assign_data(np.array([5]), np.array([[1.5, -2.5]]))
    states = {int(batch["index"][0]): batch["state"][0].tolist() for batch in view}
    assert states[6] == [[0.0, 0.0], [0.0, 0.0]]
    assert states[7] == [[0.0, 0.0], [1.5, -2.5]]
    assert states[8] == [[1.5, -2.5], [0.0, 0.0]]

def test_multi_column_label_master_is_refused(setup_counter):
    table = frame(sessions(8))
    table[("label", "LABEL1")] = table[LABEL].to_numpy()
    handler = DataHandlerLP.from_df(table)
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, handler=handler)


def test_two_instrument_daily_groups_keep_global_ids(setup_counter):
    sessions_by_name = {"A": list(range(7)), "B": [0, 1, 2, 4, 5, 6]}
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[4])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
    view = root.prepare("window")
    batches = list(view)
    assert [batch["daily_index"].tolist() for batch in batches] == [[2], [3], [5], [6]]
    assert [int(batch["daily_count"][0]) for batch in batches] == [2, 1, 1, 2]
    assert batches[0]["index"].tolist() == [2, 9]
    last = batches[-1]
    assert last["index"].tolist() == [6, 12]
    assert last["data"][1][:, 0].tolist() == [104.0, 105.0, 106.0]
    assert last["data"][0][:, 0].tolist() == [4.0, 5.0, 6.0]


def test_extra_raw_keys_are_reported_not_used(setup_counter):
    labels = raw_labels(sessions(8))
    extra_index = pd.MultiIndex.from_tuples([("Z", DATES[0])], names=["instrument", "datetime"])
    labels = pd.concat([labels, pd.Series([1.0], index=extra_index, dtype="float64")])
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labels)
    assert root._raw_extra_keys == 1
    assert len(root._index) == 8


def test_equal_ranges_resolve_by_name_and_policy(setup_counter):
    root = build(
        sessions(8),
        {"train": admission(2, 4, "train"), "retro": admission(2, 4, "retrospective", cutoff=6)},
    )
    trained = root.prepare("train")
    retro = root.prepare("retro")
    assert [int(value) for batch in trained for value in batch["index"]] == [2]
    assert [int(value) for batch in retro for value in batch["index"]] == [2, 3, 4]
    assert trained._admission.purpose == "train" and retro._admission.purpose == "retrospective"


def test_independent_roots_do_not_share_memory(setup_counter):
    first = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    second = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    first.prepare("window").assign_data(np.array([3]), np.ones((1, 2)))
    assert first._memory[3].tolist() == [1.0, 1.0]
    assert not second._memory.any()


def test_timestamp_domain_is_validated(setup_counter):
    aware = DATES.tz_localize("UTC")
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, calendar=aware)
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, calendar=DATES + pd.Timedelta(hours=9))
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)},
              calendar=pd.DatetimeIndex([DATES[0], pd.NaT] + list(DATES[1:])))
    with pytest.raises(ValueError):
        build(sessions(8), {"window": SegmentAdmission(aware[2], aware[4], "retrospective", aware[6])})
    assert setup_counter == []


def test_value_dtypes_are_validated(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    masked = np.ma.array([[np.nan, 1.0]], mask=[[True, False]])
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), masked)
    class _Subclass(np.ndarray):
        pass

    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.ones((1, 2)).view(_Subclass))
    with pytest.raises(ValueError):
        view.assign_data(np.ma.array([4], mask=[True]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        view.assign_data(np.array([4.0]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.array([["2020-03-02", "2020-03-03"]], dtype="datetime64[D]"))
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.array([[1, 2]], dtype="timedelta64[D]"))
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), np.array([[1.5, 2.5]], dtype=[("a", "f4"), ("b", "f4")]))

def test_underflow_is_tolerated_under_a_raising_caller_state(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    baseline = np.geterr().copy()
    with np.errstate(under="raise"):
        view.assign_data(np.array([4]), np.array([[1e-45, 0.0]]))
    assert np.geterr() == baseline
    assert root._memory[4].tolist() == [np.float32(1e-45).item(), 0.0]


def test_refusal_detail_names_the_mode_and_shape(setup_counter):
    for batch_size, mode in ((3, "sample"), (-3, "daily")):
        root = build(sessions(8), {"small": admission(4, 5, cutoff=8)}, batch_size=batch_size, drop_last=True)
        with pytest.raises(ValueError) as caught:
            root.prepare("small")
        message = str(caught.value)
        assert f"mode={mode}" in message
        assert "batch_size=3" in message and "drop_last=True" in message and "units=" in message


def test_permission_array_is_not_the_admission_array(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    assert not np.shares_memory(view._admitted, view._write_permitted)
    assert view._admitted.tolist() == [2, 3, 4, 5, 6]
    assert view._write_permitted.tolist() == [2, 3, 4, 5, 6]
    clone = copy.copy(view)
    assert not np.shares_memory(clone._admitted, clone._write_permitted)
    view._write_permitted[0] = -1
    assert 2 in view._admitted.tolist()


@pytest.mark.parametrize("dtype", ["bool", "complex128", "object", "string"])
def test_non_real_raw_label_dtypes_are_refused(setup_counter, dtype):
    labels = raw_labels(sessions(8)).astype(dtype)
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labels)
    assert setup_counter == []


def test_unsorted_or_duplicated_calendar_is_refused(setup_counter):
    duplicated = pd.DatetimeIndex([DATES[0], DATES[1], DATES[1]] + list(DATES[2:]))
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, calendar=duplicated)
    unsorted = pd.DatetimeIndex(list(DATES[::-1]))
    with pytest.raises(ValueError):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, calendar=unsorted)
    assert setup_counter == []


def test_named_repreparation_keeps_the_parent_window(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    narrowed = root.prepare("window")._prepare_seg(slice(DATES[3], DATES[5]))
    assert [int(value) for batch in narrowed for value in batch["index"]] == [3, 4, 5]
    reopened = narrowed.prepare("window")
    assert reopened._window == (DATES[3], DATES[5])
    assert [int(value) for batch in reopened for value in batch["index"]] == [3, 4, 5]
    assert reopened._write_permitted.tolist() == [3, 4, 5]


def test_unsupported_torch_dtype_is_a_value_error(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    with pytest.raises(ValueError):
        view.assign_data(np.array([4]), torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        view.assign_data(torch.tensor([4], dtype=torch.bfloat16), np.ones((1, 2)))


def test_nested_child_cannot_write_outside_its_window(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    child = view._prepare_seg(slice(DATES[4], DATES[5]))
    assert [int(value) for batch in child for value in batch["index"]] == [4, 5]
    assert child._write_permitted.tolist() == [4, 5]
    with pytest.raises(ValueError):
        child.assign_data(np.array([6]), np.ones((1, 2)))
    child.assign_data(np.array([5]), np.ones((1, 2)))
    assert root._memory[5].tolist() == [1.0, 1.0]


def test_copy_preserves_a_dynamic_subclass(setup_counter):
    class Tagged(ContainedMTSDatasetH):
        pass

    handler = DataHandlerLP.from_df(frame(sessions(8)))
    root = Tagged(handler, {"window": admission(2, 6, cutoff=8)}, calendar=DATES,
                  raw_labels=raw_labels(sessions(8)), seq_len=3, num_states=2, shuffle=False)
    root.marker = "dynamic"
    view = root.prepare("window")
    clone = copy.copy(view)
    assert type(clone) is Tagged
    assert clone.marker == "dynamic"
    assert [int(value) for batch in clone for value in batch["index"]] == [2, 3, 4, 5, 6]


def test_validation_purpose_excludes_unmatured_endpoints(setup_counter):
    root = build(sessions(9), {"validation": admission(2, 6, "validation")})
    view = root.prepare("validation")
    assert view._admission.purpose == "validation"
    assert [int(value) for batch in view for value in batch["index"]] == [2, 3, 4]


def test_seven_row_daily_batch_carries_its_literal_history(setup_counter):
    root = build(sessions(7), {"window": admission(2, 6, cutoff=6)})
    view = root.prepare("window")
    batches = list(view)
    assert [batch["index"].tolist() for batch in batches] == [[2], [3], [4]]
    first = batches[0]
    assert first["data"][0][:, 0].tolist() == [0.0, 1.0, 2.0]
    assert first["label"].ravel().tolist() == [3.0]


def test_diagnostics_report_the_partition_and_extra_keys(setup_counter):
    labels = raw_labels(sessions(8))
    extra_index = pd.MultiIndex.from_tuples([("Z", DATES[0])], names=["instrument", "datetime"])
    labels = pd.concat([labels, pd.Series([1.0], index=extra_index, dtype="float64")])
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labels)
    view = root.prepare("window")
    diagnostics = view._diagnostics
    assert diagnostics["raw_extra_keys"] == 1
    assert diagnostics["admitted"] == len(view._admitted) == 5
    assert diagnostics["input_in_range"] == 5


def test_mode_transition_leaves_root_and_sibling_untouched(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8), "sibling": admission(3, 5, cutoff=8)},
                 batch_size=2)
    view = root.prepare("window")
    sibling = root.prepare("sibling")
    before = [view_snapshot(root), view_snapshot(sibling)]
    view.eval()
    view.train()
    assert [view_snapshot(root), view_snapshot(sibling)] == before


def test_daily_ids_are_neither_detected_nor_reinterpreted(setup_counter):
    sessions_by_name = {"A": list(range(7)), "B": [0, 1, 2, 4, 5, 6]}
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[4])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
    view = root.prepare("window")
    batch = next(iter(view))
    assert batch["daily_index"].tolist() == [2]
    view.assign_data(batch["daily_index"], np.ones((1, 2)))
    assert root._memory[2].tolist() == [1.0, 1.0]
    assert not root._memory[9].any()
    view.assign_data(np.array([5]), np.full((1, 2), 2.0))
    assert root._memory[5].tolist() == [2.0, 2.0]
    assert not root._memory[12].any()


def test_sample_and_daily_endpoints_reconcile(setup_counter):
    # Literal expectations, derived once from the eligibility core plus the
    # admission window, never from the adapter's own mask.
    fixtures = (
        (sessions(9), [2, 3, 4, 5, 6], [1, 1, 1, 1, 1]),
        ({"A": list(range(7)), "B": [0, 1, 2, 4, 5, 6]}, [2, 3, 4, 5, 6, 9, 12], [2, 1, 1, 1, 2]),
        ({"A": list(range(2, 7)), "B": list(range(7))}, [2, 3, 4, 7, 8, 9, 10, 11], [1, 1, 2, 2, 2]),
    )
    for sessions_by_name, expected_rows, expected_counts in fixtures:
        handler = DataHandlerLP.from_df(frame(sessions_by_name))
        labels = raw_labels(sessions_by_name)
        root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
        view = root.prepare("window")
        assert sorted(int(position) for position in view._admitted) == expected_rows
        sample_rows = [int(position) for batch in view for position in batch["index"]]
        assert sorted(sample_rows) == expected_rows
        daily_rows = [int(position) for batch in view for position in batch["daily_index"]]
        assert len(daily_rows) == len(expected_counts)
        assert [int(count) for batch in view for count in batch["daily_count"]] == expected_counts
        sample_dates = [root._index[int(position)][1] for position in sample_rows]
        daily_dates = [
            pd.Timestamp(value)
            for batch in view
            for value in view.restore_daily_index(batch["daily_index"])
        ]
        assert sorted(set(sample_dates)) == sorted(set(daily_dates))
        assert len(sample_dates) == len(expected_rows)
        assert len(set(daily_dates)) == len(expected_counts)
        sample_root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels,
                            handler=handler, batch_size=2, drop_last=False)
        sample_view = sample_root.prepare("window")
        in_batches = [int(position) for batch in sample_view for position in batch["index"]]
        assert sorted(in_batches) == expected_rows


def test_raw_label_index_shape_is_validated(setup_counter):
    labels = raw_labels(sessions(8))
    flat = pd.Series(labels.to_numpy(), index=pd.RangeIndex(len(labels)))
    with pytest.raises(ValueError, match="MultiIndex"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=flat)
    duplicated = pd.concat([labels, labels.iloc[[0]]])
    with pytest.raises(ValueError, match="raw_labels index must be unique"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=duplicated)


def test_finiteness_refusal_names_the_condition(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    with pytest.raises(ValueError, match="finite"):
        view.assign_data(np.array([4]), np.array([[np.nan, 1.0]]))


def test_widening_past_a_narrow_parent_is_refused(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    child = view._prepare_seg(slice(DATES[4], DATES[5]))
    assert child._window == (DATES[4], DATES[5])
    with pytest.raises(ValueError, match="parent"):
        child.prepare(slice(DATES[3], DATES[5]))


def test_raw_label_datetime_domain_is_validated(setup_counter):
    def labelled(dates):
        index = pd.MultiIndex.from_tuples([("A", date) for date in dates], names=["instrument", "datetime"])
        return pd.Series([1.0] * len(dates), index=index, dtype="float64")

    shifted = [date + pd.Timedelta(hours=9) for date in DATES[:8]]
    with pytest.raises(ValueError, match="midnight"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labelled(shifted))
    aware = [date.tz_localize("UTC") for date in DATES[:8]]
    with pytest.raises(ValueError, match="timezone-naive"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labelled(aware))
    blanked = list(DATES[:8])
    blanked[3] = pd.NaT
    with pytest.raises(ValueError, match="timezone-naive"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labelled(blanked))


def test_blank_instrument_name_in_raw_labels_is_refused(setup_counter):
    labels = raw_labels(sessions(8)).drop(("A", DATES[0]))
    extra = pd.Series([1.0], index=pd.MultiIndex.from_tuples([("", DATES[0])], names=["instrument", "datetime"]),
                      dtype="float64")
    with pytest.raises(ValueError, match="nonempty"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=pd.concat([labels, extra]))


def test_live_view_reports_no_label_exclusion(setup_counter):
    sessions_by_name = sessions(9)
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[8])] = np.nan
    root = build(sessions_by_name, {"live": admission(8, 8, "live")}, labels=labels)
    view = root.prepare("live")
    assert view._diagnostics["label-excluded"] == 0
    assert view._diagnostics["admitted"] == len(view._admitted) == 1
    assert [int(value) for batch in view for value in batch["index"]] == [8]


def test_handler_without_a_learned_frame_is_refused(setup_counter):
    handler = DataHandlerLP.from_df(frame(sessions(8)))
    handler._learn = None
    with pytest.raises(ValueError, match="retained learned representation"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, handler=handler)


def test_named_repreparation_does_not_widen_write_permission(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    narrowed = view._prepare_seg(slice(DATES[4], DATES[5]))
    assert narrowed._write_permitted.tolist() == [4, 5]
    reopened = narrowed.prepare("window")
    assert reopened._window == (DATES[4], DATES[5])
    assert reopened._write_permitted.tolist() == [4, 5]
    reopened.assign_data(np.array([4]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        reopened.assign_data(np.array([2]), np.ones((1, 2)))
    grandchild = reopened._prepare_seg(slice(DATES[5], DATES[5]))
    assert grandchild._write_permitted.tolist() == [5]


def test_copy_owns_its_diagnostics(setup_counter):
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    clone = copy.copy(view)
    clone._diagnostics["admitted"] = -1
    assert view._diagnostics["admitted"] == 5


def test_nullable_label_column_is_read_as_a_float_series(setup_counter):
    labels = raw_labels(sessions(8)).astype("Float64")
    labels.iloc[3] = pd.NA
    root = build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=labels)
    view = root.prepare("window")
    assert np.isnan(view._raw_values).sum() == 1


def test_root_named_preparation_still_covers_the_whole_admission(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    assert view._window == (DATES[2], DATES[6])
    assert view._write_permitted.tolist() == [2, 3, 4, 5, 6]


def test_a_copy_keeps_the_window_it_was_narrowed_to(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    narrowed = root.prepare("window")._prepare_seg(slice(DATES[3], DATES[5]))
    clone = copy.copy(narrowed)
    reopened = clone.prepare("window")
    assert reopened._window == (DATES[3], DATES[5])
    assert [int(value) for batch in reopened for value in batch["index"]] == [3, 4, 5]
    with pytest.raises(ValueError):
        reopened.assign_data(np.array([2]), np.ones((1, 2)))


def test_a_bound_view_accepts_its_own_name_in_a_one_item_list(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    narrowed = root.prepare("window")._prepare_seg(slice(DATES[3], DATES[5]))
    for selector in (["window"], ("window",)):
        reopened = narrowed.prepare(selector)
        assert isinstance(reopened, list) and len(reopened) == 1
        assert reopened[0]._window == (DATES[3], DATES[5])
        assert reopened[0]._write_permitted.tolist() == [3, 4, 5]


def test_a_bound_view_refuses_a_containing_sibling(setup_counter):
    root = build(sessions(9), {"inner": admission(3, 5, cutoff=6), "outer": admission(2, 6, cutoff=6)})
    view = root.prepare("inner")
    with pytest.raises(ValueError, match="own admission"):
        view.prepare("outer")
    with pytest.raises(ValueError, match="own admission"):
        view.prepare(["inner", "outer"])


def test_daily_batches_of_three_dates_carry_two_entries(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)}, batch_size=-3)
    view = root.prepare("window")
    batches = list(view)
    assert len(view) == 2 and len(batches) == 2
    assert [int(count) for count in batches[0]["daily_count"]] == [1, 1, 1]
    assert batches[0]["daily_index"].tolist() == [2, 3, 4]
    assert batches[1]["daily_index"].tolist() == [5, 6]
    assert all(sum(int(count) for count in batch["daily_count"]) == len(batch["index"]) for batch in batches)
    assert sum(int(count) for batch in batches for count in batch["daily_count"]) == 5


def test_multi_instrument_daily_batch_is_writable(setup_counter):
    sessions_by_name = {"A": list(range(7)), "B": [0, 1, 2, 4, 5, 6]}
    handler = DataHandlerLP.from_df(frame(sessions_by_name))
    labels = raw_labels(sessions_by_name)
    labels.loc[("A", DATES[4])] = np.nan
    root = build(sessions_by_name, {"window": admission(2, 6, cutoff=8)}, labels=labels, handler=handler)
    view = root.prepare("window")
    batch = next(iter(view))
    assert batch["index"].tolist() == [2, 9]
    view.assign_data(batch["index"], np.ones((2, 2)))
    assert root._memory[2].tolist() == [1.0, 1.0]
    assert root._memory[9].tolist() == [1.0, 1.0]


def test_one_date_master_is_refused_before_upstream(setup_counter):
    dates = DATES[:1]
    with pytest.raises(ValueError, match="one-row learned master"):
        build({"A": [0]}, {"window": admission(0, 0, cutoff=0, dates=dates)}, dates=dates, calendar=dates,
              labels=raw_labels({"A": [0]}, dates))
    assert setup_counter == []


def test_empty_label_series_is_refused(setup_counter):
    empty = pd.Series([], index=pd.MultiIndex.from_arrays([[], []], names=["instrument", "datetime"]),
                      dtype="float64")
    with pytest.raises(ValueError, match="must not be empty"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, labels=empty)
    assert setup_counter == []


def test_flat_column_handler_is_refused_before_upstream(setup_counter):
    handler = DataHandlerLP.from_df(frame(sessions(8)))
    handler._learn.columns = handler._learn.columns.droplevel(0)
    with pytest.raises(ValueError, match="multi-level columns"):
        build(sessions(8), {"window": admission(2, 6, cutoff=8)}, handler=handler)
    assert setup_counter == []


def test_successful_prepare_preserves_written_memory(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    view = root.prepare("window")
    view.assign_data(np.array([4]), np.array([[1.5, -2.5]]))
    written = root._memory.tobytes()
    again = root.prepare("window")
    assert root._memory.tobytes() == written
    assert again._memory.tobytes() == written
    child = again._prepare_seg(slice(DATES[4], DATES[5]))
    assert root._memory.tobytes() == written
    assert child._memory.tobytes() == written


def test_a_class_level_cast_reaches_prepared_views(setup_counter):
    from ashare_lab.research.train import amp_assignment_cast

    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    amp_assignment_cast(root)
    view = root.prepare("window")
    # bf16 has no NumPy dtype, so only the installed cast can land this write.
    view.assign_data(np.array([4]), torch.tensor([[1.5, -2.5]], dtype=torch.bfloat16))
    assert root._memory[4].tolist() == [1.5, -2.5]


def test_an_instance_level_cast_leaves_views_bound_to_the_root(setup_counter):
    root = build(sessions(9), {"window": admission(2, 6, cutoff=8)})
    original = root.assign_data

    def _safe_assign(index, vals):
        return original(index, vals)

    root.assign_data = _safe_assign
    view = root.prepare("window")
    with pytest.raises(ValueError, match="preparation container"):
        view.assign_data(np.array([4]), np.ones((1, 2)))
