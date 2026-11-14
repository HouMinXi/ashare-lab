"""Contained sequence dataset adapter over the installed qlib class.

The adapter keeps the complete retained history and exposes only endpoints
admitted by the pure eligibility rules in :mod:`sequence_eligibility`. It
delegates construction, slicing and iteration to the installed qlib classes
and never edits or monkeypatches them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Literal, NoReturn

import numpy as np
import pandas as pd
import torch

from qlib.contrib.data.dataset import MTSDatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP

from ashare_lab.research.sequence_eligibility import (
    classify_sequence_windows,
    label_maturity_mask,
)


_PURPOSES = ("train", "validation", "retrospective", "live")
_SUPPORTED_HORIZON = 2
_LABEL_EXCLUDED = "label-excluded"
_NO_TARGETS = "no-admitted-targets"
_NO_BATCHES = "no-executable-batches"
_INSUFFICIENT = "insufficient-own-history"
_CALENDAR_GAP = "calendar-gap"


@dataclass(frozen=True)
class SegmentAdmission:
    """Immutable admission policy for one named segment."""

    start: pd.Timestamp
    end: pd.Timestamp
    purpose: Literal["train", "validation", "retrospective", "live"]
    outcome_cutoff: pd.Timestamp


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _midnight(value: object, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value) or value.tzinfo is not None:
        raise ValueError(f"{name} must be a timezone-naive midnight timestamp")
    if value != value.normalize():
        raise ValueError(f"{name} must be a midnight timestamp")
    return pd.Timestamp(value)


def _calendar(values: object) -> pd.DatetimeIndex:
    if not isinstance(values, pd.DatetimeIndex):
        raise ValueError("calendar must be a DatetimeIndex")
    if values.hasnans or values.tz is not None:
        raise ValueError("calendar must contain timezone-naive dates without NaT")
    if not values.is_unique or not values.is_monotonic_increasing:
        raise ValueError("calendar must be sorted and unique")
    if any(value != value.normalize() for value in values):
        raise ValueError("calendar must contain midnight dates")
    return pd.DatetimeIndex(values.to_numpy().copy())


def _raw_labels(values: object) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise ValueError("raw_labels must be a Series")
    index = values.index
    if not isinstance(index, pd.MultiIndex) or list(index.names) != ["instrument", "datetime"]:
        raise ValueError("raw_labels must use an instrument, datetime MultiIndex")
    if len(values) == 0:
        raise ValueError("raw_labels must not be empty")
    if not index.is_unique:
        raise ValueError("raw_labels index must be unique")
    if not all(isinstance(name, str) and name for name in index.get_level_values("instrument")):
        raise ValueError("raw_labels instruments must be nonempty strings")
    dates = index.get_level_values("datetime")
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or dates.tz is not None:
        raise ValueError("raw_labels dates must be timezone-naive without NaT")
    if any(value != value.normalize() for value in dates):
        raise ValueError("raw_labels dates must be midnight")
    if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError("raw_labels must hold real numbers")
    if pd.api.types.is_complex_dtype(values.dtype) or pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError("raw_labels must hold real numbers")
    return values.copy()


def _admissions(values: object) -> dict[str, SegmentAdmission]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("segments must be a nonempty mapping")
    copied: dict[str, SegmentAdmission] = {}
    for name, admission in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("segment names must be nonempty strings")
        if not isinstance(admission, SegmentAdmission):
            raise ValueError("segment values must be SegmentAdmission instances")
        start = _midnight(admission.start, "segment start")
        end = _midnight(admission.end, "segment end")
        cutoff = _midnight(admission.outcome_cutoff, "segment cutoff")
        if admission.purpose not in _PURPOSES:
            raise ValueError("segment purpose is not supported")
        if start > end:
            raise ValueError("segment start must not follow its end")
        if admission.purpose == "live":
            if start != end or cutoff != end:
                raise ValueError("a live segment is one day with cutoff equal to it")
        if admission.purpose in ("train", "validation") and cutoff != end:
            raise ValueError("train and validation cutoffs must equal the segment end")
        copied[name] = SegmentAdmission(start, end, admission.purpose, cutoff)
    purposes = {admission.purpose for admission in copied.values()}
    if "live" in purposes and len(purposes) > 1:
        raise ValueError("live segments cannot share a root with other purposes")
    return copied


class ContainedMTSDatasetH(MTSDatasetH):
    """MTSDatasetH that emits only admitted endpoints and refuses unsafe writes."""

    def __init__(
        self,
        handler: DataHandlerLP,
        segments: Mapping[str, SegmentAdmission],
        *,
        calendar: pd.DatetimeIndex,
        raw_labels: pd.Series,
        seq_len: int,
        num_states: int,
        horizon: int = _SUPPORTED_HORIZON,
        batch_size: int = -1,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(handler, DataHandlerLP):
            raise ValueError("handler must be an initialized DataHandlerLP instance")
        seq_len = _positive_integer(seq_len, "seq_len")
        num_states = _positive_integer(num_states, "num_states")
        horizon = _positive_integer(horizon, "horizon")
        if horizon != _SUPPORTED_HORIZON:
            raise ValueError("only the accepted two-session outcome horizon is supported")
        if seq_len <= horizon:
            raise ValueError("seq_len must exceed the outcome horizon")
        if isinstance(batch_size, (bool, np.bool_, np.timedelta64)) or not isinstance(batch_size, Integral):
            raise ValueError("batch_size must be a nonzero integer")
        if batch_size == 0:
            raise ValueError("batch_size must be a nonzero integer")
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise ValueError("shuffle and drop_last must be booleans")
        calendar_index = _calendar(calendar)
        labels = _raw_labels(raw_labels)
        admissions = _admissions(segments)
        final_session = calendar_index[-1] if len(calendar_index) else None
        if final_session is not None:
            for admission in admissions.values():
                if admission.purpose == "retrospective" and admission.outcome_cutoff > final_session:
                    raise ValueError("a retrospective cutoff must not exceed the observed calendar")
        master = self._retained_master(handler)
        self._adapter_booting = True
        self._master_size = len(master)
        self._calendar = calendar_index
        self._admissions = MappingProxyType(admissions)
        self._raw_source = labels
        self._bound = False
        self._admission_name: str | None = None
        self._admission: SegmentAdmission | None = None
        self._window: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self._admitted = np.empty(0, dtype=np.int64)
        self._write_permitted = np.empty(0, dtype=np.int64)
        self._diagnostics: dict[str, int] = {}
        try:
            super().__init__(
                handler,
                {name: (admission.start, admission.end) for name, admission in admissions.items()},
                seq_len=seq_len,
                horizon=horizon,
                num_states=num_states,
                memory_mode="sample",
                batch_size=int(batch_size),
                n_samples=None,
                shuffle=shuffle,
                drop_last=drop_last,
                input_size=None,
            )
        finally:
            self._adapter_booting = False

    @staticmethod
    def _retained_master(handler: DataHandlerLP) -> pd.DataFrame:
        learned = getattr(handler, "_learn", None)
        if learned is None or not isinstance(learned, pd.DataFrame):
            raise ValueError("handler has no retained learned representation")
        columns = learned.columns
        if not isinstance(columns, pd.MultiIndex):
            raise ValueError("handler learned representation must use multi-level columns")
        present = "label" in columns.get_level_values(0)
        if not present:
            raise ValueError("handler learned representation has no label column")
        if len(learned) == 0:
            raise ValueError("the learned master is empty; refuse before upstream setup")
        if len(learned) == 1:
            raise ValueError("a one-row learned master is unsupported; refuse before upstream setup")
        return learned

    def setup_data(self, handler_kwargs: dict | None = None, **kwargs) -> None:
        if not self._adapter_booting:
            raise ValueError("the adapter refuses lifecycle reconfiguration")
        if handler_kwargs is not None or kwargs:
            raise ValueError("the adapter configures its own handler data")
        super().setup_data()
        self._capture_master_context()

    def _capture_master_context(self) -> None:
        if len(self._index) != self._master_size:
            raise RuntimeError("retained master changed during setup")
        self._master_index = self._index
        self._master_batch_slices = self._batch_slices
        self._master_daily_slices = self._daily_slices
        self._master_daily_index = self._daily_index
        reasons = classify_sequence_windows(self._index, self._calendar, self.seq_len)
        if reasons.index is not self._index and not reasons.index.equals(self._index):
            raise RuntimeError("structural reasons do not match the master index")
        self._reason_codes = reasons.cat.codes.to_numpy()
        self._reason_categories = list(reasons.cat.categories)
        missing = self._master_index.difference(self._raw_source.index)
        if len(missing):
            raise ValueError("raw labels miss retained master rows")
        aligned = self._raw_source.reindex(self._master_index)
        self._raw_values = aligned.to_numpy(dtype="float64", copy=True, na_value=np.nan)
        self._raw_finite = np.isfinite(self._raw_values)
        self._learned_finite = np.isfinite(self._label)
        self._raw_extra_keys = int(len(self._raw_source.index.difference(self._master_index)))
        if self._label.ndim != 1:
            raise ValueError("learned labels must be one value per retained row")
        if len(self._label) != len(self._index):
            raise ValueError("learned labels do not align with the master index")
        if self._memory.shape != (len(self._data), self.num_states):
            raise RuntimeError("upstream memory shape is not the admitted sample layout")
        if self._memory.dtype != np.float32:
            raise RuntimeError("upstream memory is not the pinned float32 storage")

    def _require_ready(self) -> None:
        if not hasattr(self, "_reason_codes"):
            raise RuntimeError("the adapter is not initialized")

    def _require_bound(self) -> None:
        self._require_ready()
        if not self._bound or self._admission is None or self._window is None:
            raise ValueError("a root is a preparation container, not an iterable view")

    def _structural(self) -> np.ndarray:
        return self._reason_codes == self._reason_categories.index("eligible")

    def _partition(self, admission: SegmentAdmission, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
        dates = self._master_index.get_level_values("datetime")
        in_window = np.asarray((dates >= window[0]) & (dates <= window[1]))
        structural = self._structural()
        insufficient_code = self._reason_categories.index(_INSUFFICIENT)
        gap_code = self._reason_categories.index(_CALENDAR_GAP)
        insufficient = int(np.count_nonzero(in_window & (self._reason_codes == insufficient_code)))
        gap = int(np.count_nonzero(in_window & (self._reason_codes == gap_code)))
        if admission.purpose == "live":
            admitted_mask = in_window & structural
            excluded = 0
        else:
            mature = label_maturity_mask(
                self._master_index, self._calendar, horizon=self.horizon, cutoff=admission.outcome_cutoff
            )
            usable = structural & self._raw_finite & self._learned_finite & mature
            admitted_mask = in_window & usable
            excluded = int(np.count_nonzero(in_window & structural & ~usable))
        admitted = np.flatnonzero(admitted_mask).astype(np.int64)
        total = int(np.count_nonzero(in_window))
        if insufficient + gap + excluded + len(admitted) != total:
            raise RuntimeError("admission partition does not sum to the input total")
        return {
            "input_in_range": total,
            "insufficient_own_history": insufficient,
            "calendar_gap": gap,
            _LABEL_EXCLUDED: excluded,
            "admitted": len(admitted),
            "positions": admitted,
        }

    def prepare(
        self,
        segments: str
        | list[str]
        | tuple[str, ...]
        | slice
        | tuple[pd.Timestamp, pd.Timestamp],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> "ContainedMTSDatasetH | list[ContainedMTSDatasetH]":
        self._require_ready()
        if col_set != DataHandler.CS_ALL or data_key != DataHandlerLP.DK_I or kwargs:
            raise ValueError("the adapter supports only its default fetch settings")
        names, window = self._resolve_selector(segments)
        if names is not None:
            if self._bound:
                if len(names) != 1 or names[0] != self._admission_name:
                    raise ValueError("a bound view may only re-prepare its own admission")
            views = [self._build_view(name, self._named_window(name)) for name in names]
            return views[0] if isinstance(segments, str) else views
        return self._build_view(self._admission_name, window)

    def _named_window(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The window a named admission resolves to from this view.

        A root gets the whole admission. A bound view keeps its own window, so
        re-preparing the name cannot widen an interval a caller narrowed.
        """
        if self._bound:
            return self._window
        admission = self._admissions[name]
        return admission.start, admission.end

    def _resolve_selector(self, selector):
        if isinstance(selector, str):
            if selector not in self._admissions:
                raise ValueError("unknown segment name")
            return [selector], None
        if isinstance(selector, slice):
            if selector.step is not None:
                raise ValueError("a slice step is not supported")
            return None, self._contained(selector.start, selector.stop)
        if isinstance(selector, (list, tuple)):
            if not selector:
                raise ValueError("an empty selector is not supported")
            if all(isinstance(item, str) for item in selector):
                if len(set(selector)) != len(selector):
                    raise ValueError("duplicate segment names are not supported")
                for name in selector:
                    if name not in self._admissions:
                        raise ValueError("unknown segment name")
                return list(selector), None
            if len(selector) == 2 and all(isinstance(item, pd.Timestamp) for item in selector):
                return None, self._contained(selector[0], selector[1])
        raise ValueError("unsupported selector")

    def _contained(self, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = _midnight(start, "range start")
        end = _midnight(end, "range end")
        if start > end:
            raise ValueError("range start must not follow its end")
        parent = self._window if self._window is not None else None
        if parent is not None:
            if start < parent[0] or end > parent[1]:
                raise ValueError("a nested range must stay inside its parent view")
        elif not self._bound:
            raise ValueError("a root requires a named admission, not an anonymous range")
        return start, end

    def _build_view(self, name: str, window: tuple[pd.Timestamp, pd.Timestamp]) -> "ContainedMTSDatasetH":
        admission = self._admissions[name]
        if window[0] < admission.start or window[1] > admission.end:
            raise ValueError("a view range must stay inside its admission")
        projection = self.__copy__()
        projection._batch_slices = self._master_batch_slices
        projection._daily_slices = self._master_daily_slices
        projection._daily_index = self._master_daily_index
        target = MTSDatasetH._prepare_seg(projection, (window[0], window[1]))
        partition = self._partition(admission, window)
        admitted = partition["positions"]
        if not len(admitted):
            raise ValueError(_NO_TARGETS + ": the requested range admits no endpoint")
        keep = np.isin(np.array([item.stop - 1 for item in target._batch_slices], dtype=np.int64), admitted)
        slices = np.asarray(target._batch_slices, dtype=object)[keep]
        admitted_set = set(admitted.tolist())
        daily_slices = []
        daily_positions = []
        for position, group in enumerate(target._daily_slices):
            kept = [item for item in group if item.stop - 1 in admitted_set]
            if kept:
                daily_slices.append(kept)
                daily_positions.append(position)
        target._batch_slices = slices
        target._daily_slices = np.asarray(daily_slices, dtype=object) if daily_slices else np.empty(0, dtype=object)
        target._daily_index = target._daily_index.iloc[daily_positions]
        # The contract requires the sample and daily streams to reach exactly the
        # same endpoints; both are filtered from `admitted` here, so a drift can
        # only come from a change in the delegated selector or in this projection.
        reached = np.sort(np.array([item.stop - 1 for item in slices], dtype=np.int64))
        daily_reached = np.sort(
            np.array([item.stop - 1 for group in target._daily_slices for item in group], dtype=np.int64)
        )
        settled = np.sort(admitted)
        bounds = np.array([[item.start, item.stop] for item in slices], dtype=np.int64).reshape(-1, 2)
        if (
            len(reached) != len(settled)
            or not np.array_equal(reached, settled)
            or len(daily_reached) != len(settled)
            or not np.array_equal(daily_reached, settled)
        ):
            raise RuntimeError("sample and daily endpoints do not reconcile with the admission")
        if (bounds[:, 0] >= bounds[:, 1]).any() or (bounds[:, 1] > len(target._index)).any():
            raise RuntimeError("a slice bound is inconsistent with the master")
        mode = "daily" if target.batch_size < 0 else "sample"
        units = len(slices) if target.batch_size > 0 else len(target._daily_slices)
        detail = f"mode={mode} units={units} batch_size={abs(target.batch_size)} drop_last={target.drop_last}"
        if MTSDatasetH.__len__(target) == 0:
            raise ValueError(_NO_BATCHES + ": no executable batch; " + detail)
        target._bound = True
        target._admission_name = name
        target._admission = admission
        target._window = window
        target._admitted = admitted
        # A child may only write what both its own admission and its parent
        # allow; the pair of lines below and the length check after them turn a
        # widened re-preparation into a refusal rather than a quiet widening.
        permitted = admitted.copy()
        if self._bound:
            permitted = np.intersect1d(permitted, self._write_permitted)
        if admission.purpose == "live":
            target._write_permitted = np.empty(0, dtype=np.int64)
        else:
            if len(permitted) != len(admitted):
                raise RuntimeError("write permission does not match the admitted endpoints")
            target._write_permitted = permitted
        target._diagnostics = {key: value for key, value in partition.items() if key != "positions"}
        target._diagnostics["raw_extra_keys"] = self._raw_extra_keys
        return target

    def _prepare_seg(self, slc, **kwargs) -> "ContainedMTSDatasetH":
        self._require_bound()
        if kwargs:
            raise ValueError("the adapter accepts no extra selection keywords")
        if isinstance(slc, slice):
            if slc.step is not None:
                raise ValueError("a slice step is not supported")
            bounds = (slc.start, slc.stop)
        elif isinstance(slc, (list, tuple)) and len(slc) == 2:
            bounds = (slc[0], slc[1])
        else:
            raise ValueError("use a two-boundary range for a contained view")
        start, end = self._contained(bounds[0], bounds[1])
        return self._build_view(self._admission_name, (start, end))

    def __copy__(self) -> "ContainedMTSDatasetH":
        cls = self.__class__
        clone = cls.__new__(cls)
        clone.__dict__ = self.__dict__.copy()
        clone._batch_slices = np.array(self._batch_slices, dtype=object, copy=True)
        clone._daily_slices = np.array(self._daily_slices, dtype=object, copy=True)
        clone._daily_index = self._daily_index.copy()
        clone._admitted = self._admitted.copy()
        clone._write_permitted = self._write_permitted.copy()
        clone._diagnostics = dict(self._diagnostics)
        return clone

    def _count_with(self, batch_size: int, drop_last: bool) -> int:
        probe = self.__copy__()
        probe.batch_size, probe.n_samples, probe.drop_last, probe.shuffle = batch_size, None, drop_last, False
        return int(MTSDatasetH.__len__(probe))

    def __len__(self) -> int:
        self._require_bound()
        count = int(MTSDatasetH.__len__(self))
        if count == 0:
            raise ValueError(_NO_BATCHES + ": the bound view yields no executable batch")
        return count

    def __iter__(self) -> Iterator[dict]:
        self._require_bound()
        if int(MTSDatasetH.__len__(self)) == 0:
            raise ValueError(_NO_BATCHES + ": the bound view yields no executable batch")
        yield from MTSDatasetH.__iter__(self)

    def train(self) -> None:
        self._require_ready()
        if not self._bound:
            MTSDatasetH.train(self)
            return
        missing, drop_last = self.params[0], self.params[2]
        if self._count_with(missing, drop_last) == 0:
            raise ValueError(_NO_BATCHES + ": the saved training mode yields no executable batch")
        MTSDatasetH.train(self)

    def eval(self) -> None:
        self._require_ready()
        if not self._bound:
            MTSDatasetH.eval(self)
            return
        if self._count_with(-1, False) == 0:
            raise ValueError(_NO_BATCHES + ": evaluation mode yields no executable batch")
        MTSDatasetH.eval(self)

    def clear_memory(self) -> None:
        self._require_ready()
        MTSDatasetH.clear_memory(self)

    def assign_data(self, index, vals) -> None:
        """Write memory rows for one batch.

        ``index`` must be ``batch["index"]``. Those are master row positions,
        the only id space this write accepts; a daily batch id is a different
        space whose values may coincide with writable rows, so it is neither
        detected nor reinterpreted.
        """
        self._require_bound()
        positions = self._checked_positions(index)
        values = self._checked_values(positions, vals)
        MTSDatasetH.assign_data(self, positions, values)

    def _checked_positions(self, index) -> np.ndarray:
        if isinstance(index, torch.Tensor):
            try:
                index = index.detach().cpu().numpy()
            except (TypeError, RuntimeError) as exc:
                raise ValueError("index must be convertible to a NumPy array") from exc
        if type(index) is not np.ndarray or index.ndim != 1 or index.size == 0:
            raise ValueError("index must be a plain nonempty one-dimensional array")
        if index.dtype == np.bool_ or np.issubdtype(index.dtype, np.timedelta64):
            raise ValueError("index must hold integer positions")
        if not np.issubdtype(index.dtype, np.integer):
            raise ValueError("index must hold integer positions")
        positions = index.astype(np.int64, copy=True)
        if (positions < 0).any() or (positions >= len(self._index)).any():
            raise ValueError("index position is outside the master")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("index positions must be unique")
        if not np.isin(positions, self._write_permitted).all():
            raise ValueError("index position is not writable in this view")
        return positions

    def _checked_values(self, positions: np.ndarray, vals) -> np.ndarray:
        if isinstance(vals, torch.Tensor):
            try:
                vals = vals.detach().cpu().numpy()
            except (TypeError, RuntimeError) as exc:
                raise ValueError("values must be convertible to a NumPy array") from exc
        if type(vals) is not np.ndarray:
            raise ValueError("values must be a plain NumPy array or a torch tensor")
        if vals.dtype == np.bool_ or np.issubdtype(vals.dtype, np.complexfloating):
            raise ValueError("values must be real numbers")
        if vals.dtype.kind in "OSUMmV":
            raise ValueError("values must be real numbers")
        if vals.shape != (len(positions), self.num_states):
            raise ValueError("values must match the requested rows and state count")
        source = vals.astype(np.float64, copy=True)
        if not np.isfinite(source).all():
            raise ValueError("values must be finite")
        if np.abs(source).max(initial=0.0) > float(np.finfo(self._memory.dtype).max):
            raise ValueError("values do not fit the destination memory dtype")
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                converted = source.astype(self._memory.dtype)
        except FloatingPointError as exc:
            raise ValueError("values do not fit the destination memory dtype") from exc
        if converted.shape != source.shape or converted.dtype != self._memory.dtype:
            raise ValueError("destination conversion changed the requested layout")
        if not np.isfinite(converted).all():
            raise ValueError("values do not fit the destination memory dtype")
        return np.array(converted, dtype=self._memory.dtype, copy=True)

    def config(self, handler_kwargs: dict | None = None, **kwargs) -> None:
        raise ValueError("the adapter refuses lifecycle reconfiguration")

    def __getstate__(self) -> NoReturn:
        raise TypeError("the contained dataset refuses serialization")
