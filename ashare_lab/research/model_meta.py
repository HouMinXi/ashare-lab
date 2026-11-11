"""Model metadata convention for staleness gate.

Train side writes models/meta.json after saving a model.
Gate side reads it to compute model_age_days from train_date
instead of filesystem mtime.

Schema (FROZEN):
{
  "model_file": "w11.pt",      # W-NAME of the model file
  "train_date": "2026-08-04",  # Last trading day of training data
  "window_id": "w11",          # Window identifier
  "trained_at": "2026-08-05T13:22:10+08:00",  # ISO timestamp
  "train_host": "gpu-win"      # Where training ran
}
"""
from __future__ import annotations

import filecmp
import json
import logging
import socket
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Schema keys
_MODEL_FILE = "model_file"
_TRAIN_DATE = "train_date"
_WINDOW_ID = "window_id"
_TRAINED_AT = "trained_at"
_TRAIN_HOST = "train_host"

# Asia/Shanghai offset (UTC+8)
_CST = timezone(timedelta(hours=8))


def _is_byte_copy_of(candidate: Path, named: Path) -> bool:
    """True when candidate is a regular-file copy of named.

    gpu-win cannot ship latest.pt as a symlink; predict.py already
    treats that host as a byte copy. Name-only identity would then
    always mismatch on the live path.
    """
    try:
        if not named.is_file() or not candidate.is_file():
            return False
        if candidate.resolve() == named.resolve():
            return True
        return filecmp.cmp(candidate, named, shallow=False)
    except OSError:
        return False


def write_model_meta(
    model_out_path: Path,
    train_data_end: str,
    window_id: str,
) -> None:
    """Write models/meta.json after model save.

    Parameters
    ----------
    model_out_path : Path
        Path to the saved model file (e.g. models/w11.pt).
    train_data_end : str
        ISO date of the last trading day in training data
        (window["train_end"]), NOT the day training ran.
    window_id : str
        Window identifier (e.g. "w11").
    """
    # One file per checkpoint: a second artifact (the live window) shares this
    # directory, and a single meta.json would be overwritten by whichever model
    # wrote last. meta.json is still written for walk-forward models so older
    # readers keep working.
    meta_path = model_out_path.parent / f"{model_out_path.stem}.meta.json"
    meta = {
        _MODEL_FILE: model_out_path.name,
        _TRAIN_DATE: train_data_end,
        _WINDOW_ID: window_id,
        _TRAINED_AT: datetime.now(_CST).isoformat(),
        _TRAIN_HOST: socket.gethostname(),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        log.info("wrote model meta: %s", meta_path)
        if model_out_path.stem != "wlive":
            (model_out_path.parent / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
    except OSError as exc:
        log.error("failed to write model meta %s: %s", meta_path, exc)


def _read_descriptor_file(path: Path) -> tuple[Optional[dict], Optional[str]]:
    if not path.exists():
        return None, "absent"
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None, "unreadable"
    return (raw, None) if isinstance(raw, dict) else (None, "unreadable")


def read_model_descriptor(model_path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Read the single preferred descriptor, before freshness validation."""
    target = model_path.resolve() if model_path.is_symlink() else model_path
    stem_meta = target.parent / f"{target.stem}.meta.json"
    meta_path = stem_meta if stem_meta.exists() else model_path.parent / "meta.json"
    return _read_descriptor_file(meta_path)


def has_research_identity(model_path: Path) -> bool:
    """Inspect retained identity on both the selected entry and resolved target.

    Freshness chooses one descriptor. Admission must not let that precedence
    hide a research marker in another still-present descriptor or manifest.
    """
    for entry in dict.fromkeys((model_path, model_path.resolve())):
        if entry.name.startswith("wexperiment-") or entry.with_suffix(".window.json").exists():
            return True
        for path in (entry.with_suffix(".meta.json"), entry.parent / "meta.json"):
            descriptor, _ = _read_descriptor_file(path)
            if descriptor is not None and any(
                str(descriptor.get(key, "")).startswith("wexperiment-") for key in (_WINDOW_ID, _MODEL_FILE)
            ):
                return True
    return False


def read_model_meta(
    model_path: Path,
) -> tuple[Optional[dict], Optional[str]]:
    """Read models/meta.json for the staleness gate.

    Parameters
    ----------
    model_path : Path
        Path to the model file (e.g. models/w11.pt or models/latest.pt).

    Returns
    -------
    tuple[dict | None, str | None]
        (meta, None) on success; (None, reason) on every failure path.
        Reasons: "absent", "unreadable", "train_date missing",
        "train_date unparseable", "train_date future", "mismatch".
    """
    raw, reason = read_model_descriptor(model_path)
    if raw is None:
        return None, reason

    train_date_str = raw.get(_TRAIN_DATE)
    if train_date_str is None:
        return None, "train_date missing"

    try:
        train_date = date.fromisoformat(train_date_str)
    except (ValueError, TypeError):
        return None, "train_date unparseable"

    # Guard against future dates (Asia/Shanghai)
    today = datetime.now(_CST).date()
    if train_date > today:
        return None, "train_date future"

    # Identity: symlink resolve (W1 latest.pt -> wN.pt) OR a
    # regular-file copy of the named model (gpu-win cannot ship
    # the same symlink; predict.py already treats that as a copy).
    expected_model = raw.get(_MODEL_FILE)
    if expected_model is not None:
        actual_name = model_path.resolve().name
        if expected_model != actual_name and not _is_byte_copy_of(
            model_path, model_path.parent / expected_model
        ):
            return None, "mismatch"

    return raw, None
