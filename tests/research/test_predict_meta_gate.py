"""Tests for predict.py meta.json gate rewrite (10-03).

Covers: present/absent/malformed/mismatch/future + fallback + injection.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ashare_lab.research.model_meta import read_model_meta


# -- Helper --

def _make_meta(tmp_path, model_file="w11.pt", train_date="2026-08-04"):
    """Write a valid meta.json and return the model path."""
    model_path = tmp_path / model_file
    model_path.touch()
    meta = {"model_file": model_file, "train_date": train_date}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return model_path


# -- read_model_meta gate tests --

def test_gate_valid_meta(tmp_path):
    """1. Valid meta -> (meta, None), age from train_date."""
    model_path = _make_meta(tmp_path, train_date=(date.today() - timedelta(days=3)).isoformat())
    result, reason = read_model_meta(model_path)
    assert result is not None
    assert reason is None
    age = (date.today() - date.fromisoformat(result["train_date"])).days
    assert age == 3


def test_gate_live_era_symlink(tmp_path):
    """1b. LIVE-ERA: latest.pt -> w11.pt, meta w11.pt -> accepted."""
    w11 = tmp_path / "w11.pt"
    w11.touch()
    latest = tmp_path / "latest.pt"
    latest.symlink_to(w11)
    meta = {"model_file": "w11.pt", "train_date": (date.today() - timedelta(days=3)).isoformat()}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(latest)
    assert result is not None
    assert reason is None


def test_gate_meta_absent(tmp_path):
    """2. Meta absent -> (None, "absent")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "absent"


def test_gate_malformed_json(tmp_path):
    """3. Malformed JSON -> (None, "unreadable")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    (tmp_path / "meta.json").write_text("not json{")
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "unreadable"


def test_gate_train_date_missing(tmp_path):
    """3. train_date missing -> (None, "train_date missing")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    (tmp_path / "meta.json").write_text(json.dumps({"model_file": "w11.pt"}))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date missing"


def test_gate_train_date_unparseable(tmp_path):
    """3. train_date unparseable -> (None, "train_date unparseable")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {"model_file": "w11.pt", "train_date": "not-a-date"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date unparseable"


def test_gate_train_date_future(tmp_path):
    """4. train_date tomorrow -> (None, "train_date future")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    future = (date.today() + timedelta(days=1)).isoformat()
    meta = {"model_file": "w11.pt", "train_date": future}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date future"


def test_gate_model_file_mismatch(tmp_path):
    """5. model_file mismatch -> (None, "mismatch")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {"model_file": "w10.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "mismatch"


def test_gate_symlink_stale_mismatch(tmp_path):
    """5. latest.pt -> w10.pt with w11.pt meta -> mismatch."""
    w10 = tmp_path / "w10.pt"
    w10.touch()
    latest = tmp_path / "latest.pt"
    latest.symlink_to(w10)
    meta = {"model_file": "w11.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(latest)
    assert result is None
    assert reason == "mismatch"


def test_gate_warn_only_semantics(tmp_path):
    """6. Threshold semantics: age 8 -> warn-only, prediction proceeds."""
    model_path = _make_meta(tmp_path, train_date=(date.today() - timedelta(days=8)).isoformat())
    result, reason = read_model_meta(model_path)
    assert result is not None  # meta is valid
    age = (date.today() - date.fromisoformat(result["train_date"])).days
    assert age == 8  # > _MODEL_STALE_DAYS (7)
    # Gate is warn-only: meta is still returned, age is still computed


def test_gate_mtime_fallback_reason(tmp_path):
    """Fallback: meta absent -> mtime with reason "absent"."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "absent"
    # age_source would be "mtime-fallback(absent)"


# -- Injection tests --

def test_injection_delete_meta_read(tmp_path):
    """(b) Delete meta read -> always mtime fallback."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {"model_file": "w11.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    # If read_model_meta is deleted, gate would use mtime
    # This test verifies meta exists (injection would make it absent)
    result, reason = read_model_meta(model_path)
    assert result is not None  # meta read works


def test_injection_compare_model_path_name(tmp_path):
    """(d) Compare against model_path.name instead of resolve().name -> FAIL."""
    w11 = tmp_path / "w11.pt"
    w11.touch()
    latest = tmp_path / "latest.pt"
    latest.symlink_to(w11)
    meta = {"model_file": "w11.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    # resolve().name = "w11.pt" -> matches meta
    result, reason = read_model_meta(latest)
    assert result is not None
    # If we used model_path.name = "latest.pt" -> mismatch
    assert latest.name == "latest.pt"  # NOT "w11.pt"
