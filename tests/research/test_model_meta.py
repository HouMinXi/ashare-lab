"""Tests for model_meta.py: write_model_meta + read_model_meta."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from ashare_lab.research.model_meta import read_model_meta, write_model_meta


# -- write_model_meta tests --

def test_write_model_meta_schema(tmp_path):
    """write_model_meta writes frozen schema with correct values."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    write_model_meta(model_path, "2026-08-04", "w11")
    meta_path = tmp_path / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_file"] == "w11.pt"
    assert meta["train_date"] == "2026-08-04"
    assert meta["window_id"] == "w11"
    assert "trained_at" in meta
    assert "train_host" in meta


def test_write_model_meta_train_date_not_today(tmp_path):
    """train_date is train_data_end, NOT date.today()."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    write_model_meta(model_path, "2026-07-01", "w11")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["train_date"] == "2026-07-01"
    assert meta["train_date"] != date.today().isoformat()


# -- read_model_meta tests --

def test_read_meta_valid(tmp_path):
    """Valid meta -> (meta, None)."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {
        "model_file": "w11.pt",
        "train_date": "2026-08-04",
        "window_id": "w11",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is not None
    assert reason is None
    assert result["train_date"] == "2026-08-04"


def test_read_meta_absent(tmp_path):
    """No meta.json -> (None, "absent")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "absent"


def test_read_meta_unreadable(tmp_path):
    """Corrupt JSON -> (None, "unreadable")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    (tmp_path / "meta.json").write_text("not json{")
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "unreadable"


def test_read_meta_train_date_missing(tmp_path):
    """meta.json without train_date -> (None, "train_date missing")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    (tmp_path / "meta.json").write_text(json.dumps({"model_file": "w11.pt"}))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date missing"


def test_read_meta_train_date_unparseable(tmp_path):
    """train_date not ISO format -> (None, "train_date unparseable")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {"model_file": "w11.pt", "train_date": "not-a-date"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date unparseable"


def test_read_meta_train_date_future(tmp_path):
    """train_date in future -> (None, "train_date future")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    future = (date.today() + timedelta(days=1)).isoformat()
    meta = {"model_file": "w11.pt", "train_date": future}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "train_date future"


def test_read_meta_model_file_mismatch(tmp_path):
    """model_file mismatch -> (None, "mismatch")."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    meta = {"model_file": "w10.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "mismatch"


def test_read_meta_symlink_resolves(tmp_path):
    """latest.pt -> w11.pt with meta w11.pt -> accepted (R4 #3)."""
    w11 = tmp_path / "w11.pt"
    w11.touch()
    latest = tmp_path / "latest.pt"
    latest.symlink_to(w11)
    meta = {"model_file": "w11.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(latest)
    assert result is not None
    assert reason is None


def test_read_meta_symlink_mismatch(tmp_path):
    """latest.pt -> w10.pt with meta w11.pt -> mismatch."""
    w10 = tmp_path / "w10.pt"
    w10.touch()
    latest = tmp_path / "latest.pt"
    latest.symlink_to(w10)
    meta = {"model_file": "w11.pt", "train_date": "2026-08-04"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, reason = read_model_meta(latest)
    assert result is None
    assert reason == "mismatch"


# -- Injection tests --

def test_injection_delete_writer(tmp_path):
    """Delete write_model_meta call -> meta never created."""
    model_path = tmp_path / "w11.pt"
    model_path.touch()
    # Simulate: writer not called
    result, reason = read_model_meta(model_path)
    assert result is None
    assert reason == "absent"
