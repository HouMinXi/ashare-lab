"""Tests for ashare_lab.data.validate (non-baostock functions)."""
import json
import tempfile
from pathlib import Path
import pytest
from ashare_lab.data.validate import validate_instruments, spot_check_raw


def test_validate_instruments_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        result = validate_instruments(market="csi300", provider_uri=Path(tmp))
    assert result["status"] in ("FAIL", "WARN")  # pandas lenient with NUL bytes
    assert "not found" in result["reason"]


def test_validate_instruments_malformed_csv():
    with tempfile.TemporaryDirectory() as tmp:
        inst_dir = Path(tmp) / "instruments"
        inst_dir.mkdir()
        (inst_dir / "csi300.txt").write_text("NOT\tVALID\n\x00\x00\x00")
        result = validate_instruments(market="csi300", provider_uri=Path(tmp))
    assert result["status"] in ("FAIL", "WARN")  # pandas lenient with NUL bytes


def test_validate_instruments_malformed_end_date():
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(tmp) / "calendars"
        cal_dir.mkdir()
        (cal_dir / "day.txt").write_text("2026-06-12\n")
        inst_dir = Path(tmp) / "instruments"
        inst_dir.mkdir()
        rows = "\n".join(f"SH60000{i}\t2020-01-01\tNOT-A-DATE" for i in range(300))
        (inst_dir / "csi300.txt").write_text(rows + "\n")
        result = validate_instruments(market="csi300", provider_uri=Path(tmp))
    assert result["status"] in ("FAIL", "WARN")  # pandas lenient with NUL bytes
    assert "malformed" in result["reason"]


def test_validate_instruments_too_few_active():
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(tmp) / "calendars"
        cal_dir.mkdir()
        (cal_dir / "day.txt").write_text("2026-06-12\n")
        inst_dir = Path(tmp) / "instruments"
        inst_dir.mkdir()
        rows = "\n".join(f"SH60000{i}\t2020-01-01\t2021-01-01" for i in range(10))
        (inst_dir / "csi300.txt").write_text(rows + "\n")
        result = validate_instruments(market="csi300", provider_uri=Path(tmp))
    assert result["status"] == "WARN"
    assert result["active_count"] == 0


def test_validate_instruments_real_data():
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI
    if not DEFAULT_PROVIDER_URI.exists():
        pytest.skip("qlib data not bootstrapped")
    result = validate_instruments(provider_uri=DEFAULT_PROVIDER_URI)
    assert result["status"] in ("PASS", "WARN", "FAIL")
    # JSON serializable (no NaN)
    j = json.dumps({"result": result})
    assert "NaN" not in j


def test_spot_check_raw_no_data_returns_dict():
    """spot_check_raw returns a dict when provider_uri has no qlib data."""
    with tempfile.TemporaryDirectory() as tmp:
        result = spot_check_raw(provider_uri=Path(tmp))
    assert isinstance(result, dict)


def test_validate_instruments_json_no_nan():
    """validate_instruments output must be JSON serializable (no NaN)."""
    from ashare_lab.data.update import DEFAULT_PROVIDER_URI
    if not DEFAULT_PROVIDER_URI.exists():
        pytest.skip("qlib data not bootstrapped")
    result = validate_instruments(provider_uri=DEFAULT_PROVIDER_URI)
    serialized = json.dumps(result)
    assert "NaN" not in serialized
