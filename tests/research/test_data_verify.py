"""Unit tests for ashare_lab.research.data_verify.

Three tests exercise instrument verification without qlib runtime:
1. Missing file raises FileNotFoundError
2. Active count counted correctly
3. Cutoff boundary (end_date == "2026-01-01" is active; "2025-12-31" is not)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ashare_lab.research.data_verify import verify_csi500_instruments


def _write_instruments(tmp_path: Path, content: str) -> Path:
    """Write a fake csi500.txt and return the provider_uri root."""
    instruments_dir = tmp_path / "instruments"
    instruments_dir.mkdir(parents=True)
    (instruments_dir / "csi500.txt").write_text(textwrap.dedent(content))
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: missing file raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path: Path) -> None:
    """verify_csi500_instruments raises FileNotFoundError when file absent."""
    # tmp_path has no instruments/ subdirectory
    with pytest.raises(FileNotFoundError, match="csi500"):
        verify_csi500_instruments(provider_uri=tmp_path)


# ---------------------------------------------------------------------------
# Test 2: active count is correct (end_date >= 2026-01-01)
# ---------------------------------------------------------------------------

def test_active_count(tmp_path: Path) -> None:
    """Active count equals rows with end_date >= 2026-01-01."""
    # 3 active rows (end_date >= 2026-01-01), 2 expired rows, 1 blank line
    content = """\
        SH600000\t2018-01-02\t2026-12-31
        SH600001\t2018-01-02\t2026-06-15
        SH600002\t2018-01-02\t2026-01-01
        SH600003\t2018-01-02\t2025-12-31
        SH600004\t2018-01-02\t2020-01-01

    """
    provider_uri = _write_instruments(tmp_path, content)
    result = verify_csi500_instruments(provider_uri=provider_uri)

    assert result["active_count"] == 3
    assert result["total"] == 5
    assert result["ok"] is False  # 3 < 500


# ---------------------------------------------------------------------------
# Test 3: cutoff boundary -- exactly "2026-01-01" is active; "2025-12-31" is not
# ---------------------------------------------------------------------------

def test_cutoff_boundary(tmp_path: Path) -> None:
    """end_date == '2026-01-01' counts as active; '2025-12-31' does not."""
    content = """\
        SH600000\t2018-01-02\t2026-01-01
        SH600001\t2018-01-02\t2025-12-31
    """
    provider_uri = _write_instruments(tmp_path, content)
    result = verify_csi500_instruments(provider_uri=provider_uri)

    assert result["active_count"] == 1
    assert result["total"] == 2
