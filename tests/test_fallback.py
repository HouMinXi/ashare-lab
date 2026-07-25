"""Tests for ashare_lab.data.fallback (pure logic only; no network calls)."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ashare_lab.data.fallback import (
    _CSV_COLUMNS,
    _all_instruments,
    _bs_to_qlib,
    _calendar_dates,
    _fetch_symbol,
    _qlib_to_bs,
    _write_csvs,
    find_missing_dates,
    gap_fill,
)


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


class TestSymbolConversion:
    def test_bs_to_qlib_sh(self):
        assert _bs_to_qlib("sh.600000") == "SH600000"

    def test_bs_to_qlib_sz(self):
        assert _bs_to_qlib("sz.000001") == "SZ000001"

    def test_bs_to_qlib_no_dot(self):
        # Degenerate input: no dot -- returned as-is
        assert _bs_to_qlib("600000") == "600000"

    def test_qlib_to_bs_sh(self):
        assert _qlib_to_bs("SH600000") == "sh.600000"

    def test_qlib_to_bs_sz(self):
        assert _qlib_to_bs("SZ000001") == "sz.000001"

    def test_qlib_to_bs_too_short(self):
        assert _qlib_to_bs("SH") == "SH"

    def test_round_trip(self):
        symbol = "SZ000001"
        assert _bs_to_qlib(_qlib_to_bs(symbol)) == symbol


# ---------------------------------------------------------------------------
# _calendar_dates
# ---------------------------------------------------------------------------


class TestCalendarDates:
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _calendar_dates(Path(tmp))
        assert result == set()

    def test_parses_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "calendars").mkdir()
            (p / "calendars" / "day.txt").write_text(
                "2024-03-01\n2024-03-04\n2024-03-05\n"
            )
            result = _calendar_dates(p)
        assert result == {"2024-03-01", "2024-03-04", "2024-03-05"}

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "calendars").mkdir()
            (p / "calendars" / "day.txt").write_text("")
            result = _calendar_dates(p)
        assert result == set()

    def test_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "calendars").mkdir()
            (p / "calendars" / "day.txt").write_text("  2024-03-01  \n2024-03-04\n")
            result = _calendar_dates(p)
        assert "2024-03-01" in result


# ---------------------------------------------------------------------------
# _all_instruments
# ---------------------------------------------------------------------------


class TestAllInstruments:
    def test_no_instruments_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _all_instruments(Path(tmp))
        assert result == []

    def test_collects_unique_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            inst = p / "instruments"
            inst.mkdir()
            (inst / "csi300.txt").write_text(
                "SH600000 2015-01-01 2999-12-31\nSZ000001 2015-01-01 2999-12-31\n"
            )
            (inst / "csi500.txt").write_text(
                "SZ000001 2015-01-01 2999-12-31\nSH600036 2015-01-01 2999-12-31\n"
            )
            result = _all_instruments(p)
        assert sorted(result) == ["SH600000", "SH600036", "SZ000001"]

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "instruments").mkdir()
            (p / "instruments" / "all.txt").write_text(
                "SZ000001 2015-01-01 2999-12-31\nSH600000 2015-01-01 2999-12-31\n"
            )
            result = _all_instruments(p)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# find_missing_dates
# ---------------------------------------------------------------------------


class TestFindMissingDates:
    def _make_provider(self, tmp_root: Path, dates: list[str]) -> Path:
        p = tmp_root / "provider"
        (p / "calendars").mkdir(parents=True)
        (p / "calendars" / "day.txt").write_text("\n".join(dates) + "\n")
        return p

    def test_no_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._make_provider(Path(tmp), ["2024-03-01"])
            result = find_missing_dates("2024-03-01", "2024-03-01", provider)
        assert result == []

    def test_detects_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = self._make_provider(Path(tmp), ["2024-03-01"])
            result = find_missing_dates("2024-03-04", "2024-03-05", provider)
        assert "2024-03-04" in result
        assert "2024-03-05" in result

    def test_empty_calendar(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "provider"
            (p / "calendars").mkdir(parents=True)
            (p / "calendars" / "day.txt").write_text("")
            result = find_missing_dates("2024-03-01", "2024-03-01", p)
        # 2024-03-01 is a Friday trading day, not in calendar
        assert "2024-03-01" in result

    def test_weekend_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "provider"
            (p / "calendars").mkdir(parents=True)
            (p / "calendars" / "day.txt").write_text("")
            # 2024-03-02 Saturday, 2024-03-03 Sunday
            result = find_missing_dates("2024-03-02", "2024-03-03", p)
        assert result == []

    def test_result_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "provider"
            (p / "calendars").mkdir(parents=True)
            (p / "calendars" / "day.txt").write_text("")
            result = find_missing_dates("2024-03-01", "2024-03-08", p)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# _fetch_symbol
# ---------------------------------------------------------------------------


class TestFetchSymbol:
    def _make_bs_mock(self, rows: list[tuple]) -> MagicMock:
        bs = MagicMock()
        rs = MagicMock()
        rs.error_code = "0"
        side_effects = [True] * len(rows) + [False]
        rs.next.side_effect = side_effects
        rs.get_row_data.side_effect = rows
        bs.query_history_k_data_plus.return_value = rs
        return bs

    def test_normal_row(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "12.5", "12.8", "12.3", "12.6", "1000000", "0.8", "1.05")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == "2024-03-01"
        assert row["open"] == pytest.approx(12.5)
        assert row["factor"] == pytest.approx(1.05)
        assert row["change"] == pytest.approx(0.008)  # 0.8 / 100

    def test_suspended_empty_close_skipped(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "0", "0", "0", "", "0", "0", "1.0")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert rows == []

    def test_suspended_zero_close_skipped(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "0", "0", "0", "0", "0", "0", "1.0")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert rows == []

    def test_missing_adjustfactor_defaults_to_one(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "12.5", "12.8", "12.3", "12.6", "500000", "0.5", "")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert rows[0]["factor"] == pytest.approx(1.0)

    def test_missing_pct_change_defaults_to_zero(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "12.5", "12.8", "12.3", "12.6", "500000", "", "1.0")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert rows[0]["change"] == pytest.approx(0.0)

    def test_unparseable_row_skipped(self):
        bs = self._make_bs_mock(
            [("2024-03-01", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-01")
        assert rows == []

    def test_uses_qlib_to_bs_conversion(self):
        bs = self._make_bs_mock([])
        _fetch_symbol(bs, "SH600000", "2024-03-01", "2024-03-01")
        call_args = bs.query_history_k_data_plus.call_args
        assert call_args[0][0] == "sh.600000"

    def test_multiple_rows(self):
        bs = self._make_bs_mock(
            [
                ("2024-03-01", "12.5", "12.8", "12.3", "12.6", "1e6", "0.8", "1.0"),
                ("2024-03-04", "12.6", "12.9", "12.4", "12.7", "9e5", "0.79", "1.0"),
            ]
        )
        rows = _fetch_symbol(bs, "SZ000001", "2024-03-01", "2024-03-04")
        assert len(rows) == 2
        assert rows[0]["date"] == "2024-03-01"
        assert rows[1]["date"] == "2024-03-04"


# ---------------------------------------------------------------------------
# _write_csvs
# ---------------------------------------------------------------------------


class TestWriteCsvs:
    def _one_row_df(self, date: str = "2024-03-01") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": date,
                    "open": 12.5,
                    "high": 12.8,
                    "low": 12.3,
                    "close": 12.6,
                    "volume": 1_000_000.0,
                    "factor": 1.0,
                    "change": 0.008,
                }
            ]
        )

    def test_creates_per_symbol_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "csv"
            _write_csvs({"SH600000": self._one_row_df()}, dest)
            assert (dest / "SH600000.csv").exists()

    def test_csv_columns_match_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "csv"
            _write_csvs({"SH600000": self._one_row_df()}, dest)
            df = pd.read_csv(dest / "SH600000.csv")
            assert list(df.columns) == _CSV_COLUMNS

    def test_sorts_by_date(self):
        rows = pd.DataFrame(
            [
                {"date": "2024-03-05", "open": 10.0, "high": 10.5, "low": 9.8,
                 "close": 10.2, "volume": 5e5, "factor": 1.0, "change": 0.02},
                {"date": "2024-03-04", "open": 9.9, "high": 10.1, "low": 9.7,
                 "close": 10.0, "volume": 4e5, "factor": 1.0, "change": 0.01},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "csv"
            _write_csvs({"SZ000001": rows}, dest)
            df = pd.read_csv(dest / "SZ000001.csv")
        assert list(df["date"]) == ["2024-03-04", "2024-03-05"]

    def test_returns_count(self):
        data = {
            "SH600000": self._one_row_df(),
            "SZ000001": self._one_row_df(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "csv"
            n = _write_csvs(data, dest)
        assert n == 2

    def test_creates_dest_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "deep" / "csv"
            assert not dest.exists()
            _write_csvs({"SH600000": self._one_row_df()}, dest)
            assert dest.exists()


# ---------------------------------------------------------------------------
# gap_fill integration (all external calls mocked)
# ---------------------------------------------------------------------------


class TestGapFill:
    def _make_provider(self, tmp: Path, dates: list[str]) -> Path:
        p = tmp / "provider"
        (p / "calendars").mkdir(parents=True)
        (p / "calendars" / "day.txt").write_text("\n".join(dates) + "\n")
        (p / "instruments").mkdir()
        (p / "instruments" / "all.txt").write_text(
            "SH600000 2015-01-01 2999-12-31\n"
        )
        return p

    def _one_row_data(self) -> dict[str, pd.DataFrame]:
        return {
            "SH600000": pd.DataFrame(
                [
                    {
                        "date": "2024-03-01",
                        "open": 12.5,
                        "high": 12.8,
                        "low": 12.3,
                        "close": 12.6,
                        "volume": 1e6,
                        "factor": 1.0,
                        "change": 0.008,
                    }
                ]
            )
        }

    def test_returns_1_when_no_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), ["2024-03-01"])
            result = gap_fill("2024-03-01", "2024-03-01", p)
        assert result == 1

    def test_raises_when_no_instruments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "provider"
            (p / "calendars").mkdir(parents=True)
            (p / "calendars" / "day.txt").write_text("")
            with pytest.raises(RuntimeError, match="no instruments found"):
                gap_fill("2024-03-01", "2024-03-01", p)

    def test_returns_2_when_baostock_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), [])
            with patch("ashare_lab.data.fallback._fetch_all", return_value={}):
                result = gap_fill("2024-03-01", "2024-03-01", p)
        assert result == 2

    def test_returns_0_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), [])

            def fake_dump(csv_dir: Path, provider_uri: Path) -> None:
                # Simulate dump_bin updating the calendar
                (provider_uri / "calendars" / "day.txt").write_text("2024-03-01\n")

            with (
                patch("ashare_lab.data.fallback._fetch_all", return_value=self._one_row_data()),
                patch("ashare_lab.data.fallback._dump_bin_update", side_effect=fake_dump),
            ):
                result = gap_fill("2024-03-01", "2024-03-01", p)
        assert result == 0

    def test_returns_2_when_calendar_not_updated(self):
        """dump_bin runs but calendar still missing -> partial fill."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), [])
            with (
                patch("ashare_lab.data.fallback._fetch_all", return_value=self._one_row_data()),
                patch("ashare_lab.data.fallback._dump_bin_update"),  # no-op, calendar unchanged
            ):
                result = gap_fill("2024-03-01", "2024-03-01", p)
        assert result == 2

    def test_session_refresh_n_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), [])
            captured: dict = {}

            def fake_fetch(symbols, start, end, session_refresh_n):
                captured["n"] = session_refresh_n
                return {}

            fake_cfg = {"fallback": {"max_symbols_per_session": 30}}
            with (
                patch("ashare_lab.data.fallback._fetch_all", side_effect=fake_fetch),
                patch("ashare_lab.config.load_config", return_value=fake_cfg),
            ):
                gap_fill("2024-03-01", "2024-03-01", p)

        assert captured.get("n") == 30

    def test_dump_bin_raises_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_provider(Path(tmp), [])
            with (
                patch("ashare_lab.data.fallback._fetch_all", return_value=self._one_row_data()),
                patch(
                    "ashare_lab.data.fallback._dump_bin_update",
                    side_effect=RuntimeError("dump failed"),
                ),
                pytest.raises(RuntimeError, match="dump failed"),
            ):
                gap_fill("2024-03-01", "2024-03-01", p)


class TestDumpBinUpdate:
    """Direct tests for _dump_bin_update (binary append implementation)."""

    @staticmethod
    def _make_qlib_dir(tmp: Path, symbols: list[str], dates: list[str]) -> Path:
        provider = tmp / "cn_data"
        (provider / "calendars").mkdir(parents=True)
        (provider / "instruments").mkdir(parents=True)
        cal = provider / "calendars" / "day.txt"
        cal.write_text("\n".join(dates) + "\n" if dates else "")

        inst_lines = []
        for sym in symbols:
            feat = provider / "features" / sym.lower()
            feat.mkdir(parents=True)
            for field in ("open", "close", "high", "low", "volume", "factor", "change"):
                bin_path = feat / f"{field}.day.bin"
                with open(bin_path, "wb") as f:
                    for _ in dates:
                        f.write(struct.pack("<f", 1.0))
            start = dates[0] if dates else "2024-01-01"
            end = dates[-1] if dates else "2024-01-01"
            inst_lines.append(f"{sym}\t{start}\t{end}")

        (provider / "instruments" / "all.txt").write_text(
            "\n".join(inst_lines) + "\n"
        )
        return provider

    @staticmethod
    def _make_csv(csv_dir: Path, symbol: str, date: str) -> None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        (csv_dir / f"{symbol.lower()}.csv").write_text(
            "date,open,high,low,close,volume,factor,change\n"
            f"{date},10.0,11.0,9.0,10.5,1000.0,1.0,0.05\n"
        )

    def test_appends_binary_and_updates_calendar(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        provider = self._make_qlib_dir(tmp_path, ["SH600519"], ["2024-03-01"])
        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, "SH600519", "2024-03-04")

        _dump_bin_update(csv_dir, provider)

        cal = (provider / "calendars" / "day.txt").read_text().splitlines()
        assert "2024-03-04" in cal

        close_bin = provider / "features" / "sh600519" / "close.day.bin"
        data = close_bin.read_bytes()
        assert len(data) == 8  # 2 x float32
        vals = struct.unpack("<2f", data)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(10.5)

    def test_extends_instrument_end_date(self, tmp_path: Path) -> None:
        """R1: only all.txt is extended; index files are NEVER touched."""
        from ashare_lab.data.fallback import _dump_bin_update

        provider = self._make_qlib_dir(tmp_path, ["SH600519"], ["2024-03-01"])
        # Add a csi1000.txt with a stale end_date to prove it is NOT extended.
        # Use 1000 members so the gate passes (the gate runs after extension).
        csi1000 = provider / "instruments" / "csi1000.txt"
        csi_lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(999)]
        csi_lines.append("SH600519\t2020-01-01\t2023-12-31")
        csi1000.write_text("\n".join(csi_lines) + "\n", encoding="utf-8")
        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, "SH600519", "2024-03-04")

        _dump_bin_update(csv_dir, provider)

        # all.txt: end_date extended to trade date.
        all_inst = (provider / "instruments" / "all.txt").read_text()
        assert "2024-03-04" in all_inst
        # csi1000.txt: end_date UNCHANGED (index files never modified).
        csi_inst = csi1000.read_text()
        assert "2023-12-31" in csi_inst
        assert "2024-03-04" not in csi_inst

    def test_no_csvs_raises(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        provider = self._make_qlib_dir(tmp_path, [], [])
        with pytest.raises(RuntimeError, match="no CSV files"):
            _dump_bin_update(tmp_path / "empty", provider)

    def test_skips_duplicate_calendar_date(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        provider = self._make_qlib_dir(tmp_path, ["SH600519"], ["2024-03-01"])
        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, "SH600519", "2024-03-01")

        _dump_bin_update(csv_dir, provider)

        cal = (provider / "calendars" / "day.txt").read_text().splitlines()
        assert cal.count("2024-03-01") == 1

    def test_idempotent_double_call(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        provider = self._make_qlib_dir(tmp_path, ["SH600519"], ["2024-03-01"])
        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, "SH600519", "2024-03-04")

        _dump_bin_update(csv_dir, provider)
        size_after_first = (provider / "features" / "sh600519" / "close.day.bin").stat().st_size

        # Second call with same date is a no-op (idempotent).
        _dump_bin_update(csv_dir, provider)
        size_after_second = (provider / "features" / "sh600519" / "close.day.bin").stat().st_size

        assert size_after_first == size_after_second


class TestIndexMembershipGate:
    """Tests for check_index_membership (R2 + A4)."""

    def _make_instruments(self, tmp_path: Path, csi1000_lines: list[str]) -> Path:
        inst_dir = tmp_path / "instruments"
        inst_dir.mkdir()
        (inst_dir / "csi1000.txt").write_text("\n".join(csi1000_lines) + "\n", encoding="utf-8")
        return inst_dir

    def test_gate_passes_at_1000(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import check_index_membership

        lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(1000)]
        inst_dir = self._make_instruments(tmp_path, lines)
        # Should not raise.
        check_index_membership(inst_dir, "2024-07-25")

    def test_gate_fails_at_2600(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import check_index_membership

        lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(2600)]
        inst_dir = self._make_instruments(tmp_path, lines)
        with pytest.raises(ValueError, match="FAILED"):
            check_index_membership(inst_dir, "2024-07-25")

    def test_gate_fails_at_500(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import check_index_membership

        lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(500)]
        inst_dir = self._make_instruments(tmp_path, lines)
        with pytest.raises(ValueError, match="FAILED"):
            check_index_membership(inst_dir, "2024-07-25")

    def test_gate_skips_missing_index_file(self, tmp_path: Path) -> None:
        """Silently skip when csi1000.txt does not exist (test fixtures)."""
        from ashare_lab.data.fallback import check_index_membership

        inst_dir = tmp_path / "instruments"
        inst_dir.mkdir()
        # No csi1000.txt -- should not raise.
        check_index_membership(inst_dir, "2024-07-25")

    def test_gate_fires_through_dump_bin_update(self, tmp_path: Path) -> None:
        """A4: gate fires via _dump_bin_update, not only via cli."""
        from ashare_lab.data.fallback import _dump_bin_update

        # Create a qlib dir with 2600 csi1000 members (inflation scenario).
        provider = self._make_qlib_dir_inflated(tmp_path, ["SH600519"], ["2024-03-01"], 2600)
        csv_dir = tmp_path / "csvs"
        TestDumpBinUpdate._make_csv(csv_dir, "SH600519", "2024-03-04")
        with pytest.raises(ValueError, match="FAILED"):
            _dump_bin_update(csv_dir, provider)

    def _make_qlib_dir_inflated(
        self, tmp_path: Path, symbols: list[str], dates: list[str], n_members: int
    ) -> Path:
        """Build a minimal qlib provider dir with inflated csi1000 membership."""
        provider = tmp_path / "qlib"
        provider.mkdir()
        cal = provider / "calendars"
        cal.mkdir()
        (cal / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")

        inst = provider / "instruments"
        inst.mkdir()
        # all.txt: just the real symbols.
        all_lines = [f"{sym}\t{dates[0]}\t" for sym in symbols]
        (inst / "all.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        # csi1000.txt: inflated membership.
        csi_lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(n_members)]
        (inst / "csi1000.txt").write_text("\n".join(csi_lines) + "\n", encoding="utf-8")

        feat = provider / "features"
        feat.mkdir()
        for sym in symbols:
            sym_dir = feat / sym.lower()
            sym_dir.mkdir()
            for field in ["open", "close", "high", "low", "volume", "factor"]:
                (sym_dir / f"{field}.day.bin").write_bytes(b"")
        return provider

    # Reuse _make_csv from TestDumpBinUpdate (same signature, includes 'change' column).
