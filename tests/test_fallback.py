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
                    # production layout: a float32 holding the calendar
                    # index of the first point, then one per point
                    f.write(struct.pack("<f", 0.0))
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
        assert len(data) == 12  # header + 2 x float32
        vals = struct.unpack("<2f", data[4:])  # skip the start-index header
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(10.5)

    def test_extends_instrument_end_date(self, tmp_path: Path) -> None:
        """Open-row-only: both all.txt and index files are extended."""
        from ashare_lab.data.fallback import _dump_bin_update

        symbols = [f"SH{i:06d}" for i in range(1000)]
        provider = self._make_qlib_dir(tmp_path, symbols, ["2024-03-01"])
        # Add a csi1000.txt with open rows (end == max_end = 2023-12-31).
        csi1000 = provider / "instruments" / "csi1000.txt"
        csi_lines = [f"{sym}\t2020-01-01\t2023-12-31" for sym in symbols]
        csi1000.write_text("\n".join(csi_lines) + "\n", encoding="utf-8")
        csv_dir = tmp_path / "csvs"
        for sym in symbols:
            self._make_csv(csv_dir, sym, "2024-03-04")

        _dump_bin_update(csv_dir, provider)

        # all.txt: open rows (end == max_end) extended to trade date.
        all_inst = (provider / "instruments" / "all.txt").read_text()
        assert "2024-03-04" in all_inst
        # csi1000.txt: open rows also extended.
        csi_inst = csi1000.read_text()
        assert "2024-03-04" in csi_inst

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


class TestOpenRowExtension:
    """T1-T4: open-row-only end-date extension (C2 work order)."""

    @staticmethod
    def _make_inst_dir(tmp_path: Path, all_lines: list[str],
                       index_lines: list[str] | None = None) -> Path:
        inst_dir = tmp_path / "instruments"
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / "all.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        if index_lines is not None:
            (inst_dir / "csi1000.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        return inst_dir

    def test_open_row_extended_closed_row_unchanged(self, tmp_path: Path) -> None:
        """T1: open rows (end == max_end) are extended; closed rows (end < max_end) are not."""
        from ashare_lab.data.fallback import _extend_instrument_end_dates

        all_lines = [
            "SYM_A\t2020-01-01\t2026-07-23",  # open (end == max_end)
            "SYM_B\t2020-01-01\t2026-06-15",  # closed (end < max_end)
        ]
        index_lines = [
            "SYM_A\t2020-01-01\t2026-07-23",  # open
            "SYM_B\t2020-01-01\t2026-06-15",  # closed
        ]
        inst_dir = self._make_inst_dir(tmp_path, all_lines, index_lines)

        _extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-24")

        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in all_txt  # extended
        assert "SYM_B\t2020-01-01\t2026-06-15" in all_txt  # unchanged

        csi_txt = (inst_dir / "csi1000.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in csi_txt  # extended
        assert "SYM_B\t2020-01-01\t2026-06-15" in csi_txt  # unchanged

    def test_chained_extension(self, tmp_path: Path) -> None:
        """T2: chained extension 07-23 -> 07-24 -> 07-27."""
        from ashare_lab.data.fallback import _extend_instrument_end_dates

        lines = ["SYM_A\t2020-01-01\t2026-07-23"]
        inst_dir = self._make_inst_dir(tmp_path, list(lines), list(lines))

        _extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-24")
        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in all_txt

        _extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-27")
        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-27" in all_txt

        csi_txt = (inst_dir / "csi1000.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-27" in csi_txt

    def test_bug_injection_extend_all_rows(self, tmp_path: Path) -> None:
        """T3: bug-injection -- revert to 'extend all end < latest' and watch T1 fail."""
        from ashare_lab.data import fallback as fb

        all_lines = [
            "SYM_A\t2020-01-01\t2026-07-23",  # open
            "SYM_B\t2020-01-01\t2026-06-15",  # closed
        ]
        inst_dir = self._make_inst_dir(tmp_path, all_lines, None)

        # Inject bug: replace open-row-only with extend-all-end<latest.
        orig = fb._extend_open_rows_in_file
        def _buggy_extend(inst_file, traded_syms, latest):
            lines = inst_file.read_text(encoding="utf-8").splitlines()
            new_lines = []
            changed = False
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2] < latest:
                    parts[2] = latest
                    changed = True
                new_lines.append("\t".join(parts))
            if changed:
                inst_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        fb._extend_open_rows_in_file = _buggy_extend

        # Run buggy code and check result directly (no try/except).
        fb._extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-24")
        all_txt = (inst_dir / "all.txt").read_text()
        bug_detected = "SYM_B\t2020-01-01\t2026-07-24" in all_txt
        fb._extend_open_rows_in_file = orig

        # Restore fixture.
        (inst_dir / "all.txt").write_text(
            "\n".join(all_lines) + "\n", encoding="utf-8"
        )

        # Verify bug was detected (closed row was extended by buggy code).
        assert bug_detected, "bug injection did not trigger: closed row was not extended"

        # Verify correct behavior after restore.
        fb._extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-24")
        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in all_txt  # open row extended
        assert "SYM_B\t2020-01-01\t2026-06-15" in all_txt  # closed row preserved

    def test_empty_end_date_treated_as_open(self, tmp_path: Path) -> None:
        """Rows with empty end_date (still active) are extended."""
        from ashare_lab.data.fallback import _extend_instrument_end_dates

        all_lines = [
            "SYM_A\t2020-01-01\t",           # empty end = still active
            "SYM_B\t2020-01-01\t2026-07-23",  # explicit open
        ]
        inst_dir = self._make_inst_dir(tmp_path, all_lines, None)

        _extend_instrument_end_dates(inst_dir, {"SYM_A", "SYM_B"}, "2026-07-24")

        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in all_txt  # empty -> extended
        assert "SYM_B\t2020-01-01\t2026-07-24" in all_txt  # open -> extended

    def test_not_traded_symbol_not_extended(self, tmp_path: Path) -> None:
        """T7: symbol with end == max_end but NOT in traded_syms is not extended.

        This prevents extending a stock removed on the bundle date:
        the removed stock is suspended, not in traded_syms, so its
        open interval is left as-is.
        """
        from ashare_lab.data.fallback import _extend_instrument_end_dates

        all_lines = [
            "SYM_A\t2020-01-01\t2026-07-23",  # open, traded today
            "SYM_B\t2020-01-01\t2026-07-23",  # open, NOT traded (removed/suspended)
        ]
        inst_dir = self._make_inst_dir(tmp_path, all_lines, None)

        # Only SYM_A is in traded_syms.
        _extend_instrument_end_dates(inst_dir, {"SYM_A"}, "2026-07-24")

        all_txt = (inst_dir / "all.txt").read_text()
        assert "SYM_A\t2020-01-01\t2026-07-24" in all_txt  # traded -> extended
        assert "SYM_B\t2020-01-01\t2026-07-23" in all_txt  # not traded -> unchanged

    def test_early_return_path_extends(self, tmp_path: Path) -> None:
        """T4: early-return path in _dump_bin_update still extends open rows."""
        from ashare_lab.data.fallback import _dump_bin_update

        symbols = [f"SH{i:06d}" for i in range(1000)]

        # Create qlib dir with date 2024-03-01 already in calendar.
        provider = tmp_path / "cn_data"
        (provider / "calendars").mkdir(parents=True)
        (provider / "instruments").mkdir(parents=True)
        (provider / "calendars" / "day.txt").write_text("2024-03-01\n", encoding="utf-8")

        # all.txt with open rows.
        all_lines = [f"{sym}\t2020-01-01\t2024-03-01" for sym in symbols]
        (provider / "instruments" / "all.txt").write_text(
            "\n".join(all_lines) + "\n", encoding="utf-8"
        )
        # csi1000.txt with 1000 open rows (gate expects [990, 1010]).
        csi_lines = [f"{sym}\t2020-01-01\t2024-03-01" for sym in symbols]
        (provider / "instruments" / "csi1000.txt").write_text(
            "\n".join(csi_lines) + "\n", encoding="utf-8"
        )

        # Create feature dirs and binary files for all symbols.
        for sym in symbols:
            feat = provider / "features" / sym.lower()
            feat.mkdir(parents=True)
            for field in ("open", "close", "high", "low", "volume", "factor", "change"):
                with open(feat / f"{field}.day.bin", "wb") as f:
                    f.write(struct.pack("<f", 1.0))

        # CSV with SAME date (2024-03-01) -- dates_to_add will be empty,
        # triggering the early-return path.
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        for sym in symbols:
            (csv_dir / f"{sym.lower()}.csv").write_text(
                "date,open,high,low,close,volume,factor,change\n"
                "2024-03-01,10.0,11.0,9.0,10.5,1000.0,1.0,0.05\n",
                encoding="utf-8",
            )

        # First call: early-return (calendar already has 2024-03-01).
        _dump_bin_update(csv_dir, provider)

        # Now manually set end dates back to simulate a stale state
        # (e.g., after a bundle swap that reset instrument files).
        stale_lines = [f"{sym}\t2020-01-01\t2024-02-28" for sym in symbols]
        (provider / "instruments" / "all.txt").write_text(
            "\n".join(stale_lines) + "\n", encoding="utf-8"
        )
        (provider / "instruments" / "csi1000.txt").write_text(
            "\n".join(stale_lines) + "\n", encoding="utf-8"
        )

        # Second call: early-return (dates_to_add empty) but open rows
        # need extension (max_end=02-28 < latest=03-01).
        _dump_bin_update(csv_dir, provider)

        # Verify: open rows should be extended on early-return path.
        all_txt = (provider / "instruments" / "all.txt").read_text()
        assert "2024-03-01" in all_txt  # extended from 02-28 to 03-01
        csi_txt = (provider / "instruments" / "csi1000.txt").read_text()
        assert "2024-03-01" in csi_txt  # index file also extended


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

    def test_gate_fails_at_1020(self, tmp_path: Path) -> None:
        """Tight tolerance (±10): 1020 is outside [990, 1010]."""
        from ashare_lab.data.fallback import check_index_membership

        lines = [f"SH{i:06d}\t2020-01-01\t" for i in range(1020)]
        inst_dir = self._make_instruments(tmp_path, lines)
        with pytest.raises(ValueError, match="FAILED"):
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


# ---------------------------------------------------------------------------
# backfill_missing_dates
# ---------------------------------------------------------------------------


class TestBackfillMissingDates:
    def _provider(self, root: Path, dates: list[str]) -> Path:
        p = root / "provider"
        (p / "calendars").mkdir(parents=True)
        (p / "calendars" / "day.txt").write_text("\n".join(dates) + "\n")
        (p / "features").mkdir()
        (p / "instruments").mkdir()
        return p

    def test_fills_gap_and_keeps_calendar_sorted(self, monkeypatch):
        import datetime as dt

        from ashare_lab.data.fallback import backfill_missing_dates

        got: list[str] = []

        def fake_fetch(trade_date: str, provider_uri=None) -> dict:
            got.append(trade_date)
            return {"sz000001": pd.DataFrame([{
                "date": trade_date, "open": 1.0, "high": 1.0, "low": 1.0,
                "close": 1.0, "volume": 1.0, "factor": 1.0, "change": 0.0,
            }])}

        def fake_dump(csv_dir: Path, provider_uri: Path) -> None:
            cal = provider_uri / "calendars" / "day.txt"
            have = [ln for ln in cal.read_text().splitlines() if ln.strip()]
            cal.write_text("\n".join(sorted(set(have) | set(got))) + "\n")

        monkeypatch.setattr(
            "ashare_lab.data.fetcher.fetch_today_data", fake_fetch)
        monkeypatch.setattr(
            "ashare_lab.data.fallback._dump_bin_update", fake_dump)

        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(
                Path(tmp), ["2024-03-01", "2024-03-05"])
            filled = backfill_missing_dates(
                dt.date(2024, 3, 1), dt.date(2024, 3, 5), provider)
            calendar = (provider / "calendars" / "day.txt").read_text().splitlines()

        assert filled == ["2024-03-04"]
        assert got == ["2024-03-04"]
        assert calendar == sorted(calendar)
        assert "2024-03-04" in calendar

    def test_failed_day_does_not_undo_filled_days(self, monkeypatch):
        import datetime as dt

        from ashare_lab.data.fallback import backfill_missing_dates

        def fake_fetch(trade_date: str, provider_uri=None) -> dict:
            if trade_date == '2024-03-04':
                raise RuntimeError('token locked')
            return {'sz000001': pd.DataFrame([{
                'date': trade_date, 'open': 1.0, 'high': 1.0, 'low': 1.0,
                'close': 1.0, 'volume': 1.0, 'factor': 1.0, 'change': 0.0,
            }])}

        def fake_dump(csv_dir: Path, provider_uri: Path) -> None:
            cal = provider_uri / 'calendars' / 'day.txt'
            have = {ln.strip() for ln in cal.read_text().splitlines() if ln.strip()}
            added = set()
            for csv in csv_dir.glob('*.csv'):
                added.update(line.split(',')[0] for line in csv.read_text().splitlines()[1:])
            cal.write_text('\n'.join(sorted(have | added)) + '\n')

        monkeypatch.setattr('ashare_lab.data.fetcher.fetch_today_data', fake_fetch)
        monkeypatch.setattr('ashare_lab.data.fallback._dump_bin_update', fake_dump)

        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(Path(tmp), ['2024-03-01'])
            filled = backfill_missing_dates(
                dt.date(2024, 3, 1), dt.date(2024, 3, 5), provider)
            calendar = (provider / 'calendars' / 'day.txt').read_text().splitlines()

        assert filled == ['2024-03-05']
        assert '2024-03-05' in calendar
        assert '2024-03-04' not in calendar

    def test_calendar_rewrite_does_not_write_day_txt_directly(self, monkeypatch):
        import datetime as dt

        from ashare_lab.data.fallback import backfill_missing_dates

        def fake_fetch(trade_date: str, provider_uri=None) -> dict:
            return {"sz000001": pd.DataFrame([{
                "date": trade_date, "open": 1.0, "high": 1.0, "low": 1.0,
                "close": 1.0, "volume": 1.0, "factor": 1.0, "change": 0.0}])}

        real_write = Path.write_text

        def guard(self, *a, **k):
            if self.name == "day.txt":
                raise OSError("disk full")
            return real_write(self, *a, **k)

        monkeypatch.setattr("ashare_lab.data.fetcher.fetch_today_data", fake_fetch)

        with tempfile.TemporaryDirectory() as tmp:
            provider = self._provider(Path(tmp), ["2024-03-05", "2024-03-01"])
            (provider / "features" / "sz000001").mkdir(parents=True)
            for field in ["open", "close", "high", "low", "volume", "factor"]:
                (provider / "features" / "sz000001" / f"{field}.day.bin").write_bytes(
                    bytes(4))
            (provider / "instruments" / "all.txt").write_text(
                "SZ000001\t2024-03-01\t2024-03-01\n")
            before = (provider / "calendars" / "day.txt").read_bytes()
            monkeypatch.setattr(Path, "write_text", guard)
            backfill_missing_dates(
                dt.date(2024, 3, 1), dt.date(2024, 3, 5), provider)
            after = (provider / "calendars" / "day.txt").read_bytes()
        assert after != before
        assert b"2024-03-04" in after


# ---------------------------------------------------------------------------
# Index instruments
# ---------------------------------------------------------------------------


class TestFetchSymbolIndexMode:
    """baostock rejects the adjustfactor field for index instruments."""

    def _make_bs(self, rows):
        bs = MagicMock()
        rs = MagicMock()
        rs.error_code = "0"
        rs.next.side_effect = [True] * len(rows) + [False]
        rs.get_row_data.side_effect = list(rows)
        bs.query_history_k_data_plus.return_value = rs
        return bs

    def test_index_request_omits_adjustfactor(self):
        bs = self._make_bs([])
        _fetch_symbol(bs, "SH000852", "2024-03-01", "2024-03-01", index=True)
        fields = bs.query_history_k_data_plus.call_args.kwargs["fields"]
        assert "adjustfactor" not in fields

    def test_stock_request_keeps_adjustfactor(self):
        bs = self._make_bs([])
        _fetch_symbol(bs, "SH600000", "2024-03-01", "2024-03-01")
        fields = bs.query_history_k_data_plus.call_args.kwargs["fields"]
        assert "adjustfactor" in fields

    def test_index_row_carries_unit_factor(self):
        rows = [
            ["2024-03-01", "6700.0", "6750.0", "6600.0", "6630.7753", "1000", "1.5"]
        ]
        bs = self._make_bs(rows)
        out = _fetch_symbol(bs, "SH000852", "2024-03-01", "2024-03-01", index=True)
        assert len(out) == 1
        assert out[0]["factor"] == 1.0
        assert out[0]["close"] == pytest.approx(6630.7753)
        assert out[0]["change"] == pytest.approx(0.015)


class TestInstrumentEndDates:
    def test_reads_symbol_end_dates(self):
        from ashare_lab.data.fallback import _instrument_end_dates

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "instruments").mkdir()
            (root / "instruments" / "all.txt").write_text(
                "SH000852\t2005-01-04\t2026-10-08\n"
                "SZ000001\t2000-01-04\t2026-10-26\n"
            )
            got = _instrument_end_dates(root)
        assert got["SH000852"] == "2026-10-08"
        assert got["SZ000001"] == "2026-10-26"

    def test_missing_file_gives_empty_mapping(self):
        from ashare_lab.data.fallback import _instrument_end_dates

        with tempfile.TemporaryDirectory() as td:
            assert _instrument_end_dates(Path(td)) == {}


class TestBackfillStaleIndexes:
    """An index only refreshes on a full package swap, so it falls behind."""

    def _write_instruments(self, root, end_dates):
        (root / "instruments").mkdir(parents=True, exist_ok=True)
        body = "".join(f"{s}\t2005-01-04\t{d}\n" for s, d in end_dates.items())
        (root / "instruments" / "all.txt").write_text(body)

    def _frame(self, date):
        return pd.DataFrame(
            [[date, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 0.0]], columns=list(_CSV_COLUMNS)
        )

    def test_fills_index_that_lags_behind(self):
        from ashare_lab.data.fallback import backfill_stale_indexes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_instruments(
                root, {"SH000852": "2026-10-08", "SZ000001": "2026-10-26"}
            )
            with patch(
                "ashare_lab.data.fallback._fetch_all",
                return_value={"SH000852": self._frame("2026-10-09")},
            ) as fetch, patch(
                "ashare_lab.data.fallback._dump_bin_update"
            ) as dump:
                filled = backfill_stale_indexes("2026-10-26", provider_uri=root)

        assert filled == ["SH000852"]
        assert dump.called
        args, kwargs = fetch.call_args
        assert list(args[0]) == ["SH000852"], "only the lagging index is fetched"
        assert args[1] == "2026-10-09", "starts the day after the stored end date"
        assert args[2] == "2026-10-26"
        assert args[3] == len(args[0]), "one login covers the batch"
        assert kwargs.get("index") is True, "index mode must reach _fetch_all"

    def test_one_login_covers_the_whole_batch(self):
        """Fetching every stale index takes about two seconds, inside the
        baostock idle timeout, so re-logging in per symbol only adds cost."""
        from ashare_lab.data.fallback import backfill_stale_indexes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_instruments(
                root,
                {
                    "SH000852": "2026-10-08",
                    "SH000300": "2026-10-08",
                    "SZ000001": "2026-10-26",
                },
            )
            with patch(
                "ashare_lab.data.fallback._fetch_all",
                return_value={"SH000852": self._frame("2026-10-09")},
            ) as fetch, patch("ashare_lab.data.fallback._dump_bin_update"):
                backfill_stale_indexes("2026-10-26", provider_uri=root)

        args, _ = fetch.call_args
        assert sorted(args[0]) == ["SH000300", "SH000852"], "both stale indexes"
        assert args[3] == 2, "the interval spans the batch, not one symbol"

    def test_skips_index_already_current(self):
        from ashare_lab.data.fallback import backfill_stale_indexes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_instruments(root, {"SH000852": "2026-10-26"})
            with patch("ashare_lab.data.fallback._fetch_all") as fetch, patch(
                "ashare_lab.data.fallback._dump_bin_update"
            ) as dump:
                filled = backfill_stale_indexes("2026-10-26", provider_uri=root)

        assert filled == []
        assert not fetch.called
        assert not dump.called

    def test_empty_fetch_writes_nothing(self):
        from ashare_lab.data.fallback import backfill_stale_indexes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_instruments(root, {"SH000852": "2026-10-08"})
            with patch(
                "ashare_lab.data.fallback._fetch_all", return_value={}
            ), patch("ashare_lab.data.fallback._dump_bin_update") as dump:
                filled = backfill_stale_indexes("2026-10-26", provider_uri=root)

        assert filled == []
        assert not dump.called

    def test_unknown_symbol_is_not_fetched(self):
        from ashare_lab.data.fallback import backfill_stale_indexes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_instruments(root, {"SZ000001": "2026-10-26"})
            with patch("ashare_lab.data.fallback._fetch_all") as fetch:
                filled = backfill_stale_indexes("2026-10-26", provider_uri=root)

        assert filled == []
        assert not fetch.called


class TestDumpBinTrailingSymbol:
    """A symbol can lag the calendar that its peers moved forward.

    Unlike the helper in TestDumpBinUpdate, these bins carry the production
    layout: a leading float32 holding the calendar index of the first point.
    The append path never reads that header, but deciding whether a symbol
    has fallen behind does.
    """

    @staticmethod
    def _make_provider(
        root: Path,
        calendar: list[str],
        points: int,
        start_index: int = 0,
        field_points: dict[str, int] | None = None,
    ) -> Path:
        (root / "calendars").mkdir(parents=True)
        (root / "calendars" / "day.txt").write_text(
            "\n".join(calendar) + "\n", encoding="utf-8"
        )
        (root / "instruments").mkdir()
        first = calendar[start_index]
        last = calendar[start_index + points - 1]
        (root / "instruments").joinpath("all.txt").write_text(
            f"SH000852\t{first}\t{last}\n", encoding="utf-8"
        )
        feat = root / "features" / "sh000852"
        feat.mkdir(parents=True)
        fields = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "factor",
            "change",
            "adjclose",
            "amount",
            "vwap",
        )
        for field in fields:
            n = (field_points or {}).get(field, points)
            with open(feat / f"{field}.day.bin", "wb") as f:
                f.write(struct.pack("<f", float(start_index)))
                for _ in range(n):
                    f.write(struct.pack("<f", 1.0))
        return root

    @staticmethod
    def _make_csv(csv_dir: Path, dates: list[str]) -> None:
        """Write the csv the way _write_csvs does: upper-case symbol name,
        while the feature directory it must reach is lower case."""
        csv_dir.mkdir(parents=True, exist_ok=True)
        head = "date,open,high,low,close,volume,factor,change\n"
        body = "".join(f"{d},2,2,2,2,200,1.0,0.0\n" for d in dates)
        (csv_dir / "SH000852.csv").write_text(head + body, encoding="utf-8")

    def test_appends_rows_a_trailing_symbol_is_missing(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04", "2024-03-05"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        bin_path = provider / "features" / "sh000852" / "close.day.bin"
        before = bin_path.stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-04", "2024-03-05"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert bin_path.stat().st_size == before + 8, "two points must land"
        cal_after = (
            (provider / "calendars" / "day.txt").read_text().splitlines()
        )
        assert cal_after == cal, "the calendar already had them; no duplicates"

    def test_leaves_a_current_symbol_alone(self, tmp_path: Path) -> None:
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=2)
        bin_path = provider / "features" / "sh000852" / "close.day.bin"
        before = bin_path.stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-04"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert bin_path.stat().st_size == before, "nothing to catch up"

    def test_header_offset_decides_whether_rows_are_missing(
        self, tmp_path: Path
    ) -> None:
        """The header holds the calendar index of the first point. Ignoring
        it makes a symbol that already reaches the calendar end look like it
        trails, and the rows get appended a second time."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = [
            "2024-03-01",
            "2024-03-04",
            "2024-03-05",
            "2024-03-06",
            "2024-03-07",
        ]
        provider = self._make_provider(
            tmp_path / "qlib", cal, points=2, start_index=3
        )
        bin_path = provider / "features" / "sh000852" / "close.day.bin"
        before = bin_path.stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-07"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert bin_path.stat().st_size == before, "already at the calendar end"

    def test_trails_when_a_single_field_lags(self, tmp_path: Path) -> None:
        """A run that died mid-write leaves some fields short of the others.
        Judging only by close would skip the symbol and strand volume."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04", "2024-03-05"]
        provider = self._make_provider(
            tmp_path / "qlib", cal, points=3, field_points={"volume": 1}
        )
        vol_path = provider / "features" / "sh000852" / "volume.day.bin"
        close_path = provider / "features" / "sh000852" / "close.day.bin"
        vol_before = vol_path.stat().st_size
        close_before = close_path.stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-04", "2024-03-05"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert vol_path.stat().st_size > vol_before, "the lagging field must fill"
        assert close_path.stat().st_size == close_before, (
            "close already holds those days and must not grow"
        )

    def test_repeat_run_does_not_shift_a_current_field(
        self, tmp_path: Path
    ) -> None:
        """A production stock keeps fields the daily fetch never writes, so
        adjclose sits weeks behind close. A repeat run must not read that as
        licence to append the already-current fields a second time, which
        would push every later value onto the wrong day."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04", "2024-03-05"]
        provider = self._make_provider(
            tmp_path / "qlib", cal, points=3, field_points={"adjclose": 1}
        )
        feat = provider / "features" / "sh000852"
        close_before = (feat / "close.day.bin").stat().st_size
        vol_before = (feat / "volume.day.bin").stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-05"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert (feat / "close.day.bin").stat().st_size == close_before, (
            "close already holds 03-05"
        )
        assert (feat / "volume.day.bin").stat().st_size == vol_before, (
            "volume already holds 03-05"
        )

    def test_unmaintained_field_does_not_force_a_write(
        self, tmp_path: Path
    ) -> None:
        """adjclose, amount and vwap arrive with the full package and are
        never written here, so they sit weeks behind. Letting them vote on
        whether a symbol trails would wake the writer for rows it cannot
        place, leaving it with nothing to do."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04", "2024-03-05"]
        provider = self._make_provider(
            tmp_path / "qlib", cal, points=3, field_points={"adjclose": 1}
        )
        feat = provider / "features" / "sh000852"
        sizes = {
            b.name: b.stat().st_size for b in sorted(feat.glob("*.day.bin"))
        }
        # 03-04 is exactly the slot adjclose is waiting for, and no
        # maintained field wants it
        self._make_csv(tmp_path / "csvs", ["2024-03-04"])

        _dump_bin_update(tmp_path / "csvs", provider)

        after = {b.name: b.stat().st_size for b in sorted(feat.glob("*.day.bin"))}
        assert after == sizes, "every field already holds 03-04 or is not ours"

    def test_a_blank_calendar_line_does_not_shift_positions(
        self, tmp_path: Path
    ) -> None:
        """Bin slots are calendar positions, so a blank line in the calendar
        file would offset every date after it and park values on the wrong
        day. qlib skips blank lines when it builds those positions, and so
        must every reader here."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "", "2024-03-04"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        close_path = provider / "features" / "sh000852" / "close.day.bin"
        before = close_path.stat().st_size
        self._make_csv(tmp_path / "csvs", ["2024-03-04"])

        _dump_bin_update(tmp_path / "csvs", provider)

        assert close_path.stat().st_size > before, (
            "03-04 sits at position 1 once the blank line is skipped"
        )

    def test_end_date_tracks_what_was_written(self, tmp_path: Path) -> None:
        """Advertising a date the bin does not hold turns a lookup into a
        silent miss, so the stored end date must follow the data."""
        from ashare_lab.data.fallback import _dump_bin_update, _instrument_end_dates

        cal = ["2024-03-01", "2024-03-04", "2024-03-05"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        self._make_csv(tmp_path / "csvs", ["2024-03-04", "2024-03-05"])

        _dump_bin_update(tmp_path / "csvs", provider)

        bin_path = provider / "features" / "sh000852" / "close.day.bin"
        points = (bin_path.stat().st_size - 4) // 4
        assert _instrument_end_dates(provider)["SH000852"] == cal[points - 1]

    def test_a_symbol_without_a_directory_is_not_marked_as_updated(
        self, tmp_path: Path
    ) -> None:
        """An index row for a symbol whose features have not landed must keep its
        own end date: rolling it forward claims data that does not exist, and the
        stale-index repair reads exactly this file to decide what is current."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        all_txt = provider / "instruments" / "all.txt"
        all_txt.write_text(
            all_txt.read_text(encoding="utf-8")
            + "SZ999999\t2024-03-01\t2024-03-01\n",
            encoding="utf-8",
        )

        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, ["2024-03-04"])
        head = "date,open,high,low,close,volume,factor,change\n"
        (csv_dir / "SZ999999.csv").write_text(
            head + "2024-03-04,2,2,2,2,200,1.0,0.0\n", encoding="utf-8"
        )

        _dump_bin_update(csv_dir, provider)

        rows = {
            line.split("\t")[0]: line.split("\t")[1:]
            for line in all_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert rows["SH000852"][1] == "2024-03-04", "the written symbol advances"
        assert rows["SZ999999"][1] == "2024-03-01", (
            "nothing was written for it, so its end date must not move"
        )

    def test_a_csv_without_a_feature_directory_is_skipped(self, tmp_path: Path) -> None:
        """A symbol the provider has no directory for cannot be written, but it
        must not stop the symbols that can."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01", "2024-03-04"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        csv_dir = tmp_path / "csvs"
        self._make_csv(csv_dir, ["2024-03-04"])
        head = "date,open,high,low,close,volume,factor,change\n"
        (csv_dir / "SZ999999.csv").write_text(
            head + "2024-03-04,2,2,2,2,200,1.0,0.0\n", encoding="utf-8"
        )

        _dump_bin_update(csv_dir, provider)

        bin_path = provider / "features" / "sh000852" / "close.day.bin"
        assert (bin_path.stat().st_size - 4) // 4 == 2, "the symbol with a directory still got its row"

    def test_a_run_with_nothing_to_write_raises(self, tmp_path: Path) -> None:
        """If every CSV lacks a feature directory the run wrote nothing; that is
        a broken provider layout, not a quiet success."""
        from ashare_lab.data.fallback import _dump_bin_update

        # The date must be new to the calendar, otherwise the function returns
        # early and never reaches the write loop this test is about.
        cal = ["2024-03-01"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        head = "date,open,high,low,close,volume,factor,change\n"
        (csv_dir / "SZ999999.csv").write_text(
            head + "2024-03-04,2,2,2,2,200,1.0,0.0\n", encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="no instruments updated"):
            _dump_bin_update(csv_dir, provider)

    def test_a_failed_run_leaves_the_calendar_untouched(self, tmp_path: Path) -> None:
        """The dates are appended before the write loop, so a run that writes
        nothing must put them back. Otherwise the next run finds every date
        present, skips the loop, and reports success over missing data."""
        from ashare_lab.data.fallback import _dump_bin_update

        cal = ["2024-03-01"]
        provider = self._make_provider(tmp_path / "qlib", cal, points=1)
        day_txt = provider / "calendars" / "day.txt"
        before = day_txt.read_bytes()

        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        head = "date,open,high,low,close,volume,factor,change\n"
        (csv_dir / "SZ999999.csv").write_text(
            head + "2024-03-04,2,2,2,2,200,1.0,0.0\n", encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="no instruments updated"):
            _dump_bin_update(csv_dir, provider)
        assert day_txt.read_bytes() == before, "the calendar must not move"

        with pytest.raises(RuntimeError, match="no instruments updated"):
            _dump_bin_update(csv_dir, provider)
