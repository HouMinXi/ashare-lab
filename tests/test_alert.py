"""Tests for scripts/alert.py -- format, frequency, stamp, import audit."""

import ast
import importlib.util
from datetime import date, timedelta
from pathlib import Path

# Import alert.py from scripts/ (not a package)
_alert_spec = importlib.util.spec_from_file_location(
    "alert", Path(__file__).resolve().parent.parent / "scripts" / "alert.py",
)
_alert_mod = importlib.util.module_from_spec(_alert_spec)
_alert_spec.loader.exec_module(_alert_mod)

_format_alert = _alert_mod._format_alert
_should_alert = _alert_mod._should_alert
_weekdays_between = _alert_mod._weekdays_between
_write_stamp = _alert_mod._write_stamp
clear_stamp = _alert_mod.clear_stamp


class TestFormatAlert:
    def test_content(self):
        msg = _format_alert(2, "pipeline", "err line 1\nerr line 2", 45.3)
        assert "pipeline" in msg
        assert "2" in msg
        assert "45s" in msg
        assert "err line 1" in msg

    def test_empty_stderr(self):
        msg = _format_alert(1, "data_update", "", 10.0)
        assert "data_update" in msg
        assert "1" in msg
        assert "stderr" not in msg


class TestShouldAlert:
    def test_no_stamp(self, tmp_path):
        assert _should_alert(str(tmp_path / "missing.stamp")) is True

    def test_recent_stamp(self, tmp_path):
        stamp = tmp_path / "recent.stamp"
        stamp.write_text(date.today().isoformat())
        assert _should_alert(str(stamp)) is False

    def test_old_stamp(self, tmp_path):
        stamp = tmp_path / "old.stamp"
        # 5 weekdays ago -- always enough to exceed the 3-weekday threshold
        d = date.today() - timedelta(days=7)
        stamp.write_text(d.isoformat())
        assert _should_alert(str(stamp)) is True

    def test_corrupt_stamp(self, tmp_path):
        stamp = tmp_path / "bad.stamp"
        stamp.write_text("not-a-date")
        assert _should_alert(str(stamp)) is True


class TestClearStamp:
    def test_removes_file(self, tmp_path):
        stamp = tmp_path / "test.stamp"
        stamp.write_text(date.today().isoformat())
        clear_stamp(str(stamp))
        assert not stamp.exists()

    def test_missing_file_no_error(self, tmp_path):
        clear_stamp(str(tmp_path / "nonexistent.stamp"))


class TestWeekdaysBetween:
    def test_same_day(self):
        d = date(2026, 6, 29)  # Sunday
        assert _weekdays_between(d, d) == 0

    def test_monday_to_friday(self):
        mon = date(2026, 6, 29)  # Monday (2026-06-29 is Mon)
        fri = date(2026, 7, 3)
        assert _weekdays_between(mon, fri) == 4

    def test_across_weekend(self):
        thu = date(2026, 6, 25)  # Thursday
        tue = date(2026, 6, 30)  # Tuesday
        # Fri(26) + Mon(29) + Tue(30) = 3 weekdays
        assert _weekdays_between(thu, tue) == 3


class TestNoAshareImports:
    def test_stdlib_only(self):
        src = (Path(__file__).resolve().parent.parent / "scripts" / "alert.py").read_text()
        tree = ast.parse(src)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        bad = [i for i in imports if "ashare" in i or "aiohttp" in i or "requests" == i]
        assert bad == [], f"Forbidden imports: {bad}"
