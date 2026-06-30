"""Verify systemd timer schedules and WorkingDirectory in deploy.sh."""

import os
import re
import stat
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parent.parent / "scripts" / "deploy.sh"
CHENDITC_SH = Path(__file__).resolve().parent.parent / "scripts" / "ashare-chenditc.sh"


@pytest.fixture(scope="module")
def deploy_text() -> str:
    return DEPLOY_SH.read_text()


class TestTimerSchedules:
    """OnCalendar values must match D-72-01 schedule."""

    def test_timer1_17_00(self, deploy_text: str) -> None:
        # Timer 1 description + OnCalendar block
        m = re.search(
            r"ashare-data-update\.timer.*?OnCalendar=\*-\*-\*\s+(\d{2}:\d{2}:\d{2})",
            deploy_text,
            re.DOTALL,
        )
        assert m is not None, "ashare-data-update.timer OnCalendar not found"
        assert m.group(1) == "17:00:00"

    def test_timer2_18_00(self, deploy_text: str) -> None:
        m = re.search(
            r"ashare-pipeline\.timer.*?OnCalendar=\*-\*-\*\s+(\d{2}:\d{2}:\d{2})",
            deploy_text,
            re.DOTALL,
        )
        assert m is not None, "ashare-pipeline.timer OnCalendar not found"
        assert m.group(1) == "18:00:00"

    def test_timer3_21_00(self, deploy_text: str) -> None:
        m = re.search(
            r"ashare-chenditc\.timer.*?OnCalendar=\*-\*-\*\s+(\d{2}:\d{2}:\d{2})",
            deploy_text,
            re.DOTALL,
        )
        assert m is not None, "ashare-chenditc.timer OnCalendar not found"
        assert m.group(1) == "21:00:00"


class TestWorkingDirectory:
    """All three service units must have WorkingDirectory."""

    @pytest.mark.parametrize(
        "service_name",
        [
            "ashare-data-update.service",
            "ashare-pipeline.service",
            "ashare-chenditc.service",
        ],
    )
    def test_working_directory_present(self, deploy_text: str, service_name: str) -> None:
        # Extract the service unit block (from its cat line to next UNIT delimiter)
        pattern = re.compile(
            rf'cat > .*{re.escape(service_name)}.*?<<\s*UNIT\n(.*?)\nUNIT',
            re.DOTALL,
        )
        m = pattern.search(deploy_text)
        assert m is not None, f"{service_name} block not found in deploy.sh"
        assert "WorkingDirectory=" in m.group(1), (
            f"{service_name} missing WorkingDirectory"
        )


class TestChenditcScript:
    """ashare-chenditc.sh must exist and be executable."""

    def test_exists(self) -> None:
        assert CHENDITC_SH.exists()

    def test_executable(self) -> None:
        mode = CHENDITC_SH.stat().st_mode
        assert mode & stat.S_IXUSR, "ashare-chenditc.sh not executable"
