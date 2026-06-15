"""Verification helpers for CSI500 instrument list and SH000300 price data."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Date threshold for "active" instrument (end_date string comparison is safe;
# qlib uses YYYY-MM-DD format so lexicographic order matches date order).
_ACTIVE_CUTOFF = "2026-01-01"
_MIN_ACTIVE_COUNT = 500


def verify_csi500_instruments(
    provider_uri: Path | None = None,
) -> dict:
    """Verify the CSI500 instrument list from the qlib data store.

    Parses provider_uri/instruments/csi500.txt (tab-separated, no header,
    columns: symbol, start_date, end_date). Counts rows where
    end_date >= "2026-01-01" (active instruments).

    Args:
        provider_uri: Path to qlib data root. If None, falls back to
            DEFAULT_PROVIDER_URI from ashare_lab.data.update.

    Returns:
        VerifyInstrumentsResult dict:
            active_count (int): rows where end_date >= "2026-01-01"
            total (int): total rows in csi500.txt
            ok (bool): active_count >= 500

    Raises:
        FileNotFoundError: if csi500.txt does not exist.
    """
    if provider_uri is None:
        from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415

        provider_uri = DEFAULT_PROVIDER_URI

    instruments_file = Path(provider_uri) / "instruments" / "csi500.txt"
    if not instruments_file.exists():
        raise FileNotFoundError(
            f"CSI500 instrument file not found: {instruments_file}"
        )

    total = 0
    active_count = 0
    with instruments_file.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            total += 1
            end_date = parts[2].strip()
            if end_date >= _ACTIVE_CUTOFF:
                active_count += 1

    ok = active_count >= _MIN_ACTIVE_COUNT
    if not ok:
        log.warning(
            "CSI500 active instrument count %d < %d (cutoff %s)",
            active_count,
            _MIN_ACTIVE_COUNT,
            _ACTIVE_CUTOFF,
        )
    return {"active_count": active_count, "total": total, "ok": ok}


def verify_sh000300_data(start_time: str, end_time: str) -> dict:
    """Verify SH000300 close price data is available in the qlib store.

    Initialises qlib (idempotent), fetches SH000300 close prices for
    [start_time, end_time] via D.features.

    Args:
        start_time: Start date string, YYYY-MM-DD.
        end_time: End date string, YYYY-MM-DD.

    Returns:
        VerifyPriceResult dict:
            rows (int): number of close price rows loaded
            first_date (str): earliest date in data (YYYY-MM-DD)
            last_date (str): latest date in data (YYYY-MM-DD)
            ok (bool): always True on success (RuntimeError raised on failure)

    Raises:
        RuntimeError: if zero rows are returned.
    """
    import qlib  # noqa: PLC0415
    from qlib.config import REG_CN  # noqa: PLC0415
    from qlib.data import D  # noqa: PLC0415

    from ashare_lab.data.update import DEFAULT_PROVIDER_URI  # noqa: PLC0415

    qlib.init(provider_uri=str(DEFAULT_PROVIDER_URI), region=REG_CN)

    df = D.features(
        instruments=["SH000300"],
        fields=["$close"],
        start_time=start_time,
        end_time=end_time,
    )

    if df is None or len(df) == 0:
        raise RuntimeError(
            f"SH000300 close price data returned zero rows for "
            f"{start_time} to {end_time}"
        )

    # D.features returns MultiIndex(datetime, instrument); reset to get dates.
    dates = df.index.get_level_values("datetime")
    first_date = str(dates.min().date())
    last_date = str(dates.max().date())
    rows = len(df)

    return {"rows": rows, "first_date": first_date, "last_date": last_date, "ok": True}
