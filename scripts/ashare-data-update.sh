#!/bin/bash
# ashare-data-update.sh -- Timer 1 wrapper (00:30 CST)
# Updates market data, writes atomic sentinel stamp, alerts on failure.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).
set -uo pipefail

REPO="$HOME/code/ashare-lab"
STAMP="$HOME/.cache/ashare-data-update.stamp"
STDERR_LOG="/tmp/ashare-data-update-stderr.log"
SECONDS_START=$SECONDS

mkdir -p "$HOME/.cache"
cd "$REPO" || exit 1
rm -f "$STAMP"

rc=1
for attempt in 1 2; do
    python3 -m ashare_lab.cli data update 2>"$STDERR_LOG"
    rc=$?
    if [ $rc -eq 0 ]; then
        date -Iseconds > "${STAMP}.tmp"
        mv "${STAMP}.tmp" "$STAMP"
        break
    fi
    if [ $attempt -eq 1 ]; then
        echo "Data update attempt 1 failed (rc=$rc), retrying in 5min"
        sleep 300
    fi
done

if [ $rc -ne 0 ]; then
    python3 "$REPO/scripts/alert.py" "$rc" "data_update" "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
fi

exit $rc
