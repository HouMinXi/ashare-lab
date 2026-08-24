#!/bin/bash
# ashare-track-s.sh -- Daily Track S dynamic-slippage offline replay (R6)
# Runs daily after the 18:00 CST pipeline settles (proposed timer slot: 19:30 CST).
# Reads paper.db (read-only), updates shadow ledgers paper_s_<model>.db and artifacts.

REPO="$HOME/code/ashare-lab"
RUN_LOG_DIR="$REPO/logs/track_s"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="$RUN_LOG_DIR/$(date +%Y%m%d-%H%M%S).log"

if [[ "${__TRACK_S_LOGGING:-}" != "1" ]]; then
    export __TRACK_S_LOGGING=1
    "$0" "$@" 2>&1 | tee -a "$RUN_LOG"
    exit "${PIPESTATUS[0]}"
fi

set -uo pipefail

# Mutex to prevent concurrent replay runs
LOCKFILE="/tmp/ashare-track-s.lock"
exec 9<>"$LOCKFILE"
if ! flock -w 60 9; then
    echo "ERROR: another track_s replay is already running (lock: $LOCKFILE)"
    exit 1
fi

VENV="$REPO/.venv"
PYTHON="$VENV/bin/python3"
PAPER_DB="$REPO/paper.db"
SHADOW_DIR="$REPO/shadow_slippage"
ARTIFACT_DIR="$REPO/experiments/shadow_slippage"
CALIBRATION="$ARTIFACT_DIR/calibration_illiq.json"

mkdir -p "$SHADOW_DIR" "$ARTIFACT_DIR"

echo "=== Track S replay started at $(date -Iseconds) ==="

# Step 1: Check trading day
IS_TRADING=$("$PYTHON" -c "
from ashare_lab.data.calendar import is_trading_day
import datetime as dt
print(is_trading_day(dt.date.today()))
") || { echo "ERROR: failed to check trading day"; exit 1; }

FORCE=0
for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE=1
done

if [ "$IS_TRADING" != "True" ] && [ "$FORCE" -eq 0 ]; then
    echo "Non-trading day, exiting"
    exit 0
fi

# Step 2: Ensure ILLIQ calibration artifact exists
if [ ! -f "$CALIBRATION" ]; then
    echo "Generating frozen ILLIQ calibration artifact..."
    "$PYTHON" -m ashare_lab.research.calibrate_illiq \
        --end-date "2026-08-17" \
        --output "$CALIBRATION" || {
        echo "ERROR: ILLIQ calibration failed"
        exit 1
    }
fi

# Step 3: Run Replays (S0, S2, S3)
echo "Running S0 replay (Gate 1 validator)..."
"$PYTHON" -m ashare_lab.research.track_s_replay \
    --paper-db "$PAPER_DB" \
    --model S0 \
    --shadow-dir "$SHADOW_DIR" \
    --artifact-dir "$ARTIFACT_DIR" || {
    echo "ERROR: S0 replay / Gate 1 check failed"
    exit 1
}

echo "Running S2 replay (Amihud dynamic impact)..."
"$PYTHON" -m ashare_lab.research.track_s_replay \
    --paper-db "$PAPER_DB" \
    --model S2 \
    --calibration "$CALIBRATION" \
    --shadow-dir "$SHADOW_DIR" \
    --artifact-dir "$ARTIFACT_DIR" || {
    echo "ERROR: S2 replay failed"
    exit 1
}

echo "Running S3 replay (Liquidity-banded fixed slippage)..."
"$PYTHON" -m ashare_lab.research.track_s_replay \
    --paper-db "$PAPER_DB" \
    --model S3 \
    --calibration "$CALIBRATION" \
    --shadow-dir "$SHADOW_DIR" \
    --artifact-dir "$ARTIFACT_DIR" || {
    echo "ERROR: S3 replay failed"
    exit 1
}

echo "=== Track S replay finished successfully at $(date -Iseconds) ==="
exit 0
