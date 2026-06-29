#!/bin/bash
# ashare-pipeline.sh -- Timer 2 wrapper (01:00 CST)
# Trading day check -> GPU inference (retry-first) -> stale fallback ->
# pipeline with ASHARE_USE_STALE env -> pipeline retry -> alert.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).
set -uo pipefail

REPO="$HOME/code/ashare-lab"
GPU_HOST="192.168.100.11"
GPU_MAC="04:7C:16:49:BE:32"
GPU_USER="admin"
GPU_PREDICT_TIMEOUT=1800
DATA_STAMP="$HOME/.cache/ashare-data-update.stamp"
STDERR_LOG="/tmp/ashare-pipeline-stderr.log"
PREDICTIONS_DIR="$REPO/predictions"
SECONDS_START=$SECONDS
USE_STALE=0
PRED_ARG=""

# ---- Step 1: Trading day check (D-D13) ----

TRADE_DATE=$(python3 -c "
from ashare_lab.data.calendar import latest_trading_day
import datetime as dt
print(latest_trading_day(dt.date.today()))
") || { echo "ERROR: failed to resolve trade date"; exit 1; }

IS_TRADING=$(python3 -c "
from ashare_lab.data.calendar import is_trading_day
import datetime as dt
print(is_trading_day(dt.date.today()))
") || { echo "ERROR: failed to check trading day"; exit 1; }

# Validate TRADE_DATE format (guard against Python stderr leaking into var)
if ! [[ "$TRADE_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: invalid TRADE_DATE '$TRADE_DATE'"
    exit 1
fi

if [ "$IS_TRADING" != "True" ]; then
    echo "Not a trading day, skipping"
    exit 0
fi

# ---- Step 2: Check data stamp freshness (D-D12) ----

if [ -f "$DATA_STAMP" ]; then
    stamp_age=$(( $(date +%s) - $(date -d "$(cat "$DATA_STAMP")" +%s) ))
    if [ "$stamp_age" -gt 7200 ]; then
        echo "WARNING: data stamp is ${stamp_age}s old, proceeding with stale data"
    fi
else
    echo "WARNING: no data stamp found, proceeding anyway"
fi

# ---- Step 3: GPU inference with retry (B6: retry GPU BEFORE pipeline) ----

try_gpu_inference() {
    local GPU_START=$SECONDS

    # WoL
    wol "$GPU_MAC"

    # Ping poll: 5s interval, 40 attempts = 200s max
    local ping_ok=0
    for _i in $(seq 1 40); do
        if ping -c1 -W1 "$GPU_HOST" >/dev/null 2>&1; then
            ping_ok=1
            break
        fi
        sleep 5
    done
    if [ "$ping_ok" -eq 0 ]; then
        echo "ERROR: GPU not reachable after 200s ping poll"
        return 1
    fi

    # SSH readiness poll (H6: ping up before SSH ready on Windows)
    while ! ssh -o ConnectTimeout=3 -o BatchMode=yes "${GPU_USER}@${GPU_HOST}" "echo ok" >/dev/null 2>&1; do
        if [ $(( SECONDS - GPU_START )) -ge 300 ]; then
            echo "ERROR: SSH not ready within 300s"
            return 1
        fi
        sleep 5
    done

    # Run predict.py on GPU
    if ! timeout $GPU_PREDICT_TIMEOUT ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        "cd H:/ashare-lab && python -m ashare_lab.research.predict --date $TRADE_DATE"; then
        echo "ERROR: GPU predict.py failed or timed out"
        return 1
    fi

    # SCP predictions back
    mkdir -p "$PREDICTIONS_DIR"
    if ! scp "${GPU_USER}@${GPU_HOST}":"'H:/ashare-lab/predictions/${TRADE_DATE}.parquet'" "$PREDICTIONS_DIR/"; then
        echo "ERROR: SCP predictions failed"
        return 1
    fi

    return 0
}

# First GPU attempt
try_gpu_inference
GPU_RC=$?

# GPU retry: wait 30min, try once more (BEFORE pipeline, B6)
if [ $GPU_RC -ne 0 ]; then
    echo "GPU attempt 1 failed, retrying in 30min"
    sleep 1800
    try_gpu_inference
    GPU_RC=$?
fi

# Stale fallback: ONLY after all GPU retries exhausted
if [ $GPU_RC -ne 0 ]; then
    USE_STALE=1
    LATEST_PRED=$(find "$PREDICTIONS_DIR" -name '*.parquet' 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST_PRED" ]; then
        echo "ERROR: GPU failed and no stale predictions available"
        python3 "$REPO/scripts/alert.py" "1" "gpu_inference" "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
        exit 1
    fi
    PRED_ARG="--pred-path $LATEST_PRED"

    # RH1: alert on GPU failure BEFORE running pipeline (D-D19: stale never silent)
    python3 "$REPO/scripts/alert.py" "1" "gpu_inference" "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
fi

# ---- Step 4: Run pipeline (R3B1: ASHARE_USE_STALE env for pipeline.py) ----

# shellcheck disable=SC2086
ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG 2>"$STDERR_LOG"
PIPELINE_RC=$?

# Pipeline retry: 2 attempts total (1 initial + 1 retry, D-D14)
if [ $PIPELINE_RC -ne 0 ]; then
    echo "Pipeline attempt 1 failed (rc=$PIPELINE_RC), retrying in 30min"
    sleep 1800
    # shellcheck disable=SC2086
    ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG 2>"$STDERR_LOG"
    PIPELINE_RC=$?
fi

# ---- Step 6: Handle result ----

if [ $PIPELINE_RC -eq 0 ] && [ "$USE_STALE" -eq 0 ]; then
    # R3H1: clear stamp only when fresh predictions AND pipeline OK
    python3 "$REPO/scripts/alert.py" --clear-stamp 2>/dev/null || \
        rm -f "$HOME/.cache/ashare-alert-last.stamp"
    exit 0
fi

if [ $PIPELINE_RC -eq 0 ] && [ "$USE_STALE" -eq 1 ]; then
    # Pipeline OK with stale predictions -- do NOT clear stamp (R3H1)
    exit 0
fi

# Pipeline failure after both attempts exhausted
ELAPSED=$((SECONDS - SECONDS_START))
python3 "$REPO/scripts/alert.py" "$PIPELINE_RC" "pipeline" "$ELAPSED" "$STDERR_LOG" || true
exit $PIPELINE_RC
