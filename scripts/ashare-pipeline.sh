#!/bin/bash
# ashare-pipeline.sh -- Timer 2 wrapper (18:00 CST)
# Trading day check -> GPU inference (retry-first) -> stale fallback ->
# pipeline with ASHARE_USE_STALE env -> pipeline retry -> alert.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).

# Script self-invocation: tee all output to per-run log with reliable exit codes.
# exec > >(tee ...) is unreliable -- pipefail doesn't cover process substitution,
# so tee failures are invisible.  Self-invocation wraps the whole script in a
# pipeline where pipefail DOES apply.
REPO="$HOME/code/ashare-lab"
RUN_LOG_DIR="$REPO/logs/pipeline"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="$RUN_LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
if [[ "${__PIPELINE_LOGGING:-}" != "1" ]]; then
    export __PIPELINE_LOGGING=1
    exec "$0" "$@" 2>&1 | tee -a "$RUN_LOG"
fi

set -uo pipefail

# Pipeline mutex: prevent double execution (no flock = race window
# where two pipelines run concurrently, corrupting state).
LOCKFILE="/tmp/ashare-pipeline.lock"
exec 8<>"$LOCKFILE"
if ! flock -w 60 8; then
    echo "ERROR: another pipeline is already running (lock: $LOCKFILE)"
    exit 1
fi

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

echo "=== ashare-pipeline run started at $(date -Iseconds) ==="

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

# ---- Step 2b: Sync code to GPU (guard: predict.py must have --provider-uri) ----

sync_code_to_gpu() {
    local STAGING="/tmp/gpu-deploy"
    rm -rf "$STAGING"
    mkdir -p "$STAGING/ashare_lab/research" "$STAGING/configs"
    cp "$REPO/ashare_lab/research/predict.py" "$STAGING/ashare_lab/research/"
    cp "$REPO/ashare_lab/research/shadow_predict.py" "$STAGING/ashare_lab/research/"
    # matrix_runner and matrix configs may not exist on fresh deploys.
    cp "$REPO/ashare_lab/research/matrix_runner.py" "$STAGING/ashare_lab/research/" 2>/dev/null || true
    cp "$REPO/ashare_lab/config.py" "$STAGING/ashare_lab/"
    touch "$STAGING/ashare_lab/__init__.py"
    touch "$STAGING/ashare_lab/research/__init__.py"
    cp "$REPO/configs/baseline.yaml" "$STAGING/configs/"
    cp "$REPO"/configs/matrix_*.yaml "$STAGING/configs/" 2>/dev/null || true

    if ! scp -r "$STAGING"/* "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/"; then
        echo "WARNING: GPU code sync failed (non-fatal)"
        rm -rf "$STAGING"
        return 1
    fi
    rm -rf "$STAGING"

    # Guard: verify GPU predict.py accepts --provider-uri
    if ! ssh -o ConnectTimeout=5 "${GPU_USER}@${GPU_HOST}" \
        "cd /d H:\\ashare-lab && py -m ashare_lab.research.predict --help" 2>&1 | grep -q "provider-uri"; then
        echo "ERROR: GPU predict.py missing --provider-uri after sync"
        return 1
    fi
    echo "GPU code synced and verified"
    return 0
}

# ---- Step 3: GPU inference with retry (B6: retry GPU BEFORE pipeline) ----

try_gpu_inference() {
    # Check GPU lock before any work (guards both first attempt and retry).
    if [ -e /tmp/ashare-gpu.lock ]; then
        exec 9<>/tmp/ashare-gpu.lock
        if ! flock -n 9; then
            echo "GPU locked by batch_experiment.py, skipping inference (using stale predictions)"
            exec 9>&-
            return 1
        fi
        flock -u 9
        exec 9>&-
    fi

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

    # Sync code to GPU before inference
    sync_code_to_gpu || echo "WARNING: code sync failed, proceeding with existing GPU code"

    # Verify GPU has today's trade date in qlib calendar (prevent stale data race)
    local gpu_has_date
    # qlib INFO logs go to stdout; strip CR (Windows SSH), grep exact yes/no
    gpu_has_date=$(ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        "py -c \"import qlib; qlib.init(provider_uri='H:/.qlib/qlib_data/cn_data', region='cn'); from qlib.data import D; cal=D.calendar(start_time='$TRADE_DATE',end_time='$TRADE_DATE'); print('yes' if len(cal)>0 else 'no')\"" 2>/dev/null | tr -d '\r' | grep -x 'yes\|no' | tail -1)
    if [ "$gpu_has_date" != "yes" ]; then
        echo "ERROR: GPU qlib data missing trade date $TRADE_DATE (data sync incomplete)"
        return 1
    fi

    # W4: detect current GPU consumer before clearing
    local gpu_restore_target="ollama"
    local _llama_running
    _llama_running=$(ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        'tasklist /FI "IMAGENAME eq llama-server.exe" /NH 2>NUL | findstr /I llama-server' 2>/dev/null | tr -d '\r')
    if [ -n "$_llama_running" ]; then
        gpu_restore_target="llama-server"
        echo "GPU consumer detected: llama-server"
    else
        echo "GPU consumer detected: ollama (default)"
    fi

    # W4: clear GPU for predict (stop ollama + llama-server)
    ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" 'H:\gpu-switch.bat training' || \
        echo "WARNING: gpu-switch training failed (non-fatal)"

    # Run predict.py on GPU
    local predict_rc=0
    if ! timeout $GPU_PREDICT_TIMEOUT ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        "cd /d H:\\ashare-lab && py -m ashare_lab.research.predict --date $TRADE_DATE --provider-uri H:/.qlib/qlib_data/cn_data"; then
        echo "ERROR: GPU predict.py failed or timed out"
        predict_rc=1
    fi

    # W4: restore previous GPU consumer (always, even on failure)
    ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" "H:\\gpu-switch.bat ${gpu_restore_target}" || \
        echo "WARNING: gpu-switch ${gpu_restore_target} failed (non-fatal)"

    [ $predict_rc -ne 0 ] && return 1

    # SCP predictions back
    mkdir -p "$PREDICTIONS_DIR"
    if ! scp "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/predictions/${TRADE_DATE}.parquet" "$PREDICTIONS_DIR/"; then
        echo "ERROR: SCP predictions failed"
        return 1
    fi
    # SCP meta.json (non-fatal: parquet is the critical file)
    scp "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/predictions/${TRADE_DATE}.meta.json" "$PREDICTIONS_DIR/" 2>/dev/null || true

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
    TODAY=$(date +%Y-%m-%d)
    LATEST_PRED=$(find "$PREDICTIONS_DIR" -name '*.parquet' 2>/dev/null \
        | awk -v today="$TODAY" '{
            f=$0; gsub(/.*\//, "", f); gsub(/\.parquet$/, "", f)
            if (f <= today) print
        }' | sort | tail -1)
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
ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG 2>>"$STDERR_LOG"
PIPELINE_RC=$?

# Pipeline retry: 2 attempts total (1 initial + 1 retry, D-D14)
if [ $PIPELINE_RC -ne 0 ]; then
    echo "Pipeline attempt 1 failed (rc=$PIPELINE_RC), retrying in 30min"
    sleep 1800
    # shellcheck disable=SC2086
    ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG 2>>"$STDERR_LOG"
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
