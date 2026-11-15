#!/bin/bash
# ashare-pipeline.sh -- Timer 2 wrapper (18:00 CST)
# Trading day check -> GPU inference (retry-first) -> stale fallback ->
# pipeline with ASHARE_USE_STALE env -> pipeline retry -> alert.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).

# Script self-invocation: tee all output to per-run log with reliable exit codes.
# exec > >(tee ...) is unreliable -- pipefail doesn't cover process substitution,
# so tee failures are invisible.  Self-invocation wraps the whole script in a
# pipeline where pipefail DOES apply.
# NOTE: plain child run + exit, NOT exec -- `exec "$0" | tee` leaves the parent
# alive in bash, which falls through and silently re-runs the entire body
# (root cause of the nightly duplicate-run + sidecar clobber, found 2026-08-11).
REPO="$HOME/code/ashare-lab"
RUN_LOG_DIR="$REPO/logs/pipeline"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="$RUN_LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
if [[ "${__PIPELINE_LOGGING:-}" != "1" ]]; then
    export __PIPELINE_LOGGING=1
    "$0" "$@" 2>&1 | tee -a "$RUN_LOG"
    exit "${PIPESTATUS[0]}"
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

GPU_PREDICT_TIMEOUT=1800
DATA_STAMP="$HOME/.cache/ashare-data-update.stamp"
STDERR_LOG="/tmp/ashare-pipeline-stderr.log"
PREDICTIONS_DIR="$REPO/predictions"
SECONDS_START=$SECONDS
USE_STALE=0
PRED_ARG=""
FORCE_ARG=""

# shellcheck source=scripts/lib/gpu-wake.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/gpu-wake.sh"

for _a in "$@"; do
    [ "$_a" = "--force" ] && FORCE_ARG="--force"
done

echo "=== ashare-pipeline run started at $(date -Iseconds) ==="

# ---- Step 1: Trading day check (D-D13) ----

TRADE_DATE=$(python3 -c "
from ashare_lab.data.calendar import latest_trading_day
import datetime as dt
print(latest_trading_day(dt.date.today()))
") || { echo "ERROR: failed to resolve trade date"
    python3 "$REPO/scripts/alert.py" "1" "calendar" "0" "$STDERR_LOG" || true
    exit 1; }

IS_TRADING=$(python3 -c "
from ashare_lab.data.calendar import is_trading_day
import datetime as dt
print(is_trading_day(dt.date.today()))
") || { echo "ERROR: failed to check trading day"
    python3 "$REPO/scripts/alert.py" "1" "calendar" "0" "$STDERR_LOG" || true
    exit 1; }

# ---- Step 1b: Calendar runway ----
# The packaged calendar ends on a fixed date; once it does every run fails at
# step 1. Page while there is still time, on its own stamp so a nightly failure
# alert and this warning do not silence each other.
RUNWAY=$(python3 -c "
from ashare_lab.data.calendar import days_until_calendar_end
print(days_until_calendar_end())
" 2>/dev/null) || RUNWAY=""
if [ -n "$RUNWAY" ] && [ "$RUNWAY" -lt 21 ] 2>/dev/null; then
    ASHARE_ALERT_STAMP="$HOME/.cache/ashare-alert-runway.stamp" \
        python3 "$REPO/scripts/alert.py" --warn \
        "exchange calendar ends in ${RUNWAY} day(s); update exchange_calendars" || true
fi

# Validate TRADE_DATE format (guard against Python stderr leaking into var)
if ! [[ "$TRADE_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: invalid TRADE_DATE '$TRADE_DATE'"
    exit 1
fi

if [ "$IS_TRADING" != "True" ]; then
    echo "Not a trading day, skipping"
    exit 0
fi

# ---- Idempotency guard: a settled day means duplicate run ----
# A second invocation (any launcher) must not reach the gpu/scp phases:
# the blind sidecar scp would clobber monitor keys the first run wrote.
# Matches run_daily's is_day_settled semantics (runs.status='settled'):
# a STALE day must NOT block the designed stale->fresh auto-retry.
# An explicit --force is the authorized rerun path and bypasses the guard.
ALREADY=$(python3 -c "
import sqlite3
db = sqlite3.connect('$REPO/paper.db')
n = db.execute(
    \"SELECT COUNT(*) FROM runs WHERE trade_date = ? AND status = 'settled'\",
    ('$TRADE_DATE',),
).fetchone()[0]
print(n)
" 2>>"$STDERR_LOG") || ALREADY=""
if [ -n "$FORCE_ARG" ]; then
    echo "Force rerun requested, idempotency guard bypassed"
elif [ -z "$ALREADY" ]; then
    echo "WARNING: idempotency check failed (paper.db unreadable), proceeding"
elif [ "$ALREADY" -gt 0 ]; then
    echo "Already settled today ($TRADE_DATE, terminal row present), exiting"
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

# ---- Step 2b: Sync code to GPU (guard: the predictor imports resolve there) ----

sync_code_to_gpu() {
    local STAGING="/tmp/gpu-deploy"

    # A tree that is not level with its upstream ships bytes nobody reviewed:
    # the manifest certifies them and every later check passes. Only a compare
    # that actually ran counts as verified; everything else is marked unknown
    # so the verifier's one printed line says so.
    ASHARE_FRESHNESS=unverified
    local branch remote merge_ref tip behind ahead
    branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    remote=""
    merge_ref=""
    if [ -n "$branch" ] && [ "$branch" != "HEAD" ]; then
        remote=$(git -C "$REPO" config --get "branch.$branch.remote" 2>/dev/null || true)
        merge_ref=$(git -C "$REPO" config --get "branch.$branch.merge" 2>/dev/null || true)
    fi
    case "$remote" in
        "" | ".")
            echo "WARNING: $REPO has no upstream for $branch; shipping its current tree" >&2
            ;;
        *)
            if ! timeout 60 git -C "$REPO" fetch --quiet "$remote" "$merge_ref" 2>/dev/null; then
                echo "WARNING: could not fetch $merge_ref from $remote; shipping its current tree" >&2
            else
                # FETCH_HEAD is what this fetch actually brought back; a
                # tracking ref can be stale or narrowed by the refspec.
                tip=$(git -C "$REPO" rev-parse FETCH_HEAD 2>/dev/null || echo "")
                behind=""
                ahead=""
                if [ -n "$tip" ]; then
                    behind=$(git -C "$REPO" rev-list --count "HEAD..$tip" 2>/dev/null || echo "")
                    ahead=$(git -C "$REPO" rev-list --count "$tip..HEAD" 2>/dev/null || echo "")
                fi
                if [ -z "$tip" ] || [ -z "$behind" ] || [ -z "$ahead" ]; then
                    echo "WARNING: cannot compare $REPO with $remote; shipping its current tree" >&2
                elif [ "$behind" -gt 0 ]; then
                    echo "ERROR: $REPO is $behind commits behind $remote; pull before shipping" >&2
                    return 1
                elif [ "$ahead" -gt 0 ]; then
                    echo "WARNING: $REPO has $ahead commits $remote does not; shipping unverified" >&2
                else
                    ASHARE_FRESHNESS=verified
                fi
            fi
            ;;
    esac
    export ASHARE_FRESHNESS

    if ! bash "$REPO/scripts/stage_gpu_code.sh" "$REPO" "$STAGING"; then
        return 1
    fi

    if ! ( shopt -s dotglob; scp -o ConnectTimeout=10 -r "$STAGING"/* "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/" ); then
        echo "ERROR: GPU code transfer failed" >&2
        rm -rf "$STAGING"
        return 1
    fi
    rm -rf "$STAGING"

    # Verify the remote tree byte-for-byte against the manifest, then import
    # the closure predict_for_date needs. A missing or changed file fails here
    # instead of at 18:00. The check imports torch and qlib on a cold box, so
    # it is bounded: a stall here would hold the flock until the deadman.
    timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        "cd /d H:\\ashare-lab && py deploy_verify.py"
    local verify_rc=$?
    if [ "$verify_rc" -eq 124 ] || [ "$verify_rc" -eq 255 ]; then
        echo "ERROR: the deploy check did not finish (rc=$verify_rc); the target tree is unconfirmed" >&2
        return 1
    elif [ "$verify_rc" -ne 0 ]; then
        echo "ERROR: the deploy check reported a mismatch" >&2
        return 1
    fi
    if [ "$ASHARE_FRESHNESS" = verified ]; then
        echo "GPU code synced and verified"
    else
        echo "GPU code synced (freshness unverified)"
    fi
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

    if ! wake_gpu "$GPU_HOST" "$GPU_MAC" "$GPU_USER" 300; then
        return 1
    fi

    # Sync code to GPU before inference
    if ! sync_code_to_gpu; then
        echo "ERROR: GPU code sync failed; refusing to run inference on unknown code" >&2
        return 1
    fi

    # Verify GPU has today's trade date in qlib calendar (prevent stale data race)
    local gpu_has_date
    # qlib INFO logs go to stdout; strip CR (Windows SSH), grep exact yes/no
    gpu_has_date=$(ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        "py -c \"import qlib; qlib.init(provider_uri='H:/.qlib/qlib_data/cn_data', region='cn'); from qlib.data import D; cal=D.calendar(start_time='$TRADE_DATE',end_time='$TRADE_DATE'); print('yes' if len(cal)>0 else 'no')\"" 2>/dev/null | tr -d '\r' | grep -x 'yes\|no' | tail -1)
    if [ "$gpu_has_date" != "yes" ]; then
        echo "ERROR: GPU qlib data missing trade date $TRADE_DATE (data sync incomplete)"
        return 1
    fi

    # W4: live process wins; else last durable consumer (survives reboot).
    local _llama_running _last_consumer gpu_restore_target
    _llama_running=$(ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        'tasklist /FI "IMAGENAME eq llama-server.exe" /NH 2>NUL | findstr /I llama-server' 2>/dev/null | tr -d '\r')
    _last_consumer=$(ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" \
        'type H:\ollama\logs\gpu-last-consumer.txt 2>NUL' 2>/dev/null | tr -d '\r')
    gpu_restore_target=$(resolve_gpu_restore_target "$_llama_running" "$_last_consumer")
    echo "GPU consumer detected: $gpu_restore_target"
    # The GPU is dedicated to this pipeline: leave it idle afterwards instead
    # of handing it back to a serving process that would pin the whole card.
    # This host has a single card and llama-server is what holds its memory,
    # so the target has to be the one that stops llama-server as well.
    # gpu-switch.bat's plain "training" stops only OllamaService and says so
    # in its own echo, which would leave all 8 GiB pinned.
    if [ "${ASHARE_GPU_DEDICATED:-1}" = "1" ]; then
        echo "GPU dedicated to ashare, releasing card instead of restoring $gpu_restore_target"
        gpu_restore_target=training-3080
    fi

    # W4: clear GPU for predict. The comment below used to say this stops
    # both, but plain "training" leaves llama-server holding the card, which
    # is why predict kept falling back to GPU 1. Use the target that stops
    # llama-server too.
    ssh -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}" 'H:\gpu-switch.bat training-3080' || \
        echo "WARNING: gpu-switch training-3080 failed (non-fatal)"

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
    if ! timeout 300 scp -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/predictions/${TRADE_DATE}.parquet" "$PREDICTIONS_DIR/"; then
        echo "ERROR: SCP predictions failed"
        return 1
    fi
    # SCP meta.json (non-fatal: parquet is the critical file)
    timeout 120 scp -o ConnectTimeout=10 "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/predictions/${TRADE_DATE}.meta.json" "$PREDICTIONS_DIR/" 2>/dev/null || true

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
ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG $FORCE_ARG 2>>"$STDERR_LOG"
PIPELINE_RC=$?

# Pipeline retry: 2 attempts total (1 initial + 1 retry, D-D14)
if [ $PIPELINE_RC -ne 0 ]; then
    echo "Pipeline attempt 1 failed (rc=$PIPELINE_RC), retrying in 30min"
    sleep 1800
    # shellcheck disable=SC2086
    ASHARE_USE_STALE=$USE_STALE python3 -m ashare_lab.cli paper run-all $PRED_ARG $FORCE_ARG 2>>"$STDERR_LOG"
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
