#!/usr/bin/env bash
# IC-driven retraining: retrain only when IC < 0.02 for 3 consecutive days.
# Runs daily via systemd timer at 22:00.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPU_HOST="admin@192.168.100.11"
GPU_REPO="H:/ashare-lab"
WOL_MAC="04:7C:16:49:BE:32"
LOG_FILE="$PROJECT_DIR/logs/retrain-$(date +%Y%m%d).log"
IC_HISTORY="$PROJECT_DIR/data/ic_history.tsv"
SENTINEL="$PROJECT_DIR/data/last_retrain_date"
TODAY=$(date +%Y-%m-%d)
IC_THRESHOLD=0.02
CONSECUTIVE_DAYS=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }


# Parse expected_live_model from baseline.yaml. No python: a quoted
# regex in python3 -c is a syntax error that 2>/dev/null swallows.
read_expected_live_model() {
    local f="$PROJECT_DIR/configs/baseline.yaml"
    [ -f "$f" ] || return 0
    awk '/^[[:space:]]*expected_live_model:/ {
        gsub(/["\047]/, "", $2)
        print tolower($2)
        exit
    }' "$f"
}

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/data"

# --- Sentinel check: prevent double training ---
if [ -f "$SENTINEL" ] && grep -q "^${TODAY}$" "$SENTINEL"; then
    log "Already retrained today ($TODAY), skipping"
    exit 0
fi

# --- Cooldown: skip if retrained within last 7 days ---
COOLDOWN_DAYS=7
if [ -f "$SENTINEL" ]; then
    last_retrain=$(cat "$SENTINEL")
    last_epoch=$(date -d "$last_retrain" +%s 2>/dev/null || echo 0)
    now_epoch=$(date +%s)
    elapsed=$(( now_epoch - last_epoch ))
    min_gap=$(( COOLDOWN_DAYS * 86400 ))
    if [ "$elapsed" -lt "$min_gap" ]; then
        log "Cooldown: last retrain was ${elapsed}s ago (need ${min_gap}s), skipping"
        exit 0
    fi
fi

# --- IC gate: check consecutive low-IC days ---
if [ ! -f "$IC_HISTORY" ]; then
    log "No IC history file at $IC_HISTORY, skipping retrain"
    exit 0
fi

low_count=0
while IFS=$'\t' read -r date ic_val; do
    [ -z "$date" ] && continue
    # Empty or null IC counts as low
    if [ -z "$ic_val" ]; then
        low_count=$((low_count + 1))
        continue
    fi
    # Compare: IC < threshold counts as low (skip non-numeric)
    if ! printf '%s' "$ic_val" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
        low_count=$((low_count + 1))
    elif awk "BEGIN { exit !($ic_val < $IC_THRESHOLD) }"; then
        low_count=$((low_count + 1))
    else
        low_count=0  # reset on good IC
    fi
done < "$IC_HISTORY"

log "IC gate: $low_count consecutive days below $IC_THRESHOLD (need $CONSECUTIVE_DAYS)"

if [ "$low_count" -lt "$CONSECUTIVE_DAYS" ]; then
    log "IC gate not triggered, no retrain needed"
    exit 0
fi

log "IC gate triggered: $low_count consecutive low-IC days, starting retrain"


# Serve-now live models (w115) sit outside walk-forward (w1..w11), so
# --force all-windows would write w11.pt on gpu-win and displace live
# w115. Train the newest window instead: get_all_windows stops at the
# last window whose test period has started, so the freshest trainable
# window needs --serve-now <step> to reach.
EXPECTED_MODEL=$(read_expected_live_model)
EXPECTED_N=$(echo "$EXPECTED_MODEL" | sed -n 's/.*[wW]\([0-9]*\).*/\1/p')
TRAIN_ARGS="--force"
if [ -n "$EXPECTED_N" ] && [ "$EXPECTED_N" -ge 100 ]; then
    # get_all_windows() stops at the last window whose test period has
    # started, so its length is the step of the freshest trainable window.
    SERVE_NOW_STEP=$(PYTHONPATH="$PROJECT_DIR" python3 -c '
from ashare_lab.research.smoke_test import get_all_windows
print(len(get_all_windows()))
' 2>/dev/null || true)
    if [ -z "$SERVE_NOW_STEP" ]; then
        log "serve-now live model ${EXPECTED_MODEL}: cannot derive step; aborting rather than running --force over the walk-forward set"
        exit 1
    fi
    log "serve-now live model ${EXPECTED_MODEL}: training step ${SERVE_NOW_STEP} instead of --force all-windows"
    TRAIN_ARGS="--serve-now $SERVE_NOW_STEP --force"
fi

# --- Step 1: Wake GPU ---
# Same sequence as data-update and pipeline. The old inline loops logged
# "GPU online" and kept going when neither ping nor ssh ever answered, so
# systemd recorded a green run and the deploy fallback rewrote latest.pt
# from whatever models were already on disk.
# shellcheck source=scripts/lib/gpu-wake.sh
source "$PROJECT_DIR/scripts/lib/gpu-wake.sh"
log "Waking GPU..."
if ! wake_gpu "${GPU_HOST#*@}" "$WOL_MAC" "${GPU_HOST%%@*}" 300; then
    log "GPU not reachable, aborting retrain"
    exit 1
fi
log "GPU online"

# --- Step 2: Switch GPU to training mode ---
ssh "$GPU_HOST" "H:/nssm/nssm.exe stop llama-server" 2>/dev/null || true
sleep 5

# --- Step 3: Flock guard + Run training ---
exec 9>"$PROJECT_DIR/data/retrain.lock"
if ! flock -n 9; then
    log "retrain lock held by another run, skipping"
    exit 0
fi

# Delete gpu-side stale meta
ssh -o ConnectTimeout=5 "$GPU_HOST" "del H:\\ashare-lab\\models\\meta.json 2>NUL" || true

log "Starting training..."
PT_OK=0
TRAIN_RC=0
if ssh -o ConnectTimeout=5 "$GPU_HOST" "cd $GPU_REPO && py -m ashare_lab.research.train $TRAIN_ARGS" >> "$LOG_FILE" 2>&1; then
    :
else
    TRAIN_RC=$?
fi
log "Training exit code: $TRAIN_RC"

# --- Step 3b: Deploy gate (expected_live_model check) ---
DEPLOY_SKIP=0
NEW_MODEL=""
EXPECTED_MODEL=""

# Read new model name from gpu-side meta.json (parse remotely, don't depend on scp)
NEW_MODEL=$(ssh -o ConnectTimeout=5 "$GPU_HOST" \
    "cd $GPU_REPO/models && python -c \"import json,sys; print(json.load(open('meta.json'))['model_file'])\"" 2>/dev/null || true)

# Read expected_live_model from baseline config
EXPECTED_MODEL=$(read_expected_live_model)

# Serve-now models carry an out-of-band window number (w115), so "12 > 115"
# is false and a staler w12 would sail through the numeric gate. Compare
# training dates instead: only a model trained on newer data may deploy.
# The branch is closed: a serve-now run that cannot read both dates shelves
# rather than falling back to the numeric gate that cannot judge it.
SERVE_NOW_LIVE=0
NEW_TRAIN_DATE=""
LIVE_TRAIN_DATE=""
if [ -n "$EXPECTED_N" ] && [ "$EXPECTED_N" -ge 100 ]; then
    SERVE_NOW_LIVE=1
    NEW_TRAIN_DATE=$(ssh -o ConnectTimeout=5 "$GPU_HOST" \
        "cd $GPU_REPO/models && python -c \"import json,sys; print(json.load(open('meta.json'))['train_date'])\"" 2>/dev/null || true)
    if [ -f "$PROJECT_DIR/models/meta.json" ]; then
        LIVE_TRAIN_DATE=$(META_PATH="$PROJECT_DIR/models/meta.json" \
        PYTHONPATH="$PROJECT_DIR" python3 -c '
import json, os
print(json.load(open(os.environ["META_PATH"])).get("train_date", ""))
' 2>/dev/null || true)
    fi
fi

if [ "$SERVE_NOW_LIVE" -eq 1 ]; then
    if [ -z "$NEW_TRAIN_DATE" ] || [ -z "$LIVE_TRAIN_DATE" ]; then
        DEPLOY_SKIP=1
        log "serve-now deploy gate: train_date unavailable (new='${NEW_TRAIN_DATE}', live='${LIVE_TRAIN_DATE}'); not deploying"
        NEW_TRAIN_DATE="$NEW_TRAIN_DATE" LIVE_TRAIN_DATE="$LIVE_TRAIN_DATE" \
        PYTHONPATH="$PROJECT_DIR" python3 -c '
import os
from ashare_lab.bridge import send_bridge_alert
send_bridge_alert("ashare retrain: model shelved",
                  "serve-now deploy gate could not read train_date "
                  "(new=%s, live=%s), artifact left on gpu-win"
                  % (os.environ["NEW_TRAIN_DATE"], os.environ["LIVE_TRAIN_DATE"]))
' >>"$LOG_FILE" 2>&1 || true
    elif [ "$NEW_TRAIN_DATE" \< "$LIVE_TRAIN_DATE" ] || [ "$NEW_TRAIN_DATE" = "$LIVE_TRAIN_DATE" ]; then
        DEPLOY_SKIP=1
        log "${NEW_MODEL} trained on ${NEW_TRAIN_DATE} is not newer than live ${LIVE_TRAIN_DATE}; not deploying"
        NEW_MODEL="$NEW_MODEL" NEW_TRAIN_DATE="$NEW_TRAIN_DATE" LIVE_TRAIN_DATE="$LIVE_TRAIN_DATE" \
        PYTHONPATH="$PROJECT_DIR" python3 -c '
import os
from ashare_lab.bridge import send_bridge_alert
send_bridge_alert("ashare retrain: model shelved",
                  "%s train_date %s not newer than live %s, artifact left on gpu-win"
                  % (os.environ["NEW_MODEL"], os.environ["NEW_TRAIN_DATE"],
                     os.environ["LIVE_TRAIN_DATE"]))
' >>"$LOG_FILE" 2>&1 || true
    else
        log "${NEW_MODEL} trained on ${NEW_TRAIN_DATE} supersedes live ${LIVE_TRAIN_DATE}; deploying"
    fi
# Numeric comparison: extract digits after 'w'
elif [ -n "$NEW_MODEL" ] && [ -n "$EXPECTED_MODEL" ]; then
    NEW_N=$(echo "$NEW_MODEL" | sed -n 's/.*[wW]\([0-9]*\).*/\1/p')
    if [ -n "$NEW_N" ] && [ -n "$EXPECTED_N" ] && [ "$NEW_N" -gt "$EXPECTED_N" ]; then
        DEPLOY_SKIP=1
        log "w${NEW_N} trained but shelved (expected_live_model=w${EXPECTED_N}); not deploying"
        # Alert via the bridge module (failure must not affect script)
        NEW_N="$NEW_N" EXPECTED_N="$EXPECTED_N" \
        PYTHONPATH="$PROJECT_DIR" python3 -c '
import os
from ashare_lab.bridge import send_bridge_alert
send_bridge_alert("ashare retrain: model shelved",
                  "w%s trained but shelved (expected_live_model=w%s), "
                  "artifact left on gpu-win"
                  % (os.environ["NEW_N"], os.environ["EXPECTED_N"]))
' >>"$LOG_FILE" 2>&1 || true
    fi
elif [ -z "$NEW_MODEL" ]; then
    log "WARNING: could not read gpu-side meta.json, deploying as fallback"
fi

if [ "$DEPLOY_SKIP" -eq 1 ]; then
    # Shelved: skip scp + relink, but still restart llama-server + write sentinel
    log "Shelved model: skipping Step 4 (scp) and Step 5 (relink)"
else
    # --- Step 4: Copy models back ---
    log "Copying models..."
    mkdir -p "$PROJECT_DIR/models"
    rm -f "$PROJECT_DIR/models/meta.json"
    if scp -o ConnectTimeout=5 "$GPU_HOST:$GPU_REPO/models/w*[0-9].pt" "$PROJECT_DIR/models/"; then
        PT_OK=1
        scp -o ConnectTimeout=5 "$GPU_HOST:$GPU_REPO/models/meta.json" "$PROJECT_DIR/models/" || true
    fi

    # --- Step 5: Update latest.pt symlink ---
    # Helper: find latest model file with version sorting
    _find_latest_model() {
        find "$PROJECT_DIR/models" -maxdepth 1 -name 'w*[0-9].pt' ! -name '*_backup*' -print0 2>/dev/null | sort -zV | tail -z -n1 | tr -d '\0' || true
    }

    if [ -f "$PROJECT_DIR/models/meta.json" ]; then
        META_MODEL=$(python3 -c "import json; print(json.load(open('$PROJECT_DIR/models/meta.json'))['model_file'])" 2>/dev/null || true)
        if [ -n "$META_MODEL" ] && [ -f "$PROJECT_DIR/models/$META_MODEL" ]; then
            ln -sf "$META_MODEL" "$PROJECT_DIR/models/latest.pt"
            log "latest.pt -> $META_MODEL (from meta.json)"
        else
            LATEST=$(_find_latest_model)
            if [ -n "$LATEST" ]; then
                ln -sf "$(basename "$LATEST")" "$PROJECT_DIR/models/latest.pt"
                log "latest.pt -> $(basename "$LATEST") (from version-sorted fallback)"
            fi
        fi
    else
        LATEST=$(_find_latest_model)
        if [ -n "$LATEST" ]; then
            ln -sf "$(basename "$LATEST")" "$PROJECT_DIR/models/latest.pt"
            log "latest.pt -> $(basename "$LATEST") (from version-sorted fallback, no meta.json)"
        fi
    fi
fi

# --- Step 6: Restart llama-server (always, even if shelved) ---
ssh -o ConnectTimeout=5 "$GPU_HOST" "H:/nssm/nssm.exe start llama-server" 2>/dev/null || true

# --- Step 7: Record retrain date (only on real success) ---
# Shelved run: training success (TRAIN_RC==0) alone suffices.
# Normal deploy: the scp back must also have succeeded (PT_OK==1).
if [ "$TRAIN_RC" -eq 0 ] && { [ "$DEPLOY_SKIP" -eq 1 ] || [ "$PT_OK" -eq 1 ]; }; then
    echo "$TODAY" > "$SENTINEL"
    log "Retrain complete, sentinel written: $TODAY"
else
    log "Retrain incomplete (train_rc=$TRAIN_RC, pt_ok=$PT_OK), sentinel NOT written"
fi
