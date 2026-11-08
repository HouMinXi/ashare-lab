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

# --force runs the retrain even when the IC gate would decline, for a
# deliberate out-of-band training run.  It does not skip the deploy gate.
GATE_BYPASS=0
if [ "${1:-}" = "--force" ]; then
    GATE_BYPASS=1
fi

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

if [ "$low_count" -lt "$CONSECUTIVE_DAYS" ] && [ "$GATE_BYPASS" -eq 0 ]; then
    log "IC gate not triggered, no retrain needed"
    exit 0
fi

if [ "$GATE_BYPASS" -eq 1 ]; then
    log "IC gate bypassed by --force; proceeding with retrain"
fi

log "IC gate triggered: $low_count consecutive low-IC days, starting retrain"


# Serve-now live models (w115) sit outside walk-forward (w1..w11), so
# --force all-windows would write w11.pt on gpu-win and displace live
# w115. Train the newest window instead: get_all_windows stops at the
# last window whose test period has started, so the freshest trainable
# window needs --serve-now <step> to reach.
EXPECTED_MODEL=$(read_expected_live_model)
# Always retrain the freshest window.  get_all_windows() stops at the last
# window whose test period has started, so its length is that window's step;
# --force alone would walk the whole history, which is hours of GPU time for a
# cadence refresh that only needs the newest slice.
TRAIN_ARGS="--force"
SERVE_NOW_STEP=$(PYTHONPATH="$PROJECT_DIR" python3 -c '
from ashare_lab.research.smoke_test import get_all_windows
print(len(get_all_windows()))
' 2>/dev/null || true)
if [ -z "$SERVE_NOW_STEP" ]; then
    log "cannot derive the newest window step; aborting rather than running --force over the walk-forward set"
    exit 1
fi
log "training window step ${SERVE_NOW_STEP} (live model ${EXPECTED_MODEL:-unknown})"
TRAIN_ARGS="--serve-now $SERVE_NOW_STEP --force"

# --- Step 1: Wake GPU ---
# Same sequence as data-update and pipeline. The old inline loops logged
# "GPU online" and kept going when neither ping nor ssh ever answered, so
# systemd recorded a green run and the deploy fallback rewrote latest.pt
# from whatever models were already on disk.
# shellcheck source=scripts/lib/gpu-wake.sh
source "$PROJECT_DIR/scripts/lib/gpu-wake.sh"
# The library carries the address without a user, and sourcing it replaces the
# admin@ host this script set above.  Without this reassembly wake_gpu polls
# ssh as "<ip>@<ip>" until its budget runs out and every later ssh loses the
# user too.
GPU_HOST="${GPU_USER}@${GPU_HOST}"
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
if ssh -o ConnectTimeout=5 "$GPU_HOST" "cd /d $GPU_REPO && py -m ashare_lab.research.train $TRAIN_ARGS" >> "$LOG_FILE" 2>&1; then
    :
else
    TRAIN_RC=$?
fi
log "Training exit code: $TRAIN_RC"

# --- Step 3b: Deploy gate ---
# Compare training dates, and identify both sides by name: the newest
# walk-forward checkpoint (w<digits>.pt) on each host. meta.json is deliberately
# not consulted -- a live artifact shares this directory, and whichever model
# wrote it last would otherwise supply the dates for a comparison it does not
# belong to.
DEPLOY_SKIP=0
EXPECTED_MODEL=$(read_expected_live_model)

_wf_train_date() {
    # $1 = models dir, $2 = model file name
    META_DIR="$1" META_MODEL="$2" python3 -c '
import json, os
d, m = os.environ["META_DIR"], os.environ["META_MODEL"]
stem = m[:-3] if m.endswith(".pt") else m
raw = {}
for name in (stem + ".meta.json", "meta.json"):
    try:
        candidate = json.load(open(os.path.join(d, name)))
    except Exception:
        continue
    if candidate.get("model_file", m) == m:
        raw = candidate
        break
print(raw.get("train_date", ""))
' 2>/dev/null || true
}

# The model this run produced: newest walk-forward checkpoint on the GPU host.
NEW_MODEL=$(ssh -o ConnectTimeout=5 "$GPU_HOST" "dir /b $GPU_REPO\\models\\w*.pt" 2>/dev/null \
    | tr -d '\r' | awk -F'\\' '{print $NF}' | grep -E '^w[0-9]+\.pt$' | sort -V | tail -1 || true)

# The model it would replace: newest walk-forward checkpoint already on W1.
CUR_MODEL=$(for _f in "$PROJECT_DIR"/models/w[0-9]*.pt; do [ -e "$_f" ] && basename "$_f"; done | sort -V | tail -1 || true)

NEW_TRAIN_DATE=""
LIVE_TRAIN_DATE=""
if [ -n "$NEW_MODEL" ]; then
    _TMP_META=/tmp/retrain_new_meta_$$.json
    if scp -q -o ConnectTimeout=5 "$GPU_HOST:$GPU_REPO/models/${NEW_MODEL%.pt}.meta.json" "$_TMP_META" 2>/dev/null; then
        NEW_TRAIN_DATE=$(META_PATH="$_TMP_META" python3 -c '
import json, os
print(json.load(open(os.environ["META_PATH"])).get("train_date", ""))
' 2>/dev/null || true)
    fi
    rm -f "$_TMP_META"
fi
if [ -n "$CUR_MODEL" ]; then
    LIVE_TRAIN_DATE=$(_wf_train_date "$PROJECT_DIR/models" "$CUR_MODEL")
fi

if [ -z "$NEW_MODEL" ] || [ -z "$NEW_TRAIN_DATE" ] || [ -z "$LIVE_TRAIN_DATE" ]; then
    DEPLOY_SKIP=1
    log "deploy gate: cannot read both train dates (new='${NEW_TRAIN_DATE}' from ${NEW_MODEL:-none}, live='${LIVE_TRAIN_DATE}' from ${CUR_MODEL:-none}); not deploying"
    NEW_TRAIN_DATE="$NEW_TRAIN_DATE" LIVE_TRAIN_DATE="$LIVE_TRAIN_DATE" \
    PYTHONPATH="$PROJECT_DIR" python3 -c '
import os
from ashare_lab.bridge import send_bridge_alert
send_bridge_alert("ashare retrain: model shelved",
                  "deploy gate could not read train_date "
                  "(new=%s, live=%s), artifact left on gpu-win"
                  % (os.environ["NEW_TRAIN_DATE"], os.environ["LIVE_TRAIN_DATE"]))
' >>"$LOG_FILE" 2>&1 || true
elif [ "$NEW_TRAIN_DATE" \< "$LIVE_TRAIN_DATE" ] || [ "$NEW_TRAIN_DATE" = "$LIVE_TRAIN_DATE" ]; then
    DEPLOY_SKIP=1
    log "${NEW_MODEL} trained on ${NEW_TRAIN_DATE} is not newer than ${CUR_MODEL} ${LIVE_TRAIN_DATE}; not deploying"
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
    log "${NEW_MODEL} trained on ${NEW_TRAIN_DATE} supersedes ${CUR_MODEL} ${LIVE_TRAIN_DATE}; deploying"
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

    # The GPU host runs the nightly prediction and loads latest.pt for any date
    # past the last window. On Windows that path is a real copy, not the symlink
    # used here, so relinking only this side leaves the inference host on the
    # previous model: the 2026-11-07 retrain deployed w12 and the GPU kept
    # serving w115 until the copy was made by hand.
    DEPLOYED_MODEL=$(basename "$(readlink -f "$PROJECT_DIR/models/latest.pt" 2>/dev/null)" 2>/dev/null || true)
    if [ -n "${DEPLOYED_MODEL:-}" ] && [ "$DEPLOYED_MODEL" != "latest.pt" ]; then
        if ssh -o ConnectTimeout=5 "$GPU_HOST" \
            "cd /d $GPU_REPO/models && copy /Y $DEPLOYED_MODEL latest.pt >nul" 2>/dev/null; then
            log "GPU latest.pt -> $DEPLOYED_MODEL"
        else
            log "WARNING: could not point the GPU latest.pt at $DEPLOYED_MODEL; predictions may use the previous model"
        fi
    fi

    # --- Step 5b: the live window ---
    # A walk-forward model cannot use recent data by construction: its training
    # span has to end before its own validation and test spans, which leaves the
    # freshest one months behind. The live window trains three rolling years
    # ending on the last trading day and has no held-out period, so paper
    # trading is its out-of-sample record.
    log "Training live window..."
    LIVE_RC=0
    if ssh -o ConnectTimeout=5 "$GPU_HOST" "cd /d $GPU_REPO && py -m ashare_lab.research.train --live --force" >> "$LOG_FILE" 2>&1; then
        :
    else
        LIVE_RC=$?
    fi
    log "Live window exit code: $LIVE_RC"
    if [ "$LIVE_RC" -eq 0 ] && scp -o ConnectTimeout=5 "$GPU_HOST:$GPU_REPO/models/wlive.pt" "$PROJECT_DIR/models/" 2>>"$LOG_FILE"; then
        scp -o ConnectTimeout=5 "$GPU_HOST:$GPU_REPO/models/meta.json" "$PROJECT_DIR/models/" 2>>"$LOG_FILE" || true
        ln -sf "wlive.pt" "$PROJECT_DIR/models/latest.pt"
        LIVE_TRAIN_DATE_OUT=$(python3 -c "import json; print(json.load(open('$PROJECT_DIR/models/meta.json'))['train_date'])" 2>/dev/null || echo "?")
        log "latest.pt -> wlive.pt (live window through $LIVE_TRAIN_DATE_OUT)"
        if ssh -o ConnectTimeout=5 "$GPU_HOST" "cd /d $GPU_REPO/models && copy /Y wlive.pt latest.pt >nul" 2>>"$LOG_FILE"; then
            log "GPU latest.pt -> wlive.pt"
        else
            log "WARNING: GPU latest.pt not updated to wlive.pt"
        fi
    else
        log "WARNING: live window not deployed (rc=$LIVE_RC); predictions keep the walk-forward model"
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
