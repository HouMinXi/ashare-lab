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

# --- Step 1: Wake GPU ---
log "Waking GPU..."
wol "$WOL_MAC"
for i in $(seq 1 40); do
    ping -c 1 -W 2 "${GPU_HOST#*@}" >/dev/null 2>&1 && break
    sleep 5
done

# Wait for SSH
for i in $(seq 1 60); do
    ssh -o ConnectTimeout=5 "$GPU_HOST" "echo ok" >/dev/null 2>&1 && break
    sleep 5
done

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
if ssh -o ConnectTimeout=5 "$GPU_HOST" "cd $GPU_REPO && py -m ashare_lab.research.train --force" >> "$LOG_FILE" 2>&1; then
    :
else
    TRAIN_RC=$?
fi
log "Training exit code: $TRAIN_RC"

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

# --- Step 6: Restart llama-server ---
ssh -o ConnectTimeout=5 "$GPU_HOST" "H:/nssm/nssm.exe start llama-server" 2>/dev/null || true

# --- Step 7: Record retrain date (only on real success) ---
if [ "$TRAIN_RC" -eq 0 ] && [ "$PT_OK" -eq 1 ]; then
    echo "$TODAY" > "$SENTINEL"
    log "Retrain complete, sentinel written: $TODAY"
else
    log "Retrain incomplete (train_rc=$TRAIN_RC, pt_ok=$PT_OK), sentinel NOT written"
fi
