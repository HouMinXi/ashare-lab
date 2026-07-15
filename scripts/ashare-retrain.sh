#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPU_HOST="admin@192.168.100.11"
GPU_REPO="H:/ashare-lab"
WOL_MAC="04:7C:16:49:BE:32"
LOG_FILE="$PROJECT_DIR/logs/retrain-$(date +%Y%m%d).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$PROJECT_DIR/logs"

log "Starting monthly retrain"

# Step 1: Wake GPU
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

# Step 2: Switch GPU to training mode
ssh "$GPU_HOST" "H:/nssm/nssm.exe stop llama-server" 2>/dev/null || true
sleep 5

# Step 3: Run training (10 windows)
log "Starting training..."
ssh "$GPU_HOST" "cd $GPU_REPO && py -m ashare_lab.research.train" >> "$LOG_FILE" 2>&1
TRAIN_RC=$?
log "Training exit code: $TRAIN_RC"

# Step 4: Copy models back
log "Copying models..."
mkdir -p "$PROJECT_DIR/models"
scp "$GPU_HOST:$GPU_REPO/models/"*.pt "$PROJECT_DIR/models/" 2>/dev/null || true

# Step 5: Update latest.pt symlink
LATEST=$(ls -t "$PROJECT_DIR/models"/w*.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    ln -sf "$(basename "$LATEST")" "$PROJECT_DIR/models/latest.pt"
    log "latest.pt -> $(basename "$LATEST")"
fi

# Step 6: Restart llama-server
ssh "$GPU_HOST" "H:/nssm/nssm.exe start llama-server" 2>/dev/null || true

log "Retrain complete (exit $TRAIN_RC)"
