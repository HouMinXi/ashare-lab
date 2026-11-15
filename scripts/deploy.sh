#!/bin/bash
# deploy.sh -- two-stage deploy for ashare-lab on W1
# Fresh clone or incremental sync, then install systemd timers.
#
# Usage: bash scripts/deploy.sh
#   (or run from any directory -- the script finds/creates the repo)
#
# Takes no pipeline lock: a manual deploy can run while ashare-pipeline.sh
# holds /tmp/ashare-pipeline.lock and mutates $REPO under it. The two paths
# stage to separate directories, so they cannot delete each other's tree, but
# a deploy started mid-run still ships whatever the tree looks like at that
# moment. Run it outside the 17:00-22:00 window.
set -euo pipefail

REPO="$HOME/code/ashare-lab"
REPO_URL="w1:code/ashare-lab.git"
GPU_HOST="192.168.100.11"
GPU_USER="admin"

# ---- Section 1-3: Repo detection + sync ----

if [ -d "$REPO/.git" ]; then
    echo "==> Incremental sync (repo exists at $REPO)"
    cd "$REPO"
    git pull
    uv pip install -e .
else
    echo "==> Fresh clone to $REPO"
    mkdir -p "$(dirname "$REPO")"
    git clone "$REPO_URL" "$REPO"
    cd "$REPO"
    uv pip install -e .
fi

# ---- Section 4: Data update (D-D04: forced every run) ----

echo "==> Updating market data"
python3 -m ashare_lab.cli update || echo "WARNING: data update failed (non-fatal, timer handles daily updates)"

# ---- Section 5: Verify prerequisites ----

echo "==> Checking prerequisites"

MISSING=0
for key in ashare/weixin-token ashare/weixin-chat-id api/omniroute; do
    if ! pass show "$key" >/dev/null 2>&1; then
        echo "ERROR: pass key '$key' not found. Add with: pass insert $key"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "FATAL: Missing pass secrets. Add them and re-run deploy.sh."
    exit 1
fi

if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo "ERROR: aiohttp not importable. Run: uv pip install aiohttp"
    exit 1
fi

if ! command -v wol >/dev/null 2>&1; then
    echo "ERROR: wol not found. Install: sudo dnf install wol"
    exit 1
fi

# ---- Section 5b: GPG agent non-interactive cache (B5) ----

echo "==> Configuring GPG agent for long-lived cache"
mkdir -p ~/.gnupg
chmod 700 ~/.gnupg
grep -q "max-cache-ttl" ~/.gnupg/gpg-agent.conf 2>/dev/null || \
    echo "max-cache-ttl 31536000" >> ~/.gnupg/gpg-agent.conf
grep -q "default-cache-ttl" ~/.gnupg/gpg-agent.conf 2>/dev/null || \
    echo "default-cache-ttl 31536000" >> ~/.gnupg/gpg-agent.conf
gpgconf --reload gpg-agent

if ! systemd-run --user --scope pass show ashare/weixin-token >/dev/null 2>&1; then
    echo "WARNING: GPG passphrase not cached. Enter it once interactively:"
    echo "  pass show ashare/weixin-token"
    echo "After that, the systemd timers can access pass non-interactively."
fi

# ---- Section 6: SSH host key for GPU ----

echo "==> Setting up GPU SSH"
mkdir -p ~/.ssh
ssh-keyscan "$GPU_HOST" >> ~/.ssh/known_hosts 2>/dev/null
# deduplicate known_hosts
sort -u -o ~/.ssh/known_hosts ~/.ssh/known_hosts

if ssh -o ConnectTimeout=5 -o BatchMode=yes "${GPU_USER}@${GPU_HOST}" "echo ok" >/dev/null 2>&1; then
    echo "  GPU SSH reachable"
else
    echo "  GPU SSH not reachable -- host setup continues, but the code" \
         "deploy below will be recorded as failed"
fi

# ---- Section 7: SCP predict.py + deps to GPU (staging dir, no remote mkdir -p) ----

echo "==> Staging files for GPU deploy"
STAGING="${GPU_DEPLOY_STAGING:-/tmp/gpu-deploy-manual}"
if [ ! -f scripts/stage_gpu_code.sh ]; then
    echo "ERROR: run deploy.sh from the repository root" >&2
    exit 1
fi
# Same staging contract as the scheduled pipeline, so both paths leave the
# target in the same state and the same verifier can check it.
bash scripts/stage_gpu_code.sh "$PWD" "$STAGING"

# The code transfer is one section of a host setup: a sleeping GPU must not
# leave the timers uninstalled, so record the failure and keep going, then
# report it in the exit status (including when a later section aborts).
CODE_DEPLOY_FAILED=0
trap '[ "$CODE_DEPLOY_FAILED" -eq 0 ] || echo "NOTE: the GPU tree is not confirmed as the shipped tree" >&2' EXIT
if ! ( shopt -s dotglob; scp -o ConnectTimeout=10 -r "$STAGING"/* "${GPU_USER}@${GPU_HOST}":"H:/ashare-lab/" ); then
    echo "ERROR: GPU code transfer failed; the target tree is not the shipped tree" >&2
    CODE_DEPLOY_FAILED=1
else
    if timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 \
        "${GPU_USER}@${GPU_HOST}" "cd /d H:\\ashare-lab && py deploy_verify.py"; then
        verify_rc=0
    else
        verify_rc=$?
    fi
    if [ "$verify_rc" -eq 124 ] || [ "$verify_rc" -eq 255 ]; then
        echo "ERROR: the deploy check did not complete (rc=$verify_rc); the target tree is unconfirmed" >&2
        CODE_DEPLOY_FAILED=1
    elif [ "$verify_rc" -ne 0 ]; then
        echo "ERROR: the deploy check reported a mismatch; the target tree is not the shipped tree" >&2
        CODE_DEPLOY_FAILED=1
    fi
fi
rm -rf "$STAGING"

# ---- Section 8: loginctl enable-linger (MANDATORY for 7x24 timers) ----

echo "==> Enabling linger for $(whoami)"
loginctl enable-linger "$(whoami)"

# ---- Section 9: Generate and install systemd user units ----

echo "==> Installing systemd timers"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/ashare-data-update.timer" << 'UNIT'
[Unit]
Description=ashare-lab data update (17:00 CST)

[Timer]
OnCalendar=*-*-* 17:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-data-update.service" << UNIT
[Unit]
Description=ashare-lab data update

[Service]
Type=oneshot
WorkingDirectory=$REPO
ExecStart=$REPO/scripts/ashare-data-update.sh
Environment=PATH=$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=VIRTUAL_ENV=$REPO/.venv
TimeoutStartSec=600
StandardOutput=journal
StandardError=journal
UNIT

cat > "$UNIT_DIR/ashare-pipeline.timer" << 'UNIT'
[Unit]
Description=ashare-lab pipeline (18:00 CST)

[Timer]
OnCalendar=*-*-* 18:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-pipeline.service" << UNIT
[Unit]
Description=ashare-lab daily pipeline

[Service]
Type=oneshot
WorkingDirectory=$REPO
ExecStart=$REPO/scripts/ashare-pipeline.sh
Environment=PATH=$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=VIRTUAL_ENV=$REPO/.venv
TimeoutStartSec=10800
StandardOutput=journal
StandardError=journal
UNIT

cat > "$UNIT_DIR/ashare-chenditc.timer" << 'UNIT'
[Unit]
Description=ashare-lab chenditc backup (21:00 CST)

[Timer]
OnCalendar=*-*-* 21:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-chenditc.service" << UNIT
[Unit]
Description=ashare-lab chenditc backup

[Service]
Type=oneshot
WorkingDirectory=$REPO
ExecStart=$REPO/scripts/ashare-chenditc.sh
Environment=PATH=$REPO/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=VIRTUAL_ENV=$REPO/.venv
TimeoutStartSec=3600
StandardOutput=journal
StandardError=journal
UNIT

cat > "$UNIT_DIR/ashare-retrain.timer" << 'UNIT'
[Unit]
Description=IC-driven ashare model retraining (daily check at 22:00)

[Timer]
OnCalendar=*-*-* 22:00:00
RandomizedDelaySec=600
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-retrain.service" << UNIT
[Unit]
Description=ashare model retraining
After=network-online.target

[Service]
Type=oneshot
ExecStart=$REPO/scripts/ashare-retrain.sh
TimeoutStartSec=86400
UNIT

systemctl --user daemon-reload
systemctl --user enable --now ashare-data-update.timer
systemctl --user enable --now ashare-pipeline.timer
systemctl --user enable --now ashare-chenditc.timer
systemctl --user enable --now ashare-retrain.timer

# ---- Section 10: Log rotation note ----
# ponytail: journald user logs auto-rotate via /etc/systemd/journald.conf
# SystemMaxUse= setting. Monitor with: journalctl --user --disk-usage
# Add per-user cap if growth exceeds 500MB.

# ---- Section 11: Verification summary ----

echo ""
echo "=== Deploy verification ==="
systemctl --user list-timers --no-pager
echo ""
loginctl show-user "$(whoami)" -p Linger
echo ""
if [ "$CODE_DEPLOY_FAILED" -ne 0 ]; then
    echo "Deploy finished with errors: the GPU tree is not confirmed as the shipped tree; re-run before the 18:00 job." >&2
    exit 1
fi
echo "Deploy complete."
