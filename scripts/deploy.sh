#!/bin/bash
# deploy.sh -- two-stage deploy for ashare-lab on X500
# Fresh clone or incremental sync, then install systemd timers.
#
# Usage: bash scripts/deploy.sh
#   (or run from any directory -- the script finds/creates the repo)
set -euo pipefail

REPO="$HOME/code/ashare-lab"
REPO_URL="x500:code/ashare-lab.git"
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
for key in ashare/weixin-token ashare/weixin-chat-id ashare/deepseek-api-key; do
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
    echo "  GPU SSH not reachable (non-fatal -- may be powered off)"
fi

# ---- Section 7: SCP predict.py + deps to GPU (staging dir, no remote mkdir -p) ----

echo "==> Staging files for GPU deploy"
STAGING="/tmp/gpu-deploy"
rm -rf "$STAGING"
mkdir -p "$STAGING/ashare_lab/research" "$STAGING/configs"
cp ashare_lab/research/predict.py "$STAGING/ashare_lab/research/"
cp ashare_lab/config.py "$STAGING/ashare_lab/"
touch "$STAGING/ashare_lab/__init__.py"
touch "$STAGING/ashare_lab/research/__init__.py"
cp configs/baseline.yaml "$STAGING/configs/"

scp -r "$STAGING"/* "${GPU_USER}@${GPU_HOST}":"'C:/Users/admin/ashare-lab/'" \
    || echo "  GPU SCP failed (non-fatal -- may be powered off)"
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
Description=ashare-lab data update (00:30 CST)

[Timer]
OnCalendar=*-*-* 00:30:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-data-update.service" << UNIT
[Unit]
Description=ashare-lab data update

[Service]
Type=oneshot
ExecStart=$REPO/scripts/ashare-data-update.sh
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
TimeoutStartSec=600
StandardOutput=journal
StandardError=journal
UNIT

cat > "$UNIT_DIR/ashare-pipeline.timer" << 'UNIT'
[Unit]
Description=ashare-lab pipeline (01:00 CST)

[Timer]
OnCalendar=*-*-* 01:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

cat > "$UNIT_DIR/ashare-pipeline.service" << UNIT
[Unit]
Description=ashare-lab daily pipeline

[Service]
Type=oneshot
ExecStart=$REPO/scripts/ashare-pipeline.sh
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
TimeoutStartSec=10800
StandardOutput=journal
StandardError=journal
UNIT

systemctl --user daemon-reload
systemctl --user enable --now ashare-data-update.timer
systemctl --user enable --now ashare-pipeline.timer

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
echo "Deploy complete."
