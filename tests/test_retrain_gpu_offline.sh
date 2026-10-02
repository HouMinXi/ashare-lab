#!/usr/bin/env bash
# The retrain script used to log "GPU online" and exit 0 when the host
# never answered SSH. systemd then recorded a green run while no training
# happened. This drives that path: ping and ssh both fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RETRAIN_SCRIPT="$PROJECT_DIR/scripts/ashare-retrain.sh"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p "$tmpdir"/{logs,data,models,configs,shims,bin}

for d in 1 2 3 4; do
    printf '2026-08-%02d\t0.005\n' "$d" >> "$tmpdir/data/ic_history.tsv"
done
cat > "$tmpdir/configs/baseline.yaml" <<'YAML'
research:
  expected_live_model: "w11"
YAML

mkdir -p "$tmpdir/scripts/lib"
cp "$PROJECT_DIR/scripts/lib/gpu-wake.sh" "$tmpdir/scripts/lib/gpu-wake.sh"
sed -e "s|^PROJECT_DIR=.*|PROJECT_DIR=\"$tmpdir\"|" \
    -e 's|^set -euo pipefail|set -uo pipefail|' \
    -e 's|wake_gpu \(.*\) 300|wake_gpu \1 5|' \
    "$RETRAIN_SCRIPT" > "$tmpdir/bin/retrain.sh"
chmod +x "$tmpdir/bin/retrain.sh"

cat > "$tmpdir/shims/wol" <<'SH'
#!/bin/bash
exit 0
SH
cat > "$tmpdir/shims/ping" <<'SH'
#!/bin/bash
exit 1
SH
cat > "$tmpdir/shims/ssh" <<'SH'
#!/bin/bash
echo "ssh $*" >> "$SHIM_LOG"
exit 1
SH
chmod +x "$tmpdir/shims/"*

export SHIM_LOG="$tmpdir/shim.log"
touch "$SHIM_LOG"

set +e
PATH="$tmpdir/shims:$PATH" bash "$tmpdir/bin/retrain.sh" >"$tmpdir/stdout.log" 2>&1
rc=$?
set -e

fail=0
if [ "$rc" -eq 0 ]; then
    echo "FAIL: offline GPU exited 0"
    fail=1
fi
if grep -q "GPU online" "$tmpdir"/logs/retrain-*.log 2>/dev/null; then
    echo "FAIL: logged GPU online while SSH never answered"
    fail=1
fi
if grep -q "^ssh" "$SHIM_LOG"; then
    echo "FAIL: training ssh ran after the wake failed"
    fail=1
fi
if [ -f "$tmpdir/data/last_retrain_date" ]; then
    echo "FAIL: sentinel written for a run that never reached the GPU"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: offline GPU exits $rc, no train, no sentinel"
    exit 0
fi
echo "--- log ---"
cat "$tmpdir"/logs/retrain-*.log 2>/dev/null || true
exit 1
