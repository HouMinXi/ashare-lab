#!/bin/bash
# ashare-data-update.sh -- Timer 1 wrapper (17:00 CST)
# Updates market data, writes atomic sentinel stamp, alerts on failure.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).
set -uo pipefail

REPO="$HOME/code/ashare-lab"
STAMP="$HOME/.cache/ashare-data-update.stamp"
STDERR_LOG="/tmp/ashare-data-update-stderr.log"
SECONDS_START=$SECONDS
GPU_HOST="192.168.100.11"
GPU_USER="admin"
QLIB_DIR="$HOME/.qlib/qlib_data"

sync_gpu_data() {
    local tarball="/tmp/cn_data_sync.tar.gz"
    trap 'rm -f "$tarball"' RETURN

    if ! tar czf "$tarball" -C "$QLIB_DIR" cn_data; then
        echo "sync_gpu_data: tar failed"
        return 1
    fi
    if ! scp "$tarball" "${GPU_USER}@${GPU_HOST}:H:/.qlib/qlib_data/"; then
        echo "sync_gpu_data: scp failed"
        return 1
    fi
    if ! ssh "${GPU_USER}@${GPU_HOST}" "cd /d H:/.qlib/qlib_data && tar xzf cn_data_sync.tar.gz && del cn_data_sync.tar.gz"; then
        echo "sync_gpu_data: ssh extract failed"
        return 1
    fi
}

if [ "${ASHARE_DATA_UPDATE_TESTING:-}" != "1" ]; then

mkdir -p "$HOME/.cache"
cd "$REPO" || exit 1
rm -f "$STAMP"

rc=1
for attempt in 1 2; do
    python3 -m ashare_lab.cli fetch-today 2>"$STDERR_LOG"
    rc=$?
    if [ $rc -eq 0 ]; then
        date -Iseconds > "${STAMP}.tmp"
        mv "${STAMP}.tmp" "$STAMP"
        break
    fi
    if [ $attempt -eq 1 ]; then
        echo "Data update attempt 1 failed (rc=$rc), retrying in 5min"
        sleep 300
    fi
done

if [ $rc -eq 0 ]; then
    sync_gpu_data || echo "WARNING: GPU data sync failed, GPU keeps stale data"
fi

if [ $rc -ne 0 ]; then
    python3 "$REPO/scripts/alert.py" "$rc" "data_update" "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
fi

exit $rc

fi
