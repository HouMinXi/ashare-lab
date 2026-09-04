#!/bin/bash
# ashare-data-update.sh -- Timer 1 wrapper (17:00 CST)
# Updates market data, writes atomic sentinel stamp, alerts on failure.
# No set -e: explicit error checks preserve retry/alert flow (R4H1).
set -uo pipefail

REPO="$HOME/code/ashare-lab"
STAMP="$HOME/.cache/ashare-data-update.stamp"
STDERR_LOG="/tmp/ashare-data-update-stderr.log"
SYNC_LOG="/tmp/ashare-gpu-sync.log"
SECONDS_START=$SECONDS
GPU_HOST="192.168.100.11"
GPU_MAC="04:7C:16:49:BE:32"
GPU_USER="admin"
QLIB_DIR="$HOME/.qlib/qlib_data"

# shellcheck source=scripts/lib/gpu-wake.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/gpu-wake.sh"

sync_gpu_data() {
    local sync_dir="${HOME}/.cache/ashare-sync"
    mkdir -p "$sync_dir"
    local tarball="${sync_dir}/cn_data_sync.tar.gz"
    trap 'rm -f "$tarball"' RETURN

    # The GPU box sleeps between runs.  Without this wake the scp below
    # fails with "No route to host", the sync is skipped, and the 18:00
    # pipeline then finds stale qlib data and records a stale run.
    if ! wake_gpu "$GPU_HOST" "$GPU_MAC" "$GPU_USER"; then
        echo "sync_gpu_data: GPU wake failed"
        return 1
    fi

    if ! tar czf "$tarball" -C "$QLIB_DIR" cn_data; then
        echo "sync_gpu_data: tar failed"
        return 1
    fi
    if ! scp "$tarball" "${GPU_USER}@${GPU_HOST}:H:/.qlib/qlib_data/"; then
        echo "sync_gpu_data: scp failed"
        return 1
    fi
    if ! ssh "${GPU_USER}@${GPU_HOST}" "cd /d H:\.qlib\qlib_data && rmdir /s /q cn_data 2>nul && tar xzf cn_data_sync.tar.gz && del cn_data_sync.tar.gz"; then
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
    # Capture the sync's own diagnostics: alert.py reports the tail of a
    # log file, and STDERR_LOG only holds fetch-today's output.  Without a
    # dedicated log the alert says a sync failed but not which step.
    if ! sync_gpu_data 2>&1 | tee "$SYNC_LOG"; then
        echo "WARNING: GPU data sync failed, GPU keeps stale data"
        # Without this the failure is invisible until the 18:00 pipeline
        # records a stale run 15 minutes later, pointing at the wrong layer.
        python3 "$REPO/scripts/alert.py" "1" "gpu_data_sync" \
            "$((SECONDS - SECONDS_START))" "$SYNC_LOG" || true
    fi
fi

if [ $rc -ne 0 ]; then
    python3 "$REPO/scripts/alert.py" "$rc" "data_update" "$((SECONDS - SECONDS_START))" "$STDERR_LOG" || true
fi

exit $rc

fi
