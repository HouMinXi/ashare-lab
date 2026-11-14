#!/usr/bin/env bash
# 17:30 CST heads-up wake. The timer used to exec wol directly, which
# exits 0 as soon as the magic packet is on the wire. A powered-off or
# unplugged gpu-win then looked like a clean run. Wait for SSH and fail
# if it never comes up, so the journal shows the miss.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib/gpu-wake.sh
source "$SCRIPT_DIR/lib/gpu-wake.sh"

BUDGET="${GPU_WAKE_BUDGET:-90}"
if ! wake_gpu "$GPU_HOST" "$GPU_MAC" "$GPU_USER" "$BUDGET"; then
    echo "gpu-win did not come up within ${BUDGET}s"
    exit 1
fi
echo "gpu-win ssh ready"
exit 0
