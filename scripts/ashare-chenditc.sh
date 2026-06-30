#!/bin/bash
# ashare-chenditc.sh -- Timer 3 wrapper (21:00 CST)
# Snapshot incremental close prices, run chenditc full refresh, diff check.
# Best-effort: no retry, no alert, no GPU sync.
set -uo pipefail

REPO="$HOME/code/ashare-lab"
STDERR_LOG="/tmp/ashare-chenditc-stderr.log"

cd "$REPO" || exit 1

# Three separate processes to avoid qlib memmap corruption:
# 1. Snapshot current incremental close prices
python3 -m ashare_lab.cli chenditc-snapshot 2>>"$STDERR_LOG"

# 2. Full chenditc refresh (replaces cn_data entirely)
python3 -m ashare_lab.cli update 2>>"$STDERR_LOG"

# 3. Diff incremental vs refreshed data
python3 -m ashare_lab.cli chenditc-diff 2>>"$STDERR_LOG"
