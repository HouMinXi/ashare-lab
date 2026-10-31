#!/usr/bin/env bash
# External deadman: runs on Z66 via cron, checks W1 pipeline ran today.
# Alerts via iLink (reuses alert.py's path) if stale.
# Does NOT touch pipeline.sh or any production code on W1.
set -euo pipefail

W1="houminxi@192.168.100.10"
DB="/home/houminxi/code/ashare-lab/paper.db"
ALERT_SCRIPT="/home/houminxi/code/ashare-lab/scripts/alert.py"

# ponytail: weekday-only check -- no exchange_calendars on Z66.
# Misses CN holidays (acceptable at this scale; add XSHG list if needed).
DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    exit 0
fi

TODAY=$(date +%Y-%m-%d)

# Query W1's pipeline_runs for today's row via SSH + Python (no sqlite3 CLI on W1)
ROW=$(ssh -o ConnectTimeout=10 "$W1" \
    "python3 -c \"import sqlite3; c=sqlite3.connect('${DB}'); r=c.execute('SELECT status FROM pipeline_runs WHERE trade_date=? ORDER BY created_at DESC, id DESC LIMIT 1',('${TODAY}',)).fetchone(); print(r[0] if r else 'MISSING')\"" \
    2>/dev/null) || ROW="SSH_FAIL"

case "$ROW" in
    success)
        exit 0
        ;;
    MISSING|SSH_FAIL|error|stale)
        # Alert via the existing alert.py on W1 (it has the iLink secrets)
        ssh -o ConnectTimeout=10 "$W1" \
            "python3 ${ALERT_SCRIPT} 1 'deadman: pipeline_runs ${ROW} for ${TODAY}' 0" \
            2>/dev/null || true
        ;;
esac
