#!/bin/bash
# Remote status probe for the ashare deadman watchdog. Lives on the monitored
# host and prints key=value lines; the watchdog on the other machine reads them.
#
# Kept as its own file because the queries need single-quoted SQL literals,
# which do not survive being nested inside an ssh command string.
set -uo pipefail

cd "$HOME/code/ashare-lab" 2>/dev/null || { echo "repo=missing"; exit 0; }
echo "repo=ok"

# Read-only URI so a wrong path reports missing instead of silently creating
# an empty database next to the real one.
DB="file:paper.db?mode=ro"
q() { sqlite3 "$DB" "$1" 2>/dev/null; }

if [ -f paper.db ]; then
  echo "last_success=$(q "select ifnull(max(trade_date),'none') from pipeline_runs where status='success'")"
  echo "recent_bad=$(q "select count(*) from pipeline_runs where status in ('stale','error') and trade_date >= date('now','-7 days')")"
  echo "latest_fresh=$(q "select count(*) from pipeline_runs where status='success' and trade_date=predictions_date and trade_date=(select max(trade_date) from pipeline_runs)")"
else
  echo "ledger=missing"
  echo "last_success=none"
  echo "recent_bad=0"
  echo "latest_fresh=0"
fi

echo "failed_units=$(systemctl --user list-units --state=failed --no-legend 2>/dev/null | grep -c ashare)"
ic=$(tail -1 data/ic_history.tsv 2>/dev/null | cut -f1)
echo "ic_last=${ic:-none}"
