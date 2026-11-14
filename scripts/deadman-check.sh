#!/bin/bash
# ashare deadman watchdog, run FROM z66 against the W1 host.
#
# The point of a deadman is to speak up when the monitored host cannot.
# The previous version alerted by sshing into W1 and running alert.py there,
# so a dead host silenced its own alarm; it also logged to /tmp, which Fedora's
# tmpfiles timer wipes, and ran under bare `set -e`, so an early ssh failure
# left no trace at all. This version alerts locally, keeps its log outside
# /tmp, and reports every exit path.
#
# usage: deadman-check.sh [--dry-run]  (crontab on z66, weekdays 21:30)
set -uo pipefail

TARGET_HOST="${ASHARE_TARGET_HOST:-192.168.100.10}"
TARGET_USER="${ASHARE_TARGET_USER:-houminxi}"
STATE_DIR="${ASHARE_DEADMAN_STATE:-$HOME/.local/state/ashare-deadman}"
LOG="$STATE_DIR/deadman.log"
STAMP="$STATE_DIR/last-alert.stamp"
MAX_LOG_BYTES=$((512 * 1024))
# Re-alert at most this often for a condition that stays broken. The cron runs
# once a day, so in practice a failure that persists pages once per run; this
# value only suppresses repeats if the schedule is ever tightened below a day.
REALERT_SECONDS="${ASHARE_REALERT_SECONDS:-86400}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

# cron runs with a minimal PATH that does not include ~/.local/bin, so a
# watchdog trusting PATH would detect a fault and then fail to report it.
# Resolve the outlet explicitly and refuse to run blind.
PATH="$HOME/.local/bin:$PATH"
HERMES_BIN="${ASHARE_HERMES_BIN:-}"
if [ -z "$HERMES_BIN" ]; then
  for cand in "$HOME/.local/bin/hermes" "$HOME/.hermes/hermes-agent/venv/bin/hermes" "$(command -v hermes 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && HERMES_BIN="$cand" && break
  done
fi

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "usage: ${0##*/} [--dry-run]" >&2; exit 2 ;;
esac

mkdir -p "$STATE_DIR"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$LOG"; }

rotate_log() {
  [ -f "$LOG" ] || return 0
  local size
  size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  [ "$size" -gt "$MAX_LOG_BYTES" ] || return 0
  mv -f "$LOG" "$LOG.1"
  log "log rotated at ${size} bytes"
}

# Alert from THIS machine. Never route through the host being checked.
notify() {
  local subject="$1" body="$2"
  if [ "$DRY_RUN" = 1 ]; then
    log "DRY-RUN would alert: $subject"
    printf 'DRY-RUN alert: %s\n%s\n' "$subject" "$body"
    return 0
  fi
  local now last=0
  now=$(date +%s)
  [ -f "$STAMP" ] && last=$(cat "$STAMP" 2>/dev/null || echo 0)
  case "$last" in ''|*[!0-9]*) last=0 ;; esac
  if [ $((now - last)) -lt "$REALERT_SECONDS" ]; then
    log "alert suppressed (last sent $((now - last))s ago): $subject"
    return 0
  fi
  if [ -z "$HERMES_BIN" ]; then
    log "ALERT IMPOSSIBLE: no hermes binary found; would have reported: $subject"
    echo "ashare deadman found no alert outlet; wanted to report: $subject" >&2
    return 1
  fi
  if timeout 60 "$HERMES_BIN" send -t qqbot "ashare deadman: $subject

$body" >/dev/null 2>&1; then
    printf '%s\n' "$now" >"$STAMP"
    log "alert sent: $subject"
  else
    # Losing the outlet is itself worth recording; stderr reaches cron mail.
    log "ALERT DELIVERY FAILED: $subject"
    echo "ashare deadman could not deliver alert: $subject" >&2
  fi
}

# Any unexpected exit must leave a trace rather than dying quietly. LINENO
# inside a trap reports the trap's own line, so record the status only.
# shellcheck disable=SC2329  # invoked indirectly via trap
on_exit() {
  local rc=$?
  [ "$rc" -ne 0 ] && log "exit rc=$rc"
  return 0
}
trap on_exit EXIT

rotate_log
log "check start target=$TARGET_HOST"

# Reachability first: distinguish "host is gone" from "job is broken", because
# the two need different responses and only the first one can hide itself.
if ! timeout 20 ssh "${SSH_OPTS[@]}" "$TARGET_USER@$TARGET_HOST" true 2>/dev/null; then
  log "FAIL ssh unreachable"
  notify "host unreachable" "Cannot ssh to $TARGET_USER@$TARGET_HOST. Pipeline state unknown; the host cannot report on itself."
  exit 1
fi

# Pull the facts in one round trip. The probe is a file on the remote host so
# its single-quoted SQL survives; key=value output means a partial response
# cannot be mistaken for a healthy one.
PROBE_SRC="$(dirname "$(readlink -f "$0")")/ashare-probe-remote.sh"
PROBE_DST=".cache/ashare-probe-remote.sh"
if [ -f "$PROBE_SRC" ]; then
  # Refresh the copy every run so the probe cannot drift from this watchdog.
  timeout 30 ssh "${SSH_OPTS[@]}" "$TARGET_USER@$TARGET_HOST" "mkdir -p .cache" 2>/dev/null
  timeout 30 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$PROBE_SRC" "$TARGET_USER@$TARGET_HOST:$PROBE_DST" 2>/dev/null \
    || log "warn: probe upload failed, using existing copy"
fi
remote=$(timeout 60 ssh "${SSH_OPTS[@]}" "$TARGET_USER@$TARGET_HOST" \
  "bash $PROBE_DST" 2>/dev/null) \
  || { log "FAIL remote probe errored"; notify "probe failed" "ssh succeeded but the status probe returned nothing."; exit 1; }

get() { printf '%s\n' "$remote" | sed -n "s/^$1=//p" | head -1; }
repo=$(get repo); last_settled=$(get last_success)
failed_units=$(get failed_units); ic_last=$(get ic_last)
recent_bad=$(get recent_bad); latest_fresh=$(get latest_fresh)

log "probe repo=$repo last_success=$last_settled failed_units=$failed_units ic_last=$ic_last recent_bad=$recent_bad latest_fresh=$latest_fresh"

problems=""
[ "$repo" != ok ] && problems+="- ashare-lab repo not found on the host"$'\n'

# An empty field means the probe could not answer, which is not the same as a
# clean answer. A locked database or a renamed column produces empty values, and
# a `-n` guard silently turns that into "healthy".
if [ -z "$failed_units" ]; then
  problems+="- unit state unknown (the probe returned no failed_units)"$'\n'
elif [ "$failed_units" != 0 ]; then
  problems+="- $failed_units ashare unit(s) in failed state"$'\n'
fi

if [ -z "$recent_bad" ]; then
  problems+="- run history unknown (the probe returned no recent_bad)"$'\n'
elif [ "$recent_bad" != 0 ]; then
  problems+="- $recent_bad stale/error pipeline run(s) in the last 7 days"$'\n'
fi

# A run can report success while serving yesterday's predictions, so freshness
# is checked separately from status.
if [ -z "$latest_fresh" ]; then
  problems+="- prediction freshness unknown (the probe returned no latest_fresh)"$'\n'
elif [ "$latest_fresh" = 0 ]; then
  problems+="- latest run did not use same-day predictions"$'\n'
fi

# Days between a date string and now, or the word "unknown" when the value does
# not parse. Falling back to "now" made a corrupt date indistinguishable from a
# fresh one.
age_in_days() {
  local value="$1" seconds
  seconds=$(date -d "$value" +%s 2>/dev/null) || { echo unknown; return 0; }
  echo $(( ( $(date +%s) - seconds ) / 86400 ))
}

# IC history stops updating whenever the pipeline falls back to stale
# predictions. The retrain no longer keys on it -- the retrain is weekly -- but
# the signal monitor reads this series, and a frozen series makes the monitor
# look calm.
if [ -n "$ic_last" ] && [ "$ic_last" != none ]; then
  # The series runs five trading days behind by construction: a score is only
  # judged once five sessions of forward return exist. Ten calendar days covers
  # five sessions plus two weekends.
  ic_age=$(age_in_days "$ic_last")
  if [ "$ic_age" = unknown ]; then
    problems+="- IC history date unreadable ($ic_last)"$'\n'
  elif [ "$ic_age" -gt 10 ]; then
    problems+="- IC history last updated ${ic_age} days ago ($ic_last); signal monitor is running on stale data"$'\n'
  fi
else
  problems+="- IC history unreadable or empty"$'\n'
fi

# A settled run that is several days old means the pipeline stopped producing.
# Six days leaves room for a long weekend plus one holiday; a longer break is
# what the IC check above is for.
if [ -n "$last_settled" ] && [ "$last_settled" != none ]; then
  age_days=$(age_in_days "$last_settled")
  if [ "$age_days" = unknown ]; then
    problems+="- last run date unreadable ($last_settled)"$'\n'
  elif [ "$age_days" -gt 6 ]; then
    problems+="- last successful run is ${age_days} days old ($last_settled)"$'\n'
  fi
else
  problems+="- no successful pipeline run found"$'\n'
fi

if [ -n "$problems" ]; then
  log "UNHEALTHY"
  notify "pipeline needs attention" "$problems"
  exit 1
fi

log "OK"
exit 0
