#!/usr/bin/env bash
# PATH-shim tests for the idempotency guard in ashare-pipeline.sh.
# T1 today's settled row exists -> exit before any ssh/scp work
# T2 only old rows -> proceeds into gpu phase (ssh called)
# T3 paper.db missing -> warning + proceeds (fail-open)
# T4 today stale (pipeline_runs only, not settled) -> retry proceeds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_SCRIPT="$PROJECT_DIR/scripts/ashare-pipeline.sh"

PASS=0
FAIL=0

_run_test() {
    local name="$1"          # test label
    local db_mode="$2"       # "today-success" | "old-only" | "missing"
    local expect_ssh="$3"    # "yes" = gpu phase reached, "no" = early exit

    local tmpdir
    tmpdir=$(mktemp -d)
    mkdir -p "$tmpdir"/{logs,data,models,configs,predictions}

    # --- Fixture paper.db ---
    if [ "$db_mode" != "missing" ]; then
        /usr/bin/python3 - "$tmpdir/paper.db" "$db_mode" <<'PY'
import sqlite3, sys
path, mode = sys.argv[1], sys.argv[2]
db = sqlite3.connect(path)
db.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, trade_date TEXT, status TEXT)")
db.execute("CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, trade_date TEXT, status TEXT)")
if mode == "today-settled":
    db.execute("INSERT INTO runs (trade_date, status) VALUES ('2026-08-10', 'settled')")
elif mode == "old-only":
    db.execute("INSERT INTO runs (trade_date, status) VALUES ('2026-08-07', 'settled')")
elif mode == "today-stale":
    # stale night: pipeline_runs marks the day but runs.settled is absent
    db.execute("INSERT INTO pipeline_runs (trade_date, status) VALUES ('2026-08-10', 'stale')")
db.commit()
PY
    fi

    # --- Script copy with REPO / STDERR_LOG overridden to tmpdir ---
    local script_copy="$tmpdir/bin"
    mkdir -p "$script_copy"
    sed -e "s|^REPO=.*|REPO=\"$tmpdir\"|" \
        -e "s|^STDERR_LOG=.*|STDERR_LOG=\"$tmpdir/stderr.log\"|" \
        "$PIPELINE_SCRIPT" > "$script_copy/pipeline.sh"
    chmod +x "$script_copy/pipeline.sh"

    # --- PATH shims ---
    local shim_bin="$tmpdir/shims"
    mkdir -p "$shim_bin"
    export SHIM_LOG="$tmpdir/shim.log"
    touch "$SHIM_LOG"

    for tool in wol ping ssh scp timeout; do
        cat > "$shim_bin/$tool" <<'SH'
#!/bin/bash
echo "$(basename $0) $@" >> "$SHIM_LOG"
# ssh probes that gate the gpu phase must succeed
case " $* " in
    *" echo ok "*) echo ok;;
    *qlib*)        echo yes;;   # gpu_has_date calendar probe
esac
exit 0
SH
        chmod +x "$shim_bin/$tool"
    done

    # python3 shim: calendar answers are canned; the guard's sqlite query
    # delegates to the real interpreter against the fixture db.
    cat > "$shim_bin/python3" <<'SH'
#!/bin/bash
case " $* " in
    *latest_trading_day*) echo "2026-08-10"; exit 0;;
    *is_trading_day*)     echo "True";       exit 0;;
    *"FROM runs"*)       exec /usr/bin/python3 "$@";;
    *)                    exit 0;;
esac
SH
    chmod +x "$shim_bin/python3"

    PATH="$shim_bin:$PATH" __PIPELINE_LOGGING=1 bash "$script_copy/pipeline.sh" \
        > "$tmpdir/stdout.log" 2>&1 || true

    local actual_ssh="no"
    grep -q "^ssh" "$SHIM_LOG" 2>/dev/null && actual_ssh="yes"

    local test_passed="yes"
    if [ "$actual_ssh" != "$expect_ssh" ]; then
        echo "  FAIL: ssh called=$actual_ssh, expected=$expect_ssh"
        test_passed="no"
    fi
    if [ "$expect_ssh" = "no" ]; then
        grep -q "Already settled today" "$tmpdir/stdout.log" || {
            echo "  FAIL: missing 'Already settled today' log"; test_passed="no"; }
    fi
    if [ "$db_mode" = "missing" ]; then
        grep -q "idempotency check failed" "$tmpdir/stdout.log" || {
            echo "  FAIL: missing fail-open warning"; test_passed="no"; }
    fi

    if [ "$test_passed" = "yes" ]; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAILED ($name)"
        FAIL=$((FAIL + 1))
        echo "  stdout:" && tail -5 "$tmpdir/stdout.log"
    fi

    rm -rf "$tmpdir"
}

echo "=== Pipeline idempotency guard tests ==="
echo ""

echo "T1: today's settled row exists -> early exit, no gpu work"
_run_test "T1" "today-settled" "no"

echo "T2: only old rows -> proceeds"
_run_test "T2" "old-only" "yes"

echo "T3: paper.db missing -> warning + proceeds"
_run_test "T3" "missing" "yes"

echo "T4: today stale (not settled) -> stale->fresh retry proceeds"
_run_test "T4" "today-stale" "yes"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
