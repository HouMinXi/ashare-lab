#!/usr/bin/env bash
# PATH-shim tests for the deploy gate in ashare-retrain.sh.
# T1 normal deploy: NEW=w11, EXPECTED=w11 -> scp + ln called
# T2 shelved: NEW=w11, EXPECTED=w10 -> scp NOT called, sentinel written, llama restarted
# T3 rollback: NEW=w10, EXPECTED=w11 -> normal deploy (deploy gate allows)
# T4 EXPECTED missing -> normal deploy + warning
# T5 meta.json read failure -> normal deploy + warning
# T6 normal deploy but scp fails -> sentinel NOT written (forge-fix guard)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RETRAIN_SCRIPT="$PROJECT_DIR/scripts/ashare-retrain.sh"

PASS=0
FAIL=0

_run_test() {
    local new_model="$2"       # gpu-side meta.json model_file (empty = ssh fails)
    local expected_model="$3"  # baseline.yaml expected_live_model (empty = missing)
    local expect_scp="$4"      # "yes" or "no"
    local expect_sentinel="$5" # "yes" or "no"
    local expect_skip_log="$6" # "yes" or "no"
    local scp_fails="${7:-no}" # "yes" = scp shim exits 1
    local expect_llama="${8:-yes}"
    local expect_serve_now="${9:-no}"

    local tmpdir
    tmpdir=$(mktemp -d)

    # --- Setup project structure (mirror retrain's expectations) ---
    mkdir -p "$tmpdir"/{logs,data,models,configs}

    # Fake ic_history with 4 low-IC days to trigger gate
    for d in 1 2 3 4; do
        printf '2026-08-%02d\t0.005\n' "$d" >> "$tmpdir/data/ic_history.tsv"
    done

    # baseline.yaml
    if [ -n "$expected_model" ]; then
        cat > "$tmpdir/configs/baseline.yaml" <<YAML
research:
  expected_live_model: "$expected_model"
YAML
    fi

    # Pre-create models dir with a dummy w10.pt (existing production model)
    touch "$tmpdir/models/w10.pt"
    ln -sf w10.pt "$tmpdir/models/latest.pt"

    # --- Create a copy of retrain script with PROJECT_DIR overridden ---
    # Symlink script into tmpdir so SCRIPT_DIR resolves to tmpdir
    local script_copy="$tmpdir/bin"
    mkdir -p "$script_copy"
    # Copy the script and override PROJECT_DIR at the top
    sed "s|^PROJECT_DIR=.*|PROJECT_DIR=\"$tmpdir\"|" "$RETRAIN_SCRIPT" > "$script_copy/retrain.sh"
    chmod +x "$script_copy/retrain.sh"

    # --- Create PATH shims ---
    local shim_bin="$tmpdir/shims"
    mkdir -p "$shim_bin"

    export SHIM_LOG="$tmpdir/shim.log"
    touch "$SHIM_LOG"

    # wol: no-op
    cat > "$shim_bin/wol" <<'SH'
#!/bin/bash
echo "wol $@" >> "$SHIM_LOG"
SH
    chmod +x "$shim_bin/wol"

    # ping: always succeed
    cat > "$shim_bin/ping" <<'SH'
#!/bin/bash
echo "ping $@" >> "$SHIM_LOG"
exit 0
SH
    chmod +x "$shim_bin/ping"

    # ssh: respond to different commands
    cat > "$shim_bin/ssh" <<SH
#!/bin/bash
echo "ssh \$@" >> "\$SHIM_LOG"
# meta.json read
if echo "\$*" | grep -q "meta.json"; then
    if [ -n "$new_model" ]; then
        echo "$new_model"
    else
        exit 1
    fi
# train command
elif echo "\$*" | grep -q "train"; then
    exit 0
# nssm commands
elif echo "\$*" | grep -q "nssm"; then
    exit 0
else
    echo "ok"
fi
SH
    chmod +x "$shim_bin/ssh"

    # scp: succeed or fail per scenario
    cat > "$shim_bin/scp" <<SH
#!/bin/bash
echo "scp \$@" >> "\$SHIM_LOG"
exit $([ "$scp_fails" = "yes" ] && echo 1 || echo 0)
SH
    chmod +x "$shim_bin/scp"

    # flock: always succeed
    cat > "$shim_bin/flock" <<'SH'
#!/bin/bash
echo "flock $@" >> "$SHIM_LOG"
exit 0
SH
    chmod +x "$shim_bin/flock"

    # sleep: no-op (the script sleeps 5s after stopping llama-server)
    cat > "$shim_bin/sleep" <<'SH'
#!/bin/bash
exit 0
SH
    chmod +x "$shim_bin/sleep"

    # python3 shim: handle baseline.yaml parse, delegate rest to real python3
    cat > "$shim_bin/python3" <<SH
#!/bin/bash
echo "python3 \$@" >> "\$SHIM_LOG"
if echo "\$*" | grep -q "baseline.yaml"; then
    if [ -n "$expected_model" ]; then
        echo "$expected_model"
    fi
else
    /usr/bin/python3 "\$@"
fi
SH
    chmod +x "$shim_bin/python3"

    # --- Run retrain script (ignore exit code; we check behavior via shim log) ---
    PATH="$shim_bin:$PATH" bash "$script_copy/retrain.sh" \
        > "$tmpdir/stdout.log" 2>&1 || true

    # --- Verify ---
    local actual_scp="no"
    grep -q "^scp" "$SHIM_LOG" 2>/dev/null && actual_scp="yes"

    local actual_sentinel="no"
    [ -f "$tmpdir/data/last_retrain_date" ] && actual_sentinel="yes"

    local actual_skip_log="no"
    grep -q "skipping Step 4" "$tmpdir/logs/retrain-"*.log 2>/dev/null && actual_skip_log="yes"

    local actual_llama="no"
    grep -q "nssm.*start" "$SHIM_LOG" 2>/dev/null && actual_llama="yes"

    local test_passed="yes"

    if [ "$actual_scp" != "$expect_scp" ]; then
        echo "  FAIL: scp called=$actual_scp, expected=$expect_scp"
        test_passed="no"
    fi
    if [ "$actual_sentinel" != "$expect_sentinel" ]; then
        echo "  FAIL: sentinel=$actual_sentinel, expected=$expect_sentinel"
        test_passed="no"
    fi
    if [ "$expect_skip_log" = "yes" ] && [ "$actual_skip_log" != "yes" ]; then
        echo "  FAIL: skip_log=$actual_skip_log, expected=yes"
        test_passed="no"
    fi
    if [ "$actual_llama" != "$expect_llama" ]; then
        echo "  FAIL: llama called=$actual_llama, expected=$expect_llama"
        test_passed="no"
    fi
    if [ "$expect_serve_now" = "yes" ]; then
        if ! grep -q "serve-now" "$tmpdir"/logs/retrain-*.log 2>/dev/null; then
            echo "  FAIL: serve-now skip log missing"
            test_passed="no"
        fi
        if grep -q "^wol" "$SHIM_LOG" 2>/dev/null; then
            echo "  FAIL: wol ran after serve-now skip"
            test_passed="no"
        fi
    fi

    if [ "$test_passed" = "yes" ]; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAILED"
        FAIL=$((FAIL + 1))
        # Dump shim log for debugging
        echo "  SHIM_LOG:" && cat "$SHIM_LOG" | head -20
    fi

    rm -rf "$tmpdir"
}

echo "=== Deploy gate tests ==="
echo ""

echo "T1: normal deploy (NEW=w11, EXPECTED=w11)"
_run_test "T1" "w11.pt" "w11" "yes" "yes" "no"

echo "T2: shelved (NEW=w11, EXPECTED=w10)"
_run_test "T2" "w11.pt" "w10" "no" "yes" "yes"

echo "T3: rollback (NEW=w10, EXPECTED=w11)"
_run_test "T3" "w10.pt" "w11" "yes" "yes" "no"

echo "T4: EXPECTED missing"
_run_test "T4" "w11.pt" "" "yes" "yes" "no"

echo "T5: meta.json read failure"
_run_test "T5" "" "w10" "yes" "yes" "no"

echo "T6: normal deploy + scp failure (sentinel must NOT be written)"
_run_test "T6" "w11.pt" "w11" "yes" "no" "no" "yes"

echo "T7: serve-now live (EXPECTED=w115) skips --force all-windows"
_run_test "T7" "w11.pt" "w115" "no" "no" "no" "no" "no" "yes"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
