#!/bin/bash
# gpu-wake.sh -- shared Wake-on-LAN + readiness polling for the GPU host.
#
# Sourced by ashare-data-update.sh and ashare-pipeline.sh.  Both need the
# same wake sequence: send the magic packet, wait for ICMP, then wait for
# sshd (Windows answers ping well before sshd binds).
#
# The host identity lives here too, so a hardware swap is a one-file edit
# rather than two files that must stay in lockstep.
# shellcheck disable=SC2034  # consumed by the scripts that source this file
GPU_HOST="192.168.100.11"
# shellcheck disable=SC2034
GPU_MAC="04:7C:16:49:BE:32"
# shellcheck disable=SC2034
GPU_USER="admin"

# wake_gpu HOST MAC USER [TOTAL_BUDGET_S]
#   0 = host reachable over SSH
#   1 = not reachable within budget (caller decides whether that is fatal)
#
# BUDGET covers the whole call, ping poll included.  A caller that cannot
# afford a five-minute stall on a dead host should pass a smaller one.

wake_gpu() {
    local host="$1" mac="$2" user="$3" budget="${4:-300}"
    local start=$SECONDS

    # A non-numeric budget makes the deadline comparison error out and
    # evaluate false; a zero budget makes it false on the first pass.
    # Either way no poll runs and the caller is told the host is
    # unreachable, hiding what is really a bad argument.  An empty budget
    # is not this case: ${4:-300} substitutes the default.
    if ! [[ "$budget" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: wake_gpu budget must be a positive integer, got '$budget'"
        return 1
    fi

    # Without this the magic packet silently never goes out: bash prints
    # "wol: command not found" to stderr, the ping poll still runs, and a
    # host that happens to be awake makes the misconfiguration look fine.
    if ! command -v wol >/dev/null 2>&1; then
        echo "ERROR: wol not installed; cannot wake $host"
        return 1
    fi

    # An unchecked wol turns "the packet never went out" (wrong interface,
    # no permission) into the generic unreachable error 200s later.
    if ! wol "$mac"; then
        echo "ERROR: wol failed to send the magic packet to $mac"
        return 1
    fi

    local ping_ok=0
    while [ $(( SECONDS - start )) -lt "$budget" ]; do
        if ping -c1 -W1 "$host" >/dev/null 2>&1; then
            ping_ok=1
            break
        fi
        sleep 5
    done
    if [ "$ping_ok" -eq 0 ]; then
        echo "ERROR: GPU not reachable within ${budget}s ping poll"
        return 1
    fi

    while ! ssh -o ConnectTimeout=3 -o BatchMode=yes "${user}@${host}" \
            "echo ok" >/dev/null 2>&1; do
        if [ $(( SECONDS - start )) -ge "$budget" ]; then
            echo "ERROR: SSH not ready within ${budget}s"
            return 1
        fi
        sleep 5
    done

    return 0
}
