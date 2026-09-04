#!/bin/bash
# gpu-wake.sh -- shared Wake-on-LAN + readiness polling for the GPU host.
#
# Sourced by ashare-data-update.sh and ashare-pipeline.sh.  Both need the
# same wake sequence: send the magic packet, wait for ICMP, then wait for
# sshd (Windows answers ping well before sshd binds).
#
# wake_gpu HOST MAC USER [TOTAL_BUDGET_S]
#   0 = host reachable over SSH
#   1 = not reachable within budget (caller decides whether that is fatal)
#
# BUDGET covers the whole call, ping poll included.  A caller that cannot
# afford a five-minute stall on a dead host should pass a smaller one.

wake_gpu() {
    local host="$1" mac="$2" user="$3" budget="${4:-300}"
    local start=$SECONDS

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
