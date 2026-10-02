#!/usr/bin/env bash
# ashare-wol.sh must exit non-zero when gpu-win never answers. The old
# unit ran wol directly, and wol exits 0 after transmitting the packet.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p "$tmpdir/scripts/lib" "$tmpdir/shims"

cp "$PROJECT_DIR/scripts/ashare-wol.sh" "$tmpdir/scripts/ashare-wol.sh"
cp "$PROJECT_DIR/scripts/lib/gpu-wake.sh" "$tmpdir/scripts/lib/gpu-wake.sh"
chmod +x "$tmpdir/scripts/ashare-wol.sh"

printf '#!/bin/bash\nexit 0\n' > "$tmpdir/shims/wol"
printf '#!/bin/bash\nexit 1\n' > "$tmpdir/shims/ping"
printf '#!/bin/bash\necho "ssh ran"\nexit 1\n' > "$tmpdir/shims/ssh"
chmod +x "$tmpdir/shims/"*

GPU_WAKE_BUDGET=5 PATH="$tmpdir/shims:$PATH" \
    bash "$tmpdir/scripts/ashare-wol.sh" >"$tmpdir/out" 2>&1
rc=$?

fail=0
if [ "$rc" -eq 0 ]; then
    echo "FAIL: offline gpu-win exited 0"
    fail=1
fi
if grep -q "ssh ready" "$tmpdir/out"; then
    echo "FAIL: reported ssh ready"
    fail=1
fi
if ! grep -q "did not come up" "$tmpdir/out"; then
    echo "FAIL: missing failure line"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: offline gpu-win exits $rc"
    exit 0
fi
cat "$tmpdir/out"
exit 1
