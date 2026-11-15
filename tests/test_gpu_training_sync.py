"""The selective GPU transfer must carry enough code for the real train CLI."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_selective_gpu_sync_bootstraps_train_cli(tmp_path):
    source = (ROOT / "scripts/ashare-pipeline.sh").read_text()
    function = source[source.index("sync_code_to_gpu() {"):source.index("# ---- Step 3:")]
    # Never source or execute the pipeline: it has live side effects at top level.
    # Only the exact sync function runs. Its scratch path and transport are local.
    staging = tmp_path / "staging"
    function, count = re.subn(r'local STAGING="[^"]+"', f"local STAGING={shlex.quote(str(staging))}", function)
    assert count == 1
    receiver = tmp_path / "gpu-received"
    receiver.mkdir()
    script = tmp_path / "sync.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"REPO={shlex.quote(str(ROOT))}\n"
        f"RECEIVER={shlex.quote(str(receiver))}\n"
        "GPU_USER=fixture\nGPU_HOST=fixture\n"
        'scp() {\n'
        '  local srcs=() dst=""\n'
        '  while [ $# -gt 0 ]; do\n'
        '    case "$1" in\n'
        '      -o) shift 2 ;;\n'
        '      -*) shift ;;\n'
        '      *) srcs+=("$1"); shift ;;\n'
        '    esac\n'
        '  done\n'
        '  dst="${srcs[-1]}"\n'
        '  unset \'srcs[-1]\'\n'
        '  cp -r "${srcs[@]}" "$RECEIVER/"\n'
        '}\n'
        "ssh() { return 0; }\n"
        'timeout() { shift; "$@"; }\n'
        + function + "\nsync_code_to_gpu\n",
        encoding="utf-8",
    )
    synced = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=20)
    assert synced.returncode == 0, synced.stdout + synced.stderr
    assert not staging.exists()
    # The sync ships the whole package, so the receiver stands on its own.
    assert (receiver / "ashare_lab/research/sequence_dataset.py").exists()
    assert (receiver / "ashare_lab/data").is_dir()
    # ... and it records what it shipped, so the far side can tell fresh from
    # stale. The manifest must cover every file that landed.
    manifest = receiver / "deploy-manifest.sha256"
    assert manifest.is_file(), "the sync must write a manifest"
    listed = {
        line.split(None, 1)[1].strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    shipped = {
        path.relative_to(receiver).as_posix()
        for path in receiver.rglob("*")
        if path.is_file() and path.name != "deploy-manifest.sha256"
    }
    assert listed == shipped, f"manifest misses {sorted(shipped - listed)[:5]}"
    assert (receiver / "deploy_verify.py").is_file()
    env = dict(os.environ, PYTHONPATH=str(receiver), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "ashare_lab.research.train", "--help"],
        cwd=receiver, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--train-end" in result.stdout
    assert "--live" in result.stdout
    assert "--serve-now" in result.stdout


def test_the_code_sync_leaves_no_errexit_behind(tmp_path):
    # ashare-pipeline.sh runs without errexit on purpose: later steps test
    # command status themselves. A `set -e` left inside sync_code_to_gpu
    # would make the first non-zero (a missing GPU log line, a stopped
    # llama-server, a failed inference) exit the whole run with no alert.
    source = (ROOT / "scripts" / "ashare-pipeline.sh").read_text()
    start = source.index("sync_code_to_gpu() {")
    function = source[start: source.index("\n}\n", start) + 3]
    receiver = tmp_path / "receiver"
    receiver.mkdir()
    script = tmp_path / "sync.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"REPO={shlex.quote(str(ROOT))}\n"
        f"RECEIVER={shlex.quote(str(receiver))}\n"
        "GPU_USER=fixture\nGPU_HOST=fixture\n"
        'scp() { return 0; }\n'
        'ssh() { return 0; }\n'
        'timeout() { shift; "$@"; }\n'
        + function
        + "\nsync_code_to_gpu\nfalse\necho ALIVE\n",
        encoding="utf-8",
    )

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)

    assert "ALIVE" in result.stdout, result.stdout + result.stderr
