"""The selective GPU transfer must carry enough code for the real train CLI."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
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
        'scp() { cp -r "${@:2:$#-2}" "$RECEIVER/"; }\n'
        "ssh() { printf '%s\\n' provider-uri; }\n"
        + function + "\nsync_code_to_gpu\n",
        encoding="utf-8",
    )
    synced = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=20)
    assert synced.returncode == 0, synced.stdout + synced.stderr
    assert not staging.exists()
    # The actual GPU checkout already contains the data package. Supply that
    # unchanged sibling while leaving research exactly as sent by the manifest.
    shutil.copytree(
        ROOT / "ashare_lab/data", receiver / "ashare_lab/data",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = dict(os.environ, PYTHONPATH=str(receiver), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "ashare_lab.research.train", "--help"],
        cwd=receiver, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--train-end" in result.stdout
    assert "--live" in result.stdout
    assert "--serve-now" in result.stdout
