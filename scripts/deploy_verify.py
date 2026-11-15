"""Verify a deployment: every shipped file matches, then the predictor imports.

Run on the target host from the deployment root:

    py deploy_verify.py

The manifest is written next to this script by the sync step and lists
``<sha256>  <relative path>`` for everything that was staged. Only listed files
are checked: the remote tree also holds models and predictions, so a file that
is no longer in the repository is left alone rather than deleted.
"""

from __future__ import annotations

import hashlib
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "deploy-manifest.sha256"

# Modules predict_for_date reaches for. Importing them proves the runtime
# closure is present, which importing the entry module alone does not: its
# body is deliberately stdlib-only.
CLOSURE = (
    "ashare_lab.research.predict",
    "ashare_lab.research.blend",
    "ashare_lab.research.train",
    "ashare_lab.research.smoke_test",
    "ashare_lab.research.model_meta",
    "ashare_lab.data.update",
    "ashare_lab.research.sequence_dataset",
    "ashare_lab.bridge",
)


def _revision():
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("# revision "):
            return line.split(None, 2)[2].strip()
    return "unknown"


def _entries():
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            yield digest, name.strip()


def _mismatches():
    missing, changed = [], []
    for digest, name in _entries():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    return missing, changed


def main() -> int:
    if not MANIFEST.is_file():
        print("ERROR: deploy-manifest.sha256 is missing next to this script")
        return 1

    missing, changed = _mismatches()
    if missing or changed:
        print(
            f"ERROR: deployment does not match what was shipped "
            f"({len(missing)} missing, {len(changed)} changed)"
        )
        for name in (missing + changed)[:10]:
            print(f"  {name}")
        return 1

    sys.path.insert(0, str(ROOT))
    for module in CLOSURE:
        importlib.import_module(module)
    from ashare_lab.research.sequence_dataset import (  # noqa: F401
        ContainedMTSDatasetH,
        SegmentAdmission,
    )
    from ashare_lab.config import load_config

    load_config()
    print(
        f"deployment verified: {sum(1 for _ in _entries())} files, "
        f"{len(CLOSURE)} modules, revision {_revision()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
