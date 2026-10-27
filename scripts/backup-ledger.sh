#!/usr/bin/env bash
# Nightly ledger snapshot. Invoked by ashare-backup.service.
#
# The body lives in ashare_lab/paper/backup_job.py so it can be tested. What
# this file used to do, and why none of it survived:
#
#   - copied a hardcoded data/paper.db, which has been a stale 140K file
#     since July while the real ledger sat elsewhere
#   - used cp, which cannot capture a WAL-mode SQLite ledger
#   - wrote backups/paper_DATE.db, the pipeline's own filename, and ran five
#     hours later, so it replaced real snapshots with the stale copy
#
# It reported success throughout.
set -euo pipefail
# resolve symlinks before deriving the repo root: the systemd unit calls
# the real path, but a symlinked invocation must not cd into the link's parent
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
cd "$(dirname "$SCRIPT_PATH")/.."
exec .venv/bin/python -m ashare_lab.paper.backup_job "$@"
