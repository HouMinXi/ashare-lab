#!/usr/bin/env bash
# Lint gate: run before committing anything that touches Python.
#
# The rules live in pyproject.toml so every editor and CI entry point agrees
# with this script. RUF100 is part of the set: a stale `# noqa` claims a rule
# is being suppressed when it is not, which is how a suppression rots into a
# comment nobody can trust.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

if [ -x .venv/bin/ruff ]; then
    RUFF=.venv/bin/ruff
elif command -v ruff >/dev/null 2>&1; then
    RUFF=$(command -v ruff)
else
    echo "lint: ruff not found (expected .venv/bin/ruff or ruff on PATH)" >&2
    exit 2
fi

"$RUFF" check ashare_lab tests/ scripts/
