#!/usr/bin/env bash
# Stage the code tree that the GPU host runs, then describe it.
#
# Usage: stage_gpu_code.sh <repo-root> <staging-dir>
#
# The staging directory is the whole contract with the target: the package, the
# configs it reads, the verifier that runs there, and the manifest that says
# what was shipped. Both the scheduled pipeline and the manual deploy use this
# script so the two paths cannot drift apart.

set -uo pipefail

REPO=${1:?usage: stage_gpu_code.sh <repo-root> <staging-dir>}
STAGING=${2:?usage: stage_gpu_code.sh <repo-root> <staging-dir>}

repo_real=$(cd "$REPO" 2>/dev/null && pwd -P) || {
    echo "ERROR: $REPO is not a directory" >&2
    exit 1
}
repo_clean=${repo_real%/}
staging_real=$(realpath -m -- "$STAGING")
staging_clean=${staging_real%/}
[ -n "$staging_clean" ] || staging_clean=/

case "$staging_clean" in
    "/" | "$repo_clean" | "$repo_clean"/*)
        echo "ERROR: refusing to stage into $staging_clean (it holds the repository)" >&2
        exit 1
        ;;
esac
case "$repo_clean" in
    "$staging_clean"/*)
        echo "ERROR: refusing to clear $staging_clean (it contains the repository)" >&2
        exit 1
        ;;
esac

if [ -L "${STAGING%/}" ]; then
    echo "ERROR: $STAGING is a symbolic link; pass the directory it points at" >&2
    exit 1
fi

# Everything below acts on the normalized path: the guards above judged that
# value, and `rm -rf link/` follows a trailing slash into the link target.
STAGING=$staging_clean

rm -rf -- "$STAGING"
mkdir -p -- "$STAGING/ashare_lab"

# The package, not a file list: a list is a second source of truth nobody
# updates when an import is added. The trading engine and local databases stay
# on the source host.
if ! rsync -a --delete \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude 'paper/' --exclude '*.db' \
    "$REPO/ashare_lab/" "$STAGING/ashare_lab/"; then
    echo "ERROR: cannot stage ashare_lab from $REPO" >&2
    rm -rf "$STAGING"
    exit 1
fi

if ! rsync -a --delete --exclude '__pycache__/' "$REPO/configs/" "$STAGING/configs/"; then
    echo "ERROR: cannot stage configs from $REPO" >&2
    rm -rf "$STAGING"
    exit 1
fi

if ! cp "$REPO/scripts/deploy_verify.py" "$STAGING/"; then
    echo "ERROR: cannot stage deploy_verify.py" >&2
    rm -rf "$STAGING"
    exit 1
fi

# The manifest is what turns the remote check into a freshness check: scp
# merges into the remote tree, so a stale-but-present module would otherwise
# pass. Names are NUL-separated end to end so spaces survive.
# The bytes come from the working tree, not from the commit, so a dirty tree
# must not be reported under a clean commit id.
revision=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$revision" != unknown ] && ! git -C "$REPO" diff --quiet -- ashare_lab configs 2>/dev/null; then
    revision="$revision-dirty"
fi
if ! { printf '# revision %s\n' "$revision"
       ( cd "$STAGING" && find . -type f ! -name deploy-manifest.sha256 -print0 \
        | sed -z 's|^\./||' | LC_ALL=C sort -z | xargs -0 sha256sum ); } \
        > "$STAGING/deploy-manifest.sha256"; then
    echo "ERROR: cannot build the deployment manifest" >&2
    rm -rf "$STAGING"
    exit 1
fi

exit 0
