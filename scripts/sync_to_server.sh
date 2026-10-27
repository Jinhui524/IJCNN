#!/usr/bin/env bash
# Sync this repository to the AutoDL training machine, code only.
#
#   bash scripts/sync_to_server.sh root@HOST:/root/autodl-tmp/IJCNN --port 42243
#   bash scripts/sync_to_server.sh root@HOST:/root/autodl-tmp/IJCNN --port 42243 --dry-run
#
# Environment:
#   SYNC_FROM   local repository root (default: this script's parent of parent)
#
# What is deliberately NOT sent:
#   *.venv-*        built for macOS, useless and harmful on the Linux box
#   dataset/        large CSVs that already exist on the server
#   results/ work/  completed experiments and scratch output
#   upstream/       pinned git checkouts; sources.lock.json verifies their hashes
#   original/       immutable source snapshots used for provenance
#   protocols/      split manifests, regenerated identically from the same data
#   __pycache__/    stale bytecode from a different interpreter
set -euo pipefail

TARGET=""
PORT="${SYNC_PORT:-22}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    *) TARGET="$1"; shift ;;
  esac
done
if [[ -z "$TARGET" || "$TARGET" != *:* ]]; then
  printf '%s\n' "Usage: bash scripts/sync_to_server.sh USER@HOST:/remote/path --port 42243 [--dry-run]" >&2
  exit 2
fi
FROM="${SYNC_FROM:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

EXCLUDES=(
  --exclude '.venv-torch/'
  --exclude '.venv-tensorflow/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.idea/'
  --exclude '.DS_Store'
  --exclude 'dataset/'
  --exclude 'baseline/results/'
  --exclude 'baseline/protocols/'
  --exclude 'baseline/upstream/'
  --exclude 'baseline/original/'
  --exclude 'baseline/work/'
  --exclude 'work/'
  --exclude '*.pptx'
  --exclude '~$*.pptx'
  --exclude '*.log'
)

SSH_CMD="ssh -p ${PORT} -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=10m"

printf 'From : %s\nTo   : %s (port %s)\n\n' "$FROM" "$TARGET" "$PORT"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' 'DRY RUN: nothing is transferred'
  rsync -avzn --itemize-changes -e "$SSH_CMD" "${EXCLUDES[@]}" "$FROM/" "$TARGET/"
else
  rsync -avz --progress -e "$SSH_CMD" "${EXCLUDES[@]}" "$FROM/" "$TARGET/"
  printf '\nSynced. Next on the server:\n'
  printf '  cd %s\n' "${TARGET#*:}"
  printf '  %s\n' './baseline/.venv-torch/bin/python -m baseline.sources verify'
fi
