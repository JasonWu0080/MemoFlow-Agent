#!/usr/bin/env bash
# Create a fresh, history-free repository containing only this package.
#
#   scripts/export_public_repo.sh /path/to/new/memoflow-agent
#
# The MemoFlow monorepo history must NOT be published (it contains business
# documents and infrastructure notes). This copies the audited package files
# into a new directory and makes one initial commit there. It never pushes.
set -euo pipefail

target="${1:?usage: export_public_repo.sh <empty target directory>}"
source_dir="$(cd "$(dirname "$0")/.." && pwd)"

if [ -e "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
  echo "refusing: $target exists and is not empty" >&2
  exit 1
fi
mkdir -p "$target"

# Tracked package files only (no caches, venvs, builds).
(cd "$source_dir" && git ls-files --cached --others --exclude-standard .) | while read -r file; do
  case "$file" in
    *__pycache__*|*.pyc|dist/*|build/*|*.egg-info/*) continue ;;
  esac
  mkdir -p "$target/$(dirname "$file")"
  cp "$source_dir/$file" "$target/$file"
done

python3 "$target/scripts/scan_secrets.py"
if grep -q "TO BE CONFIRMED" "$target/NOTICE"; then
  echo "NOTE: NOTICE still has the copyright placeholder — resolve before publishing." >&2
fi

cd "$target"
git init -q -b main
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-MemoFlow Agent}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@example.invalid}" \
  commit -q -m "Initial import of memoflow-agent v0.1.0"
echo "Exported to $target (1 commit, not pushed)."
