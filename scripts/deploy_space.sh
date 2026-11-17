#!/usr/bin/env bash
# Deploy the committed code to a Hugging Face Space, which builds the Dockerfile.
#
# Usage: scripts/deploy_space.sh <username>/<space-name>
# Needs `hf auth login` with a token that has write access.
set -euo pipefail

SPACE="${1:?Usage: scripts/deploy_space.sh <username>/<space-name>}"

# Only committed files are deployed, so local experiments never go live
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Commit your changes first; only committed files are deployed." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git archive HEAD | tar -x -C "$STAGE"

# The Space reads its settings from the front matter of its own README
cp deploy/space_README.md "$STAGE/README.md"

hf repos create "$SPACE" --type space --sdk docker --public --exist-ok
hf upload "$SPACE" "$STAGE" . --type space \
  --commit-message "Deploy $(git rev-parse --short HEAD)"

echo "Deployed. The Space builds for a few minutes at https://huggingface.co/spaces/$SPACE"
