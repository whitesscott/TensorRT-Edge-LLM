#!/usr/bin/env bash
# Update docs after pulling a new version folder (e.g. 0.6.0).
# Usage: ./update_docs.sh <VERSION>
# Example: ./update_docs.sh 0.6.0
#
# This script:
# 1. Removes the existing latest/ folder
# 2. Copies the new version (e.g. 0.6.0) to latest/, including _modules, _sources, _static, .html and related docs (no .nojekyll)
# 3. Updates every _static/switcher.json with all existing versions and "latest" pointing to the new content

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BASE_URL="https://nvidia.github.io/TensorRT-Edge-LLM"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <VERSION>" >&2
  echo "Example: $0 0.6.0" >&2
  exit 1
fi

VERSION="$1"

# Validate version folder exists
VERSION_DIR="$REPO_ROOT/$VERSION"
if [[ ! -d "$VERSION_DIR" ]]; then
  echo "Error: Version directory does not exist: $VERSION_DIR" >&2
  exit 1
fi

echo "Updating docs: making $VERSION the new latest..."

# (1) Remove existing latest folder
if [[ -d "$REPO_ROOT/latest" ]]; then
  echo "Removing existing latest/ ..."
  rm -rf "$REPO_ROOT/latest"
fi

# (2) Copy version to latest: _modules, _sources, _static, all .html and doc trees; do NOT copy .nojekyll
echo "Copying $VERSION/ to latest/ (excluding .nojekyll) ..."
mkdir -p "$REPO_ROOT/latest"
rsync -a --exclude='.nojekyll' "$VERSION_DIR/" "$REPO_ROOT/latest/"
# Ensure we never leave .nojekyll in latest (in case version dir had one)
rm -f "$REPO_ROOT/latest/.nojekyll"

# (3) Discover all version directories (semver-like: 0.4.0, 0.5.0, 0.6.0, ...)
VERSION_DIRS=()
for d in "$REPO_ROOT"/*; do
  if [[ -d "$d" ]]; then
    name=$(basename "$d")
    if [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      VERSION_DIRS+=("$name")
    fi
  fi
done
# Sort versions descending (newest first). Use -V if available (GNU), else numeric keys.
if sort -Vr <<<"0.1.0" &>/dev/null; then
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -Vr)); unset IFS
else
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -t. -k1,1nr -k2,2nr -k3,3nr)); unset IFS
fi

# Build switcher.json content: "latest" first (preferred), then numeric versions (newest first)
SWITCHER_ENTRIES="[
    {
        \"preferred\": true,
        \"version\": \"latest\",
        \"url\": \"${BASE_URL}/latest/\"
    }"
for v in "${sorted_versions[@]}"; do
  SWITCHER_ENTRIES+=",
    {
        \"version\": \"$v\",
        \"url\": \"${BASE_URL}/$v/\"
    }"
done
SWITCHER_ENTRIES+="
]"

# Write switcher.json to root _static and to every version's _static (including latest)
SWITCHER_PATHS=(
  "$REPO_ROOT/_static/switcher.json"
  "$REPO_ROOT/latest/_static/switcher.json"
)
for v in "${sorted_versions[@]}"; do
  p="$REPO_ROOT/$v/_static/switcher.json"
  if [[ -f "$p" || -d "$REPO_ROOT/$v/_static" ]]; then
    SWITCHER_PATHS+=("$p")
  fi
done

echo "Updating switcher.json in ${#SWITCHER_PATHS[@]} locations..."
for p in "${SWITCHER_PATHS[@]}"; do
  dir=$(dirname "$p")
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
  fi
  echo "$SWITCHER_ENTRIES" > "$p"
done

echo "Done. latest/ now reflects $VERSION; all switcher.json files list: latest, ${sorted_versions[*]}."
