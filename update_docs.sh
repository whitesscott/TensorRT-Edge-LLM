#!/usr/bin/env bash
# Update docs after pulling a new version folder (e.g. 0.6.1).
# Usage: ./update_docs.sh <VERSION>
# Example: ./update_docs.sh 0.6.1
#
# This script:
# 1. Resolves the doc source root inside the version folder
#    (uses <VERSION>/html/ if present, otherwise <VERSION>/ directly)
# 2. Syncs the doc source root -> latest/  (excluding .nojekyll)
# 3. Syncs the doc source root -> repo root (cpp_api/, developer_guide/,
#    user_guide/, _modules/, _sources/, _static/, *.html, etc.)
#    Preserves: .git, .nojekyll, README.md, update_docs.sh, latest/,
#    and all versioned release folders (e.g. 0.4.0/, 0.6.1/).
# 4. Updates every _static/switcher.json with all existing versions,
#    "latest" pointing to the new content.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BASE_URL="https://nvidia.github.io/TensorRT-Edge-LLM"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <VERSION>" >&2
  echo "Example: $0 0.6.1" >&2
  exit 1
fi

VERSION="$1"

# Validate version folder exists
VERSION_DIR="$REPO_ROOT/$VERSION"
if [[ ! -d "$VERSION_DIR" ]]; then
  echo "Error: Version directory does not exist: $VERSION_DIR" >&2
  exit 1
fi

# Resolve the actual doc root: some releases nest everything under html/
if [[ -d "$VERSION_DIR/html" ]]; then
  SOURCE_ROOT="$VERSION_DIR/html"
  echo "Detected nested html/ layout: using $VERSION/html/ as source root."
else
  SOURCE_ROOT="$VERSION_DIR"
fi

echo "Updating docs: making $VERSION the new latest..."

# (1) Remove existing latest folder and repopulate from SOURCE_ROOT
echo "Rebuilding latest/ from $SOURCE_ROOT ..."
rm -rf "$REPO_ROOT/latest"
mkdir -p "$REPO_ROOT/latest"
rsync -a --exclude='.nojekyll' "$SOURCE_ROOT/" "$REPO_ROOT/latest/"
rm -f "$REPO_ROOT/latest/.nojekyll"

# (2) Sync doc content to the repo root so that cpp_api/, developer_guide/,
#     user_guide/, _modules/, _sources/, _static/, *.html, etc. reflect the
#     new version.  We use --delete so stale files are removed, but we
#     carefully exclude everything that must not be touched.
echo "Syncing $SOURCE_ROOT -> repo root (updating cpp_api/, developer_guide/, _static/, HTML files, etc.) ..."
rsync -a --delete \
  --exclude='.git' \
  --exclude='.git/' \
  --exclude='.nojekyll' \
  --exclude='README.md' \
  --exclude='update_docs.sh' \
  --exclude='latest' \
  --exclude='latest/' \
  --exclude='[0-9]*.[0-9]*.[0-9]*' \
  --exclude='[0-9]*.[0-9]*.[0-9]*/' \
  "$SOURCE_ROOT/" "$REPO_ROOT/"

# (3) Discover all version directories (semver-like: 0.4.0, 0.5.0, 0.6.1, ...)
VERSION_DIRS=()
for d in "$REPO_ROOT"/*; do
  if [[ -d "$d" ]]; then
    name=$(basename "$d")
    if [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      VERSION_DIRS+=("$name")
    fi
  fi
done
# Sort versions descending (newest first). Use -V if available (GNU sort), else numeric keys.
if sort -Vr <<<"0.1.0" &>/dev/null; then
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -Vr)); unset IFS
else
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -t. -k1,1nr -k2,2nr -k3,3nr)); unset IFS
fi

# (4) Build switcher.json: "latest" first (preferred), then numeric versions newest-first
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

# Helper: find the _static dir inside a version folder (may be nested under html/)
version_static_dir() {
  local v="$1"
  if [[ -d "$REPO_ROOT/$v/html/_static" ]]; then
    echo "$REPO_ROOT/$v/html/_static"
  elif [[ -d "$REPO_ROOT/$v/_static" ]]; then
    echo "$REPO_ROOT/$v/_static"
  fi
}

# Collect all switcher.json paths: root, latest, and every version
SWITCHER_PATHS=(
  "$REPO_ROOT/_static/switcher.json"
  "$REPO_ROOT/latest/_static/switcher.json"
)
for v in "${sorted_versions[@]}"; do
  sdir="$(version_static_dir "$v")"
  if [[ -n "$sdir" ]]; then
    SWITCHER_PATHS+=("$sdir/switcher.json")
  fi
done

echo "Updating switcher.json in ${#SWITCHER_PATHS[@]} locations..."
for p in "${SWITCHER_PATHS[@]}"; do
  dir=$(dirname "$p")
  mkdir -p "$dir"
  echo "$SWITCHER_ENTRIES" > "$p"
done

echo ""
echo "Done."
echo "  latest/    -> $VERSION content"
echo "  repo root  -> $VERSION content (cpp_api/, developer_guide/, _static/, HTML files, ...)"
echo "  switcher   -> latest, ${sorted_versions[*]}"
