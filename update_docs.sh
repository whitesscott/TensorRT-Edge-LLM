#!/usr/bin/env bash
# Update docs with a new version release.
# Usage: ./update_docs.sh <VERSION>
# Example: ./update_docs.sh 0.6.1
#
# Steps:
# 1. Flatten the version folder: if <VERSION>/html/ exists, replace <VERSION>/
#    with its contents so <VERSION>/index.html is at the root and
#    https://.../TensorRT-Edge-LLM/<VERSION>/ always resolves correctly.
# 2. Update latest/ to match the new version content (no .nojekyll).
# 3. Update root-level doc files/folders to the new version content.
#    Protected (never touched): .git  .nojekyll  README.md  update_docs.sh
#                               latest/  and all x.y.z/ version folders.
# 4. Update every _static/switcher.json to list all versions (newest first).

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
VERSION_DIR="$REPO_ROOT/$VERSION"

if [[ ! -d "$VERSION_DIR" ]]; then
  echo "Error: version folder not found: $VERSION_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 – Flatten nested html/ layout
#   Some builds place the actual HTML under <VERSION>/html/.  We replace the
#   version folder with that content so <VERSION>/index.html sits at the top.
# ---------------------------------------------------------------------------
if [[ -d "$VERSION_DIR/html" ]]; then
  echo "[$VERSION] Nested html/ detected — flattening $VERSION/ ..."
  TMP_DIR="$(mktemp -d)"
  cp -r "$VERSION_DIR/html/." "$TMP_DIR/"
  rm -rf "$VERSION_DIR"
  mv "$TMP_DIR" "$VERSION_DIR"
fi

SOURCE="$VERSION_DIR"
echo "Releasing $VERSION as latest (source: $SOURCE) ..."

# ---------------------------------------------------------------------------
# Step 2 – Rebuild latest/
# ---------------------------------------------------------------------------
echo "Updating latest/ ..."
rm -rf "$REPO_ROOT/latest"
mkdir -p "$REPO_ROOT/latest"
cp -r "$SOURCE/." "$REPO_ROOT/latest/"
rm -f "$REPO_ROOT/latest/.nojekyll"

# ---------------------------------------------------------------------------
# Step 3 – Update root-level doc content
#   Remove every non-protected item first, then copy the new version in.
# ---------------------------------------------------------------------------
echo "Updating root-level doc content ..."

# Remove stale doc directories (skip version dirs, latest/, and .git)
for item in "$REPO_ROOT"/*/; do
  name=$(basename "$item")
  [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && continue
  [[ "$name" == "latest" ]]                  && continue
  [[ "$name" == ".git" ]]                    && continue
  rm -rf "$REPO_ROOT/$name"
done

# Remove stale doc files (skip protected files; hidden files like .nojekyll
# are not matched by * so they are naturally safe)
for f in "$REPO_ROOT"/*; do
  [[ -d "$f" ]] && continue
  name=$(basename "$f")
  case "$name" in
    README.md|update_docs.sh) continue ;;
  esac
  rm -f "$f"
done

# Copy new version content into root
cp -r "$SOURCE/." "$REPO_ROOT/"

# ---------------------------------------------------------------------------
# Step 4 – Rebuild switcher.json everywhere
# ---------------------------------------------------------------------------

# Collect all x.y.z version dirs, sorted newest-first
VERSION_DIRS=()
for d in "$REPO_ROOT"/*/; do
  name=$(basename "$d")
  [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && VERSION_DIRS+=("$name")
done

if sort -Vr <<<"0.1.0" &>/dev/null 2>&1; then
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -Vr)); unset IFS
else
  IFS=$'\n' sorted_versions=($(printf '%s\n' "${VERSION_DIRS[@]}" | sort -t. -k1,1nr -k2,2nr -k3,3nr)); unset IFS
fi

# Build JSON
SWITCHER_JSON="[
    {
        \"preferred\": true,
        \"version\": \"latest\",
        \"url\": \"${BASE_URL}/latest/\"
    }"
for v in "${sorted_versions[@]}"; do
  SWITCHER_JSON+=",
    {
        \"version\": \"$v\",
        \"url\": \"${BASE_URL}/$v/\"
    }"
done
SWITCHER_JSON+="
]"

# Write to: root _static/, latest/_static/, and every <version>/_static/
SWITCHER_PATHS=(
  "$REPO_ROOT/_static/switcher.json"
  "$REPO_ROOT/latest/_static/switcher.json"
)
for v in "${sorted_versions[@]}"; do
  [[ -d "$REPO_ROOT/$v/_static" ]] && SWITCHER_PATHS+=("$REPO_ROOT/$v/_static/switcher.json")
done

echo "Updating ${#SWITCHER_PATHS[@]} switcher.json files ..."
for p in "${SWITCHER_PATHS[@]}"; do
  mkdir -p "$(dirname "$p")"
  printf '%s\n' "$SWITCHER_JSON" > "$p"
done

# ---------------------------------------------------------------------------
echo ""
echo "Done."
echo "  $VERSION/   index.html at root (flattened if needed)"
echo "  latest/     $VERSION content"
echo "  repo root   $VERSION content"
echo "  switcher    latest  ${sorted_versions[*]}"
