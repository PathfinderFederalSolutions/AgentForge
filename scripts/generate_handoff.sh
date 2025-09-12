#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

# Run system inventory to produce latest snapshot
./generate_system_inventory.sh >/dev/null 2>&1 || true

# Determine latest inventory markdown
latest_md="$(ls -1t system_inventory_*.md 2>/dev/null | head -n1 || true)"
if [[ -z "${latest_md}" ]]; then
  echo "No system_inventory_*.md found. Exiting." >&2
  exit 1
fi

# Extract TS from filename: system_inventory_YYYYmmddTHHMMSSZ.md
TS="$(basename "$latest_md" | sed -E 's/system_inventory_([0-9TZ]+)\.md/\1/')"
HANDOFF="handoff_compiled_${TS}.md"

# Curated context preamble (optional)
PREAMBLE="${root_dir}/gpt5_context_preamble.md"

{
  echo "# Handoff Bundle"
  echo ""
  echo "## 1. Curated Context"
  if [[ -f "${PREAMBLE}" ]]; then
    cat "${PREAMBLE}"
  else
    echo "(No curated preamble file found)"
  fi
  echo ""
  echo ""
  echo "## 2. System Inventory"
  echo ""
  cat "$latest_md"
  echo ""
  echo "## 3. Repository Structure (Depth 4)"
  echo '```'
  # Generate a simple file listing up to depth 4
  find . -maxdepth 4 -type f | sed 's|^\./||' | sort
  echo '```'
} > "$HANDOFF"

echo "$HANDOFF"
