#!/usr/bin/env bash
set -euo pipefail

# Validate kustomize build for each deployment profile.
# Uses `kustomize` if available, else `kubectl kustomize`.

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
profiles=(scif govcloud saas)

build_cmd=""
if command -v kustomize >/dev/null 2>&1; then
  build_cmd="kustomize build"
elif command -v kubectl >/dev/null 2>&1; then
  build_cmd="kubectl kustomize"
else
  echo "Neither kustomize nor kubectl was found in PATH. Install one to validate profiles." >&2
  exit 2
fi

rc=0
for p in "${profiles[@]}"; do
  dir="$root_dir/k8s/profiles/$p"
  echo "==> Validating $p ($dir)"
  if ! out=$(bash -lc "$build_cmd '$dir'" 2>&1); then
    echo "Build FAILED for profile: $p" >&2
    echo "$out" >&2
    rc=1
  else
    echo "Build OK ($p). Manifests: $(echo "$out" | awk '/^kind: /{print $2}' | sort | uniq -c | tr '\n' ' ')"
  fi
  echo
done
exit $rc
