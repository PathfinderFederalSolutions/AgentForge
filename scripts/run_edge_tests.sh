#!/usr/bin/env bash
# Run edge-mode tests locally by spinning up a NATS JetStream container.
# Requires: docker, pytest, python deps installed.
# Usage: ./scripts/run_edge_tests.sh
# Env:
#   NAME        - container name (default: nats-js-test)
#   IMAGE       - NATS image (default: nats:2.10)
#   PORT        - NATS client port (default: 4222)
#   HTTP_PORT   - NATS monitoring port (default: 8222)
#   KEEP_NATS   - if set to 1, do not cleanup container on exit
#   RUN_DRAIN   - if set to 1, run local_drain_test.py before cleanup in same shell
set -euo pipefail

NAME=${NAME:-nats-js-test}
IMAGE=${IMAGE:-nats:2.10}
PORT=${PORT:-4222}
HTTP_PORT=${HTTP_PORT:-8222}

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}

# Only cleanup at script exit if KEEP_NATS is not requested
if [[ "${KEEP_NATS:-0}" != "1" ]]; then
  trap cleanup EXIT
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2; exit 1; fi

# Remove any stale container with the same name before starting fresh
cleanup || true

echo "Starting NATS ($IMAGE) with JetStream..."
docker run -d --name "$NAME" -p ${PORT}:4222 -p ${HTTP_PORT}:8222 "$IMAGE" -js >/dev/null

export NATS_URL="nats://localhost:${PORT}"
export EDGE_MODE=1

# Wait for NATS to accept connections and JetStream to be ready
for i in {1..20}; do
  if nc -z localhost "${PORT}" >/dev/null 2>&1; then
    # Try JSZ endpoint; ignore failures if curl not present
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "http://localhost:${HTTP_PORT}/jsz?timeout=1s" >/dev/null 2>&1; then
        break
      fi
    else
      break
    fi
  fi
  sleep 0.3
  if [[ $i -eq 20 ]]; then
    echo "Warning: NATS may not be fully ready yet; proceeding..." >&2
  fi
done

set +e
pytest -q tests/test_edge_store_forward.py tests/test_edge_disconnect_reconnect.py
code=$?
set -e

if [[ $code -ne 0 ]]; then
  echo "Edge tests failed with code $code" >&2
  exit $code
fi

echo "Edge tests passed."

# Optionally run the local drain test within the same shell so the NATS container remains alive
if [[ "${RUN_DRAIN:-0}" == "1" ]]; then
  echo "Running local drain test..."
  # Use project-local python if available
  if command -v python >/dev/null 2>&1; then
    python local_drain_test.py || true
  else
    echo "python not found; skipping local_drain_test.py" >&2
  fi
fi
