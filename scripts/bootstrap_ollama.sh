#!/usr/bin/env bash
set -euo pipefail
# Ollama is a local inference server. No account or paid API is used.
case "$#" in
  0) mode=script ;;
  1)
    if [ "$1" = --serve-only ]; then
      mode=serve-only
    else
      echo "Usage: $0 [--serve-only]" >&2
      exit 2
    fi
    ;;
  *)
    echo "Usage: $0 [--serve-only]" >&2
    exit 2
    ;;
esac

runner_temp="${RUNNER_TEMP:-/tmp}"
model="${SCRIPT_MODEL:-gemma4:12b}"
url="${OLLAMA_URL:-http://127.0.0.1:11434}"
if ! binary_dir="$(python3 -m scripts.install_ollama --existing)"; then
  python3 -m scripts.runner_disk --stage ollama-install
  binary_dir="$(python3 -m scripts.install_ollama)"
fi
export PATH="$binary_dir:$PATH"

server_ready() {
  curl --fail --silent --connect-timeout 1 --max-time 2 "$url/api/tags" >/dev/null 2>&1
}

# Reusing a healthy server avoids a second process fighting for the same port.
if ! server_ready; then
  ollama serve > "$runner_temp/ollama.log" 2>&1 < /dev/null &
  server_pid=$!
  for attempt in $(seq 1 60); do
    if server_ready; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if ! server_ready; then
    echo "Ollama did not become ready at $url" >&2
    tail -n 40 "$runner_temp/ollama.log" >&2 || true
    exit 1
  fi
fi

if [ "$mode" = serve-only ]; then
  python3 -m scripts.runner_disk --stage ready
  exit 0
fi

# An existing model needs only the runtime reserve. Missing models need the
# larger pull reserve without deleting restored voice/Python/model caches.
if timeout --kill-after=10s 30s ollama show "$model" >/dev/null 2>&1; then
  python3 -m scripts.runner_disk --stage ready
  exit 0
fi
python3 -m scripts.runner_disk --stage script-model

pull_status=1
for attempt in 1 2 3; do
  if timeout --kill-after=15s 1200s ollama pull "$model"; then
    python3 -m scripts.runner_disk --stage ready
    exit 0
  else
    pull_status=$?
  fi
  echo "Ollama model pull attempt $attempt/3 failed (exit $pull_status)" >&2
  if [ "$attempt" -lt 3 ]; then sleep "$((attempt * 5))"; fi
done
echo "Ollama could not prepare model $model after three bounded attempts" >&2
exit "$pull_status"
