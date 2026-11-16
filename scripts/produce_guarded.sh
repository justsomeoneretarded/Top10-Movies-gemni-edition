#!/usr/bin/env bash
# The resource guard contains this script, Ollama and the entire production tree.
set -euo pipefail
diagnostics="${RUNNER_DIAGNOSTICS_DIR:-work/runner-diagnostics}"
mkdir -p "$diagnostics"
export OLLAMA_LOG_FILE="$diagnostics/ollama.log"
export OLLAMA_PID_FILE="${RUNNER_TEMP:-/tmp}/movie-production-ollama-$$.pid"

cleanup_ollama() {
  python3 - "$OLLAMA_PID_FILE" <<'PY'
import os
import signal
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
try:
    if path.is_symlink():
        raise ValueError("Unsafe PID file")
    pid = int(path.read_text().strip())
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    # Never signal a different process whose PID happened to be reused.
    if pid <= 1 or b"serve" not in args:
        raise ValueError("Not the production Ollama server")
    if not any(Path(os.fsdecode(arg)).name == "ollama" for arg in args if arg):
        raise ValueError("Not the production Ollama server")
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(.1)
    else:
        # Recheck identity before escalation.
        if Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") == args:
            os.kill(pid, signal.SIGKILL)
except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
    pass
finally:
    if not path.is_symlink():
        path.unlink(missing_ok=True)
PY
}
trap cleanup_ollama EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The server must start inside the same resource boundary as its clients.
bash scripts/bootstrap_ollama.sh
bash scripts/run_logged.sh "Production" python -m movie_pipeline.cli
