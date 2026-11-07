#!/usr/bin/env bash
# Run a command; if it fails, show its last output lines as a run-page annotation
# (so failures are diagnosable without opening the full log).
# Usage: bash scripts/run_logged.sh "Step name" command args...
set -uo pipefail
title="$1"
shift
log="${RUNNER_TEMP:-/tmp}/step-$RANDOM.log"
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  detail="$(grep -v '^\s*$' "$log" | tail -n 12 | cut -c1-200 | sed 's/%/%25/g' | awk 'BEGIN{ORS="%0A"} {print}')"
  echo "::error title=$title failed::$detail"
fi
exit "$status"
