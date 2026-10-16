#!/usr/bin/env bash
set -euo pipefail

LOG_DIRECTORY="${ASSIGNMENT2_LOG_DIRECTORY:-${LOG_DIRECTORY:-}}"
if [[ -z "$LOG_DIRECTORY" ]]; then
  echo "Set ASSIGNMENT2_LOG_DIRECTORY to the persistent vLLM log directory." >&2
  exit 1
fi
PID_FILE="$LOG_DIRECTORY/vllm.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE; nothing was stopped."
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Sent a stop signal to vLLM PID $PID."
else
  echo "PID $PID is not running."
fi
rm -f "$PID_FILE"
