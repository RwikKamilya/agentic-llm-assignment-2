#!/usr/bin/env bash
set -euo pipefail

LOG_DIRECTORY="${LOG_DIRECTORY:-assignment2/server_logs}"
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
