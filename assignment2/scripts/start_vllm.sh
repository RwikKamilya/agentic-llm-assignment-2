#!/usr/bin/env bash
set -euo pipefail

# Run only inside a confirmed interactive or scheduled GPU allocation.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start a GPU model server." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8080}"
LOG_DIRECTORY="${LOG_DIRECTORY:-assignment2/server_logs}"
mkdir -p "$LOG_DIRECTORY"

if [[ -f "$LOG_DIRECTORY/vllm.pid" ]] && kill -0 "$(cat "$LOG_DIRECTORY/vllm.pid")" 2>/dev/null; then
  echo "A recorded vLLM process is already running: $(cat "$LOG_DIRECTORY/vllm.pid")" >&2
  exit 1
fi

nohup vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  >"$LOG_DIRECTORY/vllm.log" 2>&1 &
echo $! >"$LOG_DIRECTORY/vllm.pid"
echo "Started vLLM PID $(cat "$LOG_DIRECTORY/vllm.pid"). Log: $LOG_DIRECTORY/vllm.log"
echo "Check readiness before evaluating: curl -fsS http://127.0.0.1:$PORT/v1/models"
