#!/usr/bin/env bash
set -euo pipefail

# Run only inside a confirmed Slurm GPU allocation, never on the gateway node.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No Slurm job is active; refusing to start vLLM outside an allocation." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start a GPU model server." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_PATH="${ASSIGNMENT2_QWEN_SNAPSHOT:-$MODEL_ID}"
PORT="${PORT:-8080}"
LOG_DIRECTORY="${ASSIGNMENT2_LOG_DIRECTORY:?Set ASSIGNMENT2_LOG_DIRECTORY to verified non-home storage.}"
MODEL_CACHE="${ASSIGNMENT2_MODEL_CACHE:?Set ASSIGNMENT2_MODEL_CACHE to verified non-home storage.}"
SERVE_PYTHON="${ASSIGNMENT2_SERVE_PYTHON:?Set ASSIGNMENT2_SERVE_PYTHON to the serving environment Python executable.}"

case "$MODEL_CACHE" in
  "$HOME"|"$HOME"/*)
    echo "Refusing to use /home for model cache storage: $MODEL_CACHE" >&2
    exit 1
    ;;
esac

if [[ ! -x "$SERVE_PYTHON" ]]; then
  echo "Serving Python is not executable: $SERVE_PYTHON" >&2
  exit 1
fi

# vLLM JIT compilation invokes executables such as ninja from this environment.
export PATH="$(dirname "$SERVE_PYTHON"):$PATH"
export HF_HOME="$MODEL_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLLM_CACHE_ROOT="$MODEL_CACHE/vllm"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MODEL_CACHE/xdg-cache}"
export TORCH_HOME="$MODEL_CACHE/torch"
export TRITON_CACHE_DIR="$MODEL_CACHE/triton"
mkdir -p "$LOG_DIRECTORY"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

if [[ -f "$LOG_DIRECTORY/vllm.pid" ]] && kill -0 "$(cat "$LOG_DIRECTORY/vllm.pid")" 2>/dev/null; then
  echo "A recorded vLLM process is already running: $(cat "$LOG_DIRECTORY/vllm.pid")" >&2
  exit 1
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$SERVE_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4352 \
  >"$LOG_DIRECTORY/vllm.log" 2>&1 &
echo $! >"$LOG_DIRECTORY/vllm.pid"
echo "Started vLLM PID $(cat "$LOG_DIRECTORY/vllm.pid"). Log: $LOG_DIRECTORY/vllm.log"
echo "Model cache: $MODEL_CACHE"
echo "Check readiness before evaluating: curl -fsS http://127.0.0.1:$PORT/v1/models"
