#!/usr/bin/env bash
set -euo pipefail

EVAL_PYTHON="${1:?Usage: check_python_environments.sh EVAL_PYTHON [SERVE_PYTHON]}"
SERVE_PYTHON="${2:-$EVAL_PYTHON}"

if [[ ! -x "$EVAL_PYTHON" || ! -x "$SERVE_PYTHON" ]]; then
  echo "Both arguments must be executable Python paths." >&2
  exit 1
fi

echo "Checking evaluation environment: $EVAL_PYTHON"
"$EVAL_PYTHON" -c 'import balrog; import openai; print("BALROG evaluation imports OK")'

echo "Checking serving environment: $SERVE_PYTHON"
"$SERVE_PYTHON" -c 'import vllm; print("vLLM import OK")'

if [[ "$EVAL_PYTHON" == "$SERVE_PYTHON" ]]; then
  echo "Checking combined BALROG and vLLM environment."
  "$EVAL_PYTHON" -c 'import balrog; import vllm; print("Combined environment imports OK")'
else
  echo "Separate evaluation and serving environments are configured."
fi
