#!/usr/bin/env bash
set -euo pipefail

EVAL_PYTHON="${1:?Usage: check_python_environments.sh EVAL_PYTHON [SERVE_PYTHON]}"
SERVE_PYTHON="${2:-}"

if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "Evaluation Python must be executable." >&2
  exit 1
fi

echo "Checking evaluation environment: $EVAL_PYTHON"
"$EVAL_PYTHON" -c 'import balrog.client; import omegaconf; from balrog.environments.babyai_text.babyai_env import make_babyai_env; from balrog.environments.crafter.crafter_env import make_crafter_env; print("BALROG evaluation imports OK")'

if [[ -z "$SERVE_PYTHON" ]]; then
  exit 0
fi
if [[ ! -x "$SERVE_PYTHON" ]]; then
  echo "The selected local model requires an executable serving Python." >&2
  exit 1
fi

echo "Checking serving environment: $SERVE_PYTHON"
"$SERVE_PYTHON" -c 'import vllm; print("vLLM import OK")'

if [[ "$EVAL_PYTHON" == "$SERVE_PYTHON" ]]; then
  echo "Checking combined BALROG and vLLM environment."
  "$EVAL_PYTHON" -c 'import balrog; import vllm; print("Combined environment imports OK")'
else
  echo "Separate evaluation and serving environments are configured."
fi
