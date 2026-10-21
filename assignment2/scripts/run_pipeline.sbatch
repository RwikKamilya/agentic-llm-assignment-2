#!/usr/bin/env bash
# Evaluation runs entirely on one allocated L40S compute node, never calcium.
# The user supplies --chdir, --output, and --error to sbatch.
#SBATCH --job-name=assignment2-run
#SBATCH --partition=L40s_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=12:00:00

set -euo pipefail
SECONDS=0
trap 'exit_code=$?; echo "ASSIGNMENT2_JOB_EXIT exit_code=$exit_code"' EXIT

ASS2_ROOT="${ASS2_ROOT:?Set ASS2_ROOT=/data/s4822285/assignment2 before submission.}"
if [[ "$ASS2_ROOT" != "/data/s4822285/assignment2" ]]; then
  echo "ASS2_ROOT must be /data/s4822285/assignment2 for this deployment." >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This entry point must run inside a Slurm allocation, never calcium." >&2
  exit 1
fi

REPO_ROOT="$ASS2_ROOT/repo"
EVAL_PYTHON="$ASS2_ROOT/envs/balrog-prebuilt/bin/python"
SERVE_PYTHON="$ASS2_ROOT/envs/vllm-prebuilt/bin/python"
CACHE_ROOT="$ASS2_ROOT/cache"
MODEL_CACHE="$CACHE_ROOT/models"
PACKAGE_CACHE="$CACHE_ROOT/pip"
RESULTS_ROOT="$ASS2_ROOT/results"
LOG_DIRECTORY="$ASS2_ROOT/logs/vllm-$SLURM_JOB_ID"
PHASE="${ASS2_PHASE:?Set ASS2_PHASE to smoke or final.}"
MODELS="${ASS2_MODELS:?Set ASS2_MODELS to one or more configured profiles.}"
MIN_FREE_GB="${ASS2_MIN_FREE_GB:?Set a quota-aware minimum free-space threshold in GiB.}"
CLEANUP_RESERVE_MINUTES="${ASS2_CLEANUP_RESERVE_MINUTES:-30}"
RUN_DIRECTORY="${ASS2_RUN_DIRECTORY:-$RESULTS_ROOT/${PHASE}_slurm-$SLURM_JOB_ID}"
RESUME_DIRECTORY="${ASS2_RESUME_DIRECTORY:-}"
JOB_TIME_LIMIT_SECONDS=$((12 * 60 * 60))
TERMINATION_GRACE_SECONDS=120

if [[ ! -d "$REPO_ROOT" || ! -f "$REPO_ROOT/assignment2/config/experiment.yaml" ]]; then
  echo "Expected repository is missing or incomplete: $REPO_ROOT" >&2
  exit 1
fi
if [[ "$PHASE" != "smoke" && "$PHASE" != "final" ]]; then
  echo "ASS2_PHASE must be smoke or final." >&2
  exit 1
fi
read -r -a MODEL_ARGS <<< "$MODELS"
MODELS="${MODEL_ARGS[*]}"
if (( ${#MODEL_ARGS[@]} != 1 )); then
  echo "Launch one model per job so cloud-budget failure cannot prevent Qwen evaluation." >&2
  exit 1
fi
if [[ ! "$MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]] || [[ ! "$CLEANUP_RESERVE_MINUTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASS2_MIN_FREE_GB and ASS2_CLEANUP_RESERVE_MINUTES must be positive integers." >&2
  exit 1
fi
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "Setup environments are missing. Submit assignment2/scripts/setup.sbatch first." >&2
  exit 1
fi

for path in "$CACHE_ROOT" "$MODEL_CACHE" "$PACKAGE_CACHE" "$RESULTS_ROOT" "$LOG_DIRECTORY"; do
  mkdir -p "$path"
done

check_free_space() {
  local path="$1"
  local available_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  if [[ -z "$available_kb" || "$available_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]]; then
    echo "Insufficient visible free space at $path; need at least ${MIN_FREE_GB} GiB." >&2
    exit 1
  fi
}

check_free_space "$ASS2_ROOT"
if command -v quota >/dev/null 2>&1; then
  quota -s || echo "quota did not report a usable limit; verify quota/retention separately."
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "The compute-node timeout command is required to preserve cleanup time." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TMPDIR="$ASS2_ROOT/tmp/runtime-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PATH="$(dirname "$EVAL_PYTHON"):$PATH"
export PIP_CACHE_DIR="$PACKAGE_CACHE"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$MODEL_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLLM_CACHE_ROOT="$MODEL_CACHE/vllm"
export TORCH_HOME="$MODEL_CACHE/torch"
export TRITON_CACHE_DIR="$MODEL_CACHE/triton"
export ASSIGNMENT2_MODEL_CACHE="$MODEL_CACHE"
export ASSIGNMENT2_LOG_DIRECTORY="$LOG_DIRECTORY"
export ASSIGNMENT2_OUTPUT_ROOT="$RESULTS_ROOT"
export ASSIGNMENT2_BUDGET_LEDGER_PATH="$RESULTS_ROOT/api_spending_ledger.json"
export ASSIGNMENT2_EVAL_PYTHON="$EVAL_PYTHON"
export ASSIGNMENT2_SERVE_PYTHON="$SERVE_PYTHON"
mkdir -p "$XDG_CACHE_HOME"

cd "$REPO_ROOT"
for model_name in "${MODEL_ARGS[@]}"; do
  case "$model_name" in
    openai_mini|qwen_7b_local) ;;
    *) echo "Unknown model profile: $model_name" >&2; exit 1 ;;
  esac
done
if [[ " $MODELS " == *" openai_mini "* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set in the submitting environment." >&2
  exit 1
fi
if [[ " $MODELS " == *" qwen_7b_local "* ]]; then
  "$REPO_ROOT/assignment2/scripts/check_python_environments.sh" "$EVAL_PYTHON" "$SERVE_PYTHON"
  MODEL_ID="$("$EVAL_PYTHON" -c 'import sys; from omegaconf import OmegaConf; config = OmegaConf.load(sys.argv[1]); print(config.models.qwen_7b_local.model_id)' "$REPO_ROOT/assignment2/config/experiment.yaml")"
  export MODEL_ID
  QWEN_SNAPSHOT="$("$SERVE_PYTHON" -c '
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, cache_dir = sys.argv[1:]
snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=True))
if not (snapshot / "config.json").is_file():
    raise RuntimeError(f"Prepared model snapshot has no config.json: {snapshot}")
print(snapshot)
' "$MODEL_ID" "$HF_HUB_CACHE")"
  export ASSIGNMENT2_QWEN_SNAPSHOT="$QWEN_SNAPSHOT"
  echo "Using prepared model snapshot: $QWEN_SNAPSHOT"
else
  "$REPO_ROOT/assignment2/scripts/check_python_environments.sh" "$EVAL_PYTHON"
fi

if [[ -n "$RESUME_DIRECTORY" ]]; then
  RUN_DIRECTORY="$RESUME_DIRECTORY"
  RUN_ARGUMENTS=(--resume "$RUN_DIRECTORY")
else
  RUN_ARGUMENTS=(--run-directory "$RUN_DIRECTORY")
fi
case "$RUN_DIRECTORY" in
  "$RESULTS_ROOT"/*) ;;
  *)
    echo "Run and resume directories must be inside $RESULTS_ROOT." >&2
    exit 1
    ;;
esac

SERVER_STARTED=0
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ "$SERVER_STARTED" == "1" ]]; then
    if ! "$REPO_ROOT/assignment2/scripts/stop_vllm.sh"; then
      echo "vLLM cleanup failed." >&2
      exit_code=1
    fi
  fi
  printf 'job_id=%s\nphase=%s\nexit_code=%s\n' "$SLURM_JOB_ID" "$PHASE" "$exit_code" \
    > "$ASS2_ROOT/logs/run-$SLURM_JOB_ID.status"
  echo "ASSIGNMENT2_JOB_EXIT exit_code=$exit_code"
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Python holds the exclusive paid-ledger lock for all cloud calls and resumes.

if [[ " $MODELS " == *" qwen_7b_local "* ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to check the local vLLM endpoint." >&2
    exit 1
  fi
  if curl -fsS --max-time 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "Port 8080 already serves a model; refusing to reuse or disturb another server." >&2
    exit 1
  fi
  "$REPO_ROOT/assignment2/scripts/start_vllm.sh"
  SERVER_STARTED=1
  for attempt in $(seq 1 450); do
    if ! kill -0 "$(cat "$LOG_DIRECTORY/vllm.pid")" 2>/dev/null; then
      echo "The job's vLLM process exited; check $LOG_DIRECTORY/vllm.log" >&2
      exit 1
    fi
    if curl -fsS --max-time 5 http://127.0.0.1:8080/v1/models > "$LOG_DIRECTORY/models.json" 2>/dev/null; then
      "$EVAL_PYTHON" -c 'import json, sys; data = json.load(open(sys.argv[1])); assert sys.argv[2] in [item["id"] for item in data["data"]], "Unexpected model at local endpoint"' "$LOG_DIRECTORY/models.json" "$MODEL_ID"
      break
    fi
    sleep 2
    if [[ "$attempt" == "450" ]]; then
      echo "vLLM did not become ready. Check $LOG_DIRECTORY/vllm.log" >&2
      exit 1
    fi
  done
fi

AVAILABLE_EXPERIMENT_SECONDS=$((JOB_TIME_LIMIT_SECONDS - CLEANUP_RESERVE_MINUTES * 60 - TERMINATION_GRACE_SECONDS - SECONDS))
if (( AVAILABLE_EXPERIMENT_SECONDS <= 0 )); then
  echo "The cleanup reserve leaves no time for evaluation." >&2
  exit 1
fi
echo "Evaluation is limited to ${AVAILABLE_EXPERIMENT_SECONDS}s; ${CLEANUP_RESERVE_MINUTES} minute(s) remain reserved for saving and cleanup."
timeout --signal=TERM --kill-after="$TERMINATION_GRACE_SECONDS" "$AVAILABLE_EXPERIMENT_SECONDS" \
  "$EVAL_PYTHON" -u -m assignment2.run_experiment --phase "$PHASE" "${RUN_ARGUMENTS[@]}" --models "${MODEL_ARGS[@]}"

if [[ "$PHASE" == "final" ]]; then
  AVAILABLE_ANALYSIS_SECONDS=$((JOB_TIME_LIMIT_SECONDS - CLEANUP_RESERVE_MINUTES * 60 - TERMINATION_GRACE_SECONDS - SECONDS))
  if (( AVAILABLE_ANALYSIS_SECONDS <= 0 )); then
    echo "No time remains for final analysis before the cleanup reserve." >&2
    exit 1
  fi
  timeout --signal=TERM --kill-after="$TERMINATION_GRACE_SECONDS" "$AVAILABLE_ANALYSIS_SECONDS" \
    "$EVAL_PYTHON" -m assignment2.analysis "$RUN_DIRECTORY" --phase final --output "$RUN_DIRECTORY/analysis"
else
  echo "Smoke validation finished. Final model jobs analyze their own complete records; combine both final directories for cross-model analysis."
fi

echo "ASSIGNMENT2_PHASE_OK phase=$PHASE models=$MODELS run=$RUN_DIRECTORY"
