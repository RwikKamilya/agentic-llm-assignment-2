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
EVAL_PYTHON="$ASS2_ROOT/envs/balrog/bin/python"
SERVE_PYTHON="$ASS2_ROOT/envs/vllm/bin/python"
CACHE_ROOT="$ASS2_ROOT/cache"
MODEL_CACHE="$CACHE_ROOT/models"
PACKAGE_CACHE="$CACHE_ROOT/pip"
RESULTS_ROOT="$ASS2_ROOT/results"
LOG_DIRECTORY="$ASS2_ROOT/logs/vllm-$SLURM_JOB_ID"
PHASE="${ASS2_PHASE:?Set ASS2_PHASE to smoke, pilot, or final.}"
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
if [[ "$PHASE" != "smoke" && "$PHASE" != "pilot" && "$PHASE" != "final" ]]; then
  echo "ASS2_PHASE must be smoke, pilot, or final." >&2
  exit 1
fi
read -r -a MODEL_ARGS <<< "$MODELS"
if (( ${#MODEL_ARGS[@]} == 0 )); then
  echo "ASS2_MODELS must name at least one model profile." >&2
  exit 1
fi
if [[ ! "$MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]] || [[ ! "$CLEANUP_RESERVE_MINUTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASS2_MIN_FREE_GB and ASS2_CLEANUP_RESERVE_MINUTES must be positive integers." >&2
  exit 1
fi
if [[ ! -x "$EVAL_PYTHON" || ! -x "$SERVE_PYTHON" ]]; then
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

export PIP_CACHE_DIR="$PACKAGE_CACHE"
export HOME="$ASS2_ROOT/home"
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
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR"

cd "$REPO_ROOT"
"$REPO_ROOT/assignment2/scripts/check_python_environments.sh" "$EVAL_PYTHON" "$SERVE_PYTHON"
MODEL_ID="$("$EVAL_PYTHON" -c 'import sys; from omegaconf import OmegaConf; config = OmegaConf.load(sys.argv[1]); assert float(config.budget.total_ceiling_usd) == 10.0; print(config.models.qwen_7b_local.model_id)' "$REPO_ROOT/assignment2/config/experiment.yaml")"
export MODEL_ID
"$SERVE_PYTHON" -c '
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, cache_dir = sys.argv[1:]
snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=True))
if not (snapshot / "config.json").is_file():
    raise RuntimeError(f"Prepared model snapshot has no config.json: {snapshot}")
print(f"Using prepared model snapshot: {snapshot}")
' "$MODEL_ID" "$HF_HUB_CACHE"

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
CLOUD_LOCK_HELD=0
BUDGET_LOCK_DIRECTORY="$RESULTS_ROOT/api_spending_ledger.lock"
cleanup() {
  if [[ "$SERVER_STARTED" == "1" ]]; then
    "$REPO_ROOT/assignment2/scripts/stop_vllm.sh" || true
  fi
  if [[ "$CLOUD_LOCK_HELD" == "1" ]]; then
    rm -f "$BUDGET_LOCK_DIRECTORY/owner"
    rmdir "$BUDGET_LOCK_DIRECTORY" || true
  fi
}
trap cleanup EXIT

if [[ " $MODELS " == *" gemini_flash_lite "* ]]; then
  if ! mkdir "$BUDGET_LOCK_DIRECTORY" 2>/dev/null; then
    echo "Another cloud-evaluation job appears to hold $BUDGET_LOCK_DIRECTORY. Do not run Gemini jobs concurrently." >&2
    exit 1
  fi
  CLOUD_LOCK_HELD=1
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID" > "$BUDGET_LOCK_DIRECTORY/owner"
fi

if [[ " $MODELS " == *" qwen_7b_local "* ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to check the local vLLM endpoint." >&2
    exit 1
  fi
  "$REPO_ROOT/assignment2/scripts/start_vllm.sh"
  SERVER_STARTED=1
  for attempt in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8080/v1/models >/dev/null; then
      break
    fi
    sleep 2
    if [[ "$attempt" == "30" ]]; then
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
  "$EVAL_PYTHON" assignment2/run_experiment.py --phase "$PHASE" "${RUN_ARGUMENTS[@]}" --models "${MODEL_ARGS[@]}"

if [[ "$PHASE" == "final" && " $MODELS " == *" gemini_flash_lite "* && " $MODELS " == *" qwen_7b_local "* ]]; then
  AVAILABLE_ANALYSIS_SECONDS=$((JOB_TIME_LIMIT_SECONDS - CLEANUP_RESERVE_MINUTES * 60 - TERMINATION_GRACE_SECONDS - SECONDS))
  if (( AVAILABLE_ANALYSIS_SECONDS <= 0 )); then
    echo "No time remains for final analysis before the cleanup reserve." >&2
    exit 1
  fi
  timeout --signal=TERM --kill-after="$TERMINATION_GRACE_SECONDS" "$AVAILABLE_ANALYSIS_SECONDS" \
    "$EVAL_PYTHON" assignment2/analysis.py "$RUN_DIRECTORY" --phase final --output "$RUN_DIRECTORY/analysis"
else
  echo "Final combined analysis was not run: supply both model profiles in one final job, or combine result directories separately."
fi
