#!/usr/bin/env bash
# Setup runs on an allocated compute node. It does not call cloud APIs or run any
# benchmark episode. The user supplies --chdir, --output, and --error to sbatch.
#SBATCH --job-name=assignment2-setup
#SBATCH --partition=L40s_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=12:00:00

set -euo pipefail

ASS2_ROOT="${ASS2_ROOT:?Set ASS2_ROOT=/data/s4822285/assignment2 before submission.}"
if [[ "$ASS2_ROOT" != "/data/s4822285/assignment2" ]]; then
  echo "ASS2_ROOT must be /data/s4822285/assignment2 for this deployment." >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Setup must run inside a Slurm allocation, never on calcium." >&2
  exit 1
fi

REPO_ROOT="$ASS2_ROOT/repo"
PYTHON_VERSION="3.11.11"
EVAL_ENV="$ASS2_ROOT/envs/balrog-prebuilt"
SERVE_ENV="$ASS2_ROOT/envs/vllm-prebuilt"
EVAL_PYTHON="$EVAL_ENV/bin/python"
SERVE_PYTHON="$SERVE_ENV/bin/python"
CACHE_ROOT="$ASS2_ROOT/cache"
MODEL_CACHE="$CACHE_ROOT/models"
PACKAGE_CACHE="$CACHE_ROOT/pip"
BUILD_ROOT="$ASS2_ROOT/build"
SETUP_ROOT="$ASS2_ROOT/setup"
SETUP_RECORD="$SETUP_ROOT/setup-$SLURM_JOB_ID"
MIN_FREE_GB="${ASS2_MIN_FREE_GB:?Set a quota-aware minimum free-space threshold in GiB.}"

if [[ ! -d "$REPO_ROOT" || ! -f "$REPO_ROOT/setup.py" || ! -f "$REPO_ROOT/assignment2/config/experiment.yaml" ]]; then
  echo "Expected repository is missing or incomplete: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! "$MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASS2_MIN_FREE_GB must be a positive integer." >&2
  exit 1
fi

for command_name in curl tar git nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required setup command is unavailable: $command_name" >&2
    exit 1
  fi
done

for path in "$CACHE_ROOT" "$MODEL_CACHE" "$PACKAGE_CACHE" "$BUILD_ROOT" "$SETUP_ROOT" "$SETUP_RECORD"; do
  mkdir -p "$path"
done

export TMPDIR="$BUILD_ROOT/tmp-$SLURM_JOB_ID"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="$PACKAGE_CACHE"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$MODEL_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLLM_CACHE_ROOT="$MODEL_CACHE/vllm"
export TORCH_HOME="$MODEL_CACHE/torch"
export TRITON_CACHE_DIR="$MODEL_CACHE/triton"
export PIP_NO_INPUT=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR"

check_free_space() {
  local available_kb
  available_kb="$(df -Pk "$ASS2_ROOT" | awk 'NR == 2 {print $4}')"
  if [[ -z "$available_kb" || "$available_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]]; then
    echo "Insufficient visible free space under $ASS2_ROOT; need at least ${MIN_FREE_GB} GiB." >&2
    exit 1
  fi
}

check_free_space
if command -v quota >/dev/null 2>&1; then
  quota -s || echo "quota did not report a usable limit; verify quota/retention separately."
fi

# Use a complete prebuilt interpreter instead of compiling CPython.
export UV_PYTHON_INSTALL_DIR="$ASS2_ROOT/python/uv-managed"
export UV_PYTHON_BIN_DIR="$ASS2_ROOT/tools/python-bin"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

UV="$ASS2_ROOT/tools/uv/uv"
if [[ ! -x "$UV" ]]; then
  curl --fail --location --retry 3 \
    https://astral.sh/uv/install.sh \
    --output "$SETUP_RECORD/install-uv.sh"

  UV_UNMANAGED_INSTALL="$ASS2_ROOT/tools/uv" \
    sh "$SETUP_RECORD/install-uv.sh"
fi

"$UV" --version
"$UV" python install "$PYTHON_VERSION"
BASE_PYTHON="$("$UV" python find --managed-python "$PYTHON_VERSION")"

"$BASE_PYTHON" -c '
import ssl, ctypes, sqlite3, bz2, lzma, zlib, multiprocessing, sys
assert sys.version_info[:2] == (3, 11)
print("Interpreter validation passed:", sys.executable)
print(sys.version)
'

if [[ ! -x "$EVAL_PYTHON" ]]; then
  "$BASE_PYTHON" -m venv "$EVAL_ENV"
fi
"$EVAL_PYTHON" -m pip install --upgrade pip "setuptools==80.9.0" wheel
"$EVAL_PYTHON" -m pip install --editable "$REPO_ROOT" matplotlib
"$EVAL_PYTHON" -m pip install "setuptools==80.9.0"
"$EVAL_PYTHON" -c 'import pkg_resources; print("pkg_resources available")'
"$EVAL_PYTHON" -m pip check


# BabyAI and Crafter need no optional BALROG asset downloads.
mkdir -p "$SETUP_RECORD/balrog-assets"
(
  cd "$SETUP_RECORD/balrog-assets"
  echo "Skipping optional benchmark asset downloads for BabyAI/Crafter setup."
)

if [[ ! -x "$SERVE_PYTHON" ]]; then
  "$BASE_PYTHON" -m venv "$SERVE_ENV"
fi
"$SERVE_PYTHON" -m pip install --upgrade pip setuptools wheel
"$SERVE_PYTHON" -m pip install vllm
"$SERVE_PYTHON" -m pip check

"$REPO_ROOT/assignment2/scripts/check_python_environments.sh" "$EVAL_PYTHON" "$SERVE_PYTHON"
"$EVAL_PYTHON" -m unittest discover -s "$REPO_ROOT/tests" -v

MODEL_ID="$("$EVAL_PYTHON" -c 'import sys; from omegaconf import OmegaConf; config = OmegaConf.load(sys.argv[1]); print(config.models.qwen_7b_local.model_id)' "$REPO_ROOT/assignment2/config/experiment.yaml")"
"$SERVE_PYTHON" -c '
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, cache_dir = sys.argv[1:]
snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=cache_dir))
if not (snapshot / "config.json").is_file():
    raise RuntimeError(f"Model snapshot has no config.json: {snapshot}")
print(f"Prepared model snapshot: {snapshot}")
' "$MODEL_ID" "$HF_HUB_CACHE"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$SETUP_RECORD/gpu.csv"
if ! grep -qi 'NVIDIA L40S' "$SETUP_RECORD/gpu.csv"; then
  echo "Allocated GPU is not reported as NVIDIA L40S." >&2
  exit 1
fi
"$SERVE_PYTHON" -c 'import torch, vllm; assert torch.cuda.is_available(); print(f"torch={torch.__version__} vllm={vllm.__version__}")'

"$EVAL_PYTHON" -m pip freeze > "$SETUP_RECORD/evaluation-requirements.txt"
"$SERVE_PYTHON" -m pip freeze > "$SETUP_RECORD/serving-requirements.txt"
printf '%s\n' "$MODEL_ID" > "$SETUP_RECORD/model-id.txt"
printf '%s\n' "$REPO_ROOT" > "$SETUP_RECORD/repository-root.txt"

echo "ASSIGNMENT2_SETUP_OK"
