#!/usr/bin/env bash
# Allocated diagnostics: offline serving readiness, no inference or API calls.
#SBATCH --job-name=assignment2-diagnostic
#SBATCH --partition=L40s_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation.}"
: "${ASS2_ROOT:?Set ASS2_ROOT=/data/s4822285/assignment2.}"
cd "$ASS2_ROOT/repo"
EVAL_PYTHON="$ASS2_ROOT/envs/balrog-prebuilt/bin/python"
SERVE_PYTHON="$ASS2_ROOT/envs/vllm-prebuilt/bin/python"
export TMPDIR="$ASS2_ROOT/tmp/runtime-$SLURM_JOB_ID"
export PIP_CACHE_DIR="$ASS2_ROOT/cache/pip"
export XDG_CACHE_HOME="$ASS2_ROOT/cache/xdg"
export ASSIGNMENT2_MODEL_CACHE="$ASS2_ROOT/cache/models"
export ASSIGNMENT2_LOG_DIRECTORY="$ASS2_ROOT/logs/diagnostic-vllm-$SLURM_JOB_ID"
export ASSIGNMENT2_SERVE_PYTHON="$SERVE_PYTHON"
export ASSIGNMENT2_BUDGET_LEDGER_PATH="$ASS2_ROOT/results/api_spending_ledger.json"
export HF_HOME="$ASSIGNMENT2_MODEL_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PATH="$(dirname "$EVAL_PYTHON"):$PATH"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$ASSIGNMENT2_LOG_DIRECTORY"
SERVER_STARTED=0
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ "$SERVER_STARTED" == 1 ]]; then
    assignment2/scripts/stop_vllm.sh || exit_code=1
  fi
  printf 'job_id=%s\nexit_code=%s\n' "$SLURM_JOB_ID" "$exit_code" \
    > "$ASS2_ROOT/logs/diagnostic-$SLURM_JOB_ID.status"
  echo "ASSIGNMENT2_DIAGNOSTIC_EXIT exit_code=$exit_code"
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u
hostname -f
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# The old setup omitted plotting. Repair only that dependency if absent.
if ! "$EVAL_PYTHON" -c 'import importlib.util; assert importlib.util.find_spec("matplotlib")'; then
  "$EVAL_PYTHON" -m pip install matplotlib
fi
"$EVAL_PYTHON" - <<'PY'
import importlib.metadata
import os
import sys
print('evaluation_python', sys.version)
for name in ['balrog','openai','omegaconf','gym','gymnasium','crafter','minigrid','matplotlib']:
    print(name, importlib.metadata.version(name))
print('OPENAI_API_KEY', 'present' if os.environ.get('OPENAI_API_KEY') else 'absent')
from assignment2.runner import source_provenance
print('source_provenance', source_provenance('.'))
from omegaconf import OmegaConf
from balrog.environments.babyai_text.babyai_env import make_babyai_env
from balrog.environments.crafter.crafter_env import make_crafter_env
config = OmegaConf.load('balrog/config/config.yaml')
# Reset/close checks construct observations; no agent, action, or episode is run.
for benchmark, factory, task in [
    ('babyai', make_babyai_env, 'BabyAI-MixedTrainLocal-v0/goto'),
    ('crafter', make_crafter_env, 'default'),
]:
    environment = factory(benchmark, task, config)
    try:
        observation, info = environment.reset(seed=101 if benchmark == 'babyai' else None)
        assert isinstance(observation, dict) and 'text' in observation
        print('BENCHMARK_RESET_OK', benchmark, 'actions=0')
    finally:
        environment.close()
import matplotlib
matplotlib.use('Agg')
import assignment2.analysis
PY
assignment2/scripts/check_python_environments.sh "$EVAL_PYTHON" "$SERVE_PYTHON"
"$SERVE_PYTHON" - <<'PY'
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
print('serving_python', sys.version)
for name in ['torch','vllm','huggingface-hub']:
    print(name, importlib.metadata.version(name))
assert torch.cuda.is_available(), 'CUDA is unavailable in this allocation'
cache = Path(os.environ['ASS2_ROOT'])/'cache/models/huggingface/hub'
snapshot = Path(snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir=str(cache), local_files_only=True))
assert (snapshot/'config.json').is_file()
weights = list(snapshot.glob('*.safetensors'))
assert weights and all(path.is_file() and path.stat().st_size > 0 for path in weights)
index = json.loads((snapshot/'model.safetensors.index.json').read_text())
assert all((snapshot/name).is_file() for name in set(index['weight_map'].values()))
assert (snapshot/'tokenizer.json').is_file()
model_config = json.loads((snapshot/'config.json').read_text())
assert model_config['model_type'] == 'qwen2'
assert (snapshot/'tokenizer_config.json').is_file()
assert all((snapshot/name).stat().st_size > 0 for name in set(index['weight_map'].values()))
print('cached_qwen_snapshot', snapshot)
print('weight_shards', len(weights))
print('weight_bytes', sum(path.stat().st_size for path in weights))
Path(os.environ['TMPDIR'], 'qwen-snapshot.txt').write_text(str(snapshot))
PY
"$EVAL_PYTHON" -m unittest discover -s tests -v
"$EVAL_PYTHON" -m assignment2.audit_results "$ASS2_ROOT/results"
"$EVAL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from collections import Counter
from assignment2.validation import validate_run_directory, IncompletePhaseError
path = Path(os.environ['ASS2_ROOT'])/'results/api_spending_ledger.json'
if path.exists():
    state = json.loads(path.read_text())
    print('ledger', {key: state.get(key) for key in
                    ['schema_version','ceiling_usd','estimated_spent_usd','reserved_usd','pricing_policy']})
    print('ledger_call_count', len(state.get('calls', [])))
    assert state['schema_version'] == 1 and isinstance(state['calls'], list)
    assert len(state['calls']) == 26, 'Historical ledger changed; review before migration'
    assert abs(state['estimated_spent_usd'] - .203296) < 1e-12
    assert state['reserved_usd'] == 0
    assert all(call['estimated_cost_usd'] >= 0 for call in state['calls'])
    assert abs(sum(call['estimated_cost_usd'] for call in state['calls']) - state['estimated_spent_usd']) < 1e-10
    print('historical_call_roles', dict(Counter(call.get('role') for call in state['calls'])))
    if state['ceiling_usd'] == 10:
        (Path(os.environ['TMPDIR'])/'ledger-before-migration.json').write_bytes(path.read_bytes())
else:
    print('No ledger file at', path)
    raise SystemExit('Expected existing ledger is missing')
for name in ['smoke_slurm-22287', 'smoke_slurm-22289']:
    directory = path.parent / name
    records = [json.loads(item.read_text()) for item in (directory/'episodes').glob('*.json')]
    print('historical_run', name, 'records', len(records),
          'status_counts', dict(Counter(item.get('status') for item in records)),
          'records_with_steps', sum(bool(item.get('steps')) for item in records))
    try:
        validate_run_directory(directory)
        print('HISTORICAL_CURRENT_VALIDATION_OK', name)
    except IncompletePhaseError as error:
        print('HISTORICAL_CURRENT_VALIDATION_FAILED', name, str(error).splitlines()[0])
PY
# Explicit reviewed CLI: no arguments. It creates its own exact-byte backup.
"$EVAL_PYTHON" -m assignment2.migrate_budget
"$EVAL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ['ASSIGNMENT2_BUDGET_LEDGER_PATH'])
state = json.loads(path.read_text())
backup = Path(state['migrations'][-1]['backup'])
before = json.loads(backup.read_text())
original = Path(os.environ['TMPDIR'])/'ledger-before-migration.json'
if original.exists():
    assert backup.read_bytes() == original.read_bytes(), 'Backup differs from original bytes'
assert before['ceiling_usd'] == 10 and state['ceiling_usd'] == 60
for field in ['calls', 'estimated_spent_usd', 'reserved_usd']:
    assert before[field] == state[field], f'Migration changed historical {field}'
print('LEDGER_MIGRATION_VERIFIED', 'backup', backup, 'calls', len(state['calls']),
      'spent_usd', state['estimated_spent_usd'], 'reserved_usd', state['reserved_usd'],
      'remaining_usd', 60 - state['estimated_spent_usd'] - state['reserved_usd'])
PY
"$EVAL_PYTHON" -m pip check
"$SERVE_PYTHON" -m pip check
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export ASSIGNMENT2_QWEN_SNAPSHOT="$(cat "$TMPDIR/qwen-snapshot.txt")"
if curl -fsS --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo 'Port 8080 is occupied; refusing to disturb its server.' >&2
  exit 1
fi
assignment2/scripts/start_vllm.sh
SERVER_STARTED=1
ready=0
for attempt in $(seq 1 450); do
  kill -0 "$(cat "$ASSIGNMENT2_LOG_DIRECTORY/vllm.pid")" 2>/dev/null || {
    echo "vLLM exited; inspect $ASSIGNMENT2_LOG_DIRECTORY/vllm.log" >&2
    exit 1
  }
  if curl -fsS --max-time 5 http://127.0.0.1:8080/v1/models \
      > "$ASSIGNMENT2_LOG_DIRECTORY/models.json" 2>/dev/null; then
    "$EVAL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
data = json.loads((Path(os.environ['ASSIGNMENT2_LOG_DIRECTORY'])/'models.json').read_text())
ids = [item['id'] for item in data['data']]
assert os.environ['MODEL_ID'] in ids
print('QWEN_OFFLINE_STARTUP_OK', ids, 'inference_requests=0')
PY
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'vLLM readiness timed out.' >&2; exit 1; }
"$EVAL_PYTHON" -m pip freeze > "$ASS2_ROOT/logs/diagnostic-$SLURM_JOB_ID-evaluation.txt"
"$SERVE_PYTHON" -m pip freeze > "$ASS2_ROOT/logs/diagnostic-$SLURM_JOB_ID-serving.txt"
echo ASSIGNMENT2_DIAGNOSTIC_OK
