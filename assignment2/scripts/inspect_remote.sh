#!/usr/bin/env bash
# Gateway-safe audit: filesystem, Git and Slurm only. Never Python or GPU checks.
set -euo pipefail
ASS2_ROOT="${ASS2_ROOT:-/data/s4822285/assignment2}"
REPO_ROOT="$ASS2_ROOT/repo"
date -u
hostname -f
id
if [[ "$(hostname -s)" != calcium ]]; then
  echo "Expected calcium gateway was not confirmed; do not deploy until resolved."
fi
cd "$REPO_ROOT"
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' || true
git rev-list --left-right --count 'HEAD...@{upstream}' || true
git diff --name-status
git ls-files --others --exclude-standard
# Filenames only: never read or diff suspected credential files.
git ls-files | awk '/(^|\/)(SECRETS|\.env|credentials)(\/|$|\.)/ {print "tracked suspected secret: " $0}'
git log -5 --format='%H %cs %s' -- assignment2/config/experiment.yaml
wc -c assignment2/config/experiment.yaml
find assignment2/config -maxdepth 2 -type f -name '*experiment*' -exec ls -ln {} \;
find "$ASS2_ROOT" -maxdepth 3 -type f \( -name 'experiment.yaml*' -o -name '*experiment*.bak' \) -print
find assignment2 balrog -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
  -not -path '*/results/*' -not -path '*/__pycache__/*' -exec sha256sum {} \;
squeue -u "$(id -un)" -o '%.18i %.20j %.10T %.12M %.20R'
scontrol show partition L40s_students
sinfo -p L40s_students -o '%P %a %l %G'
df -h "$ASS2_ROOT"
if command -v quota >/dev/null 2>&1; then quota -s || true; fi
for path in envs cache setup results logs; do
  if [[ -e "$ASS2_ROOT/$path" ]]; then
    ls -ldn "$ASS2_ROOT/$path"
    du -sh "$ASS2_ROOT/$path"
  fi
done
find "$ASS2_ROOT/setup" -maxdepth 2 -type f \( -name '*requirements.txt' -o -name 'model-id.txt' -o -name 'gpu.csv' \) -print 2>/dev/null || true
find "$ASS2_ROOT/logs" -maxdepth 1 -type f -name 'setup-*.out' -exec grep -lFx ASSIGNMENT2_SETUP_OK {} \; 2>/dev/null || true
find "$ASS2_ROOT/cache/models/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" \
  -maxdepth 2 -name config.json -exec ls -ln {} \; 2>/dev/null || true
find "$ASS2_ROOT/results" -maxdepth 2 -mindepth 1 -type d -print 2>/dev/null || true
for path in "$ASS2_ROOT/results/api_spending_ledger.json" "$ASS2_ROOT/results/api_spending_ledger.lock"; do
  if [[ -e "$path" ]]; then ls -ldn "$path"; fi
done
# Inspect structured, nonsensitive owner metadata only.
if [[ -f "$ASS2_ROOT/results/api_spending_ledger.lock/owner" ]]; then
  cat "$ASS2_ROOT/results/api_spending_ledger.lock/owner"
fi
if command -v jq >/dev/null 2>&1 && [[ -f "$ASS2_ROOT/results/api_spending_ledger.json" ]]; then
  jq '{schema_version, ceiling_usd, estimated_spent_usd, reserved_usd, calls: (.calls | length)}' \
    "$ASS2_ROOT/results/api_spending_ledger.json"
fi
for key in OPENAI_API_KEY GEMINI_API_KEY; do
  if [[ -n "${!key:-}" ]]; then printf '%s present\n' "$key"; else printf '%s absent\n' "$key"; fi
done
