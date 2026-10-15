#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "User: $(id -un)"
echo "Working directory: $(pwd)"
echo
echo "GPU visibility (this does not establish an allocation):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi is unavailable. Do not start a local server until the site documents the allocation method."
fi
echo
echo "Scheduler commands found (discovery only; no job is submitted):"
for command_name in squeue sbatch sinfo qstat qsub bsub; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%s: %s\n' "$command_name" "$(command -v "$command_name")"
  fi
done
echo
echo "Python: $(command -v python3 || true)"
python3 --version || true
