#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "User: $(id -un)"
echo "Working directory: $(pwd)"
echo
echo "Known deployment status: l40s connects to the calcium.liacs.nl gateway/head node."
echo "No computational jobs, model servers, or experiments may run on this node."
echo "Slurm is available; request an allocation before checking GPU visibility."
echo
echo "GPU visibility (only meaningful inside a Slurm allocation):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi is unavailable. Do not start a local server until the site documents the allocation method."
fi
echo
echo "Slurm commands (discovery only; no job is submitted):"
for command_name in sbatch srun squeue sinfo; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%s: %s\n' "$command_name" "$(command -v "$command_name")"
  fi
done
echo
echo "Python: $(command -v python3 || true)"
python3 --version || true
echo
echo "Home filesystem status (do not use /home for environments, package caches, or model caches):"
df -h "$HOME" || true
