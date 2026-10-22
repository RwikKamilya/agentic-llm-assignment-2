#!/usr/bin/env bash
# Run on calcium. Only scheduling/file operations run on the login node.
set -euo pipefail
export ASS2_ROOT=/data/s4822285/assignment2
export ASS2_REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RELEASE_NAME="$(basename "$ASS2_REPO_ROOT")"
if [[ "$RELEASE_NAME" != ass2-v2-* || ! -f "$ASS2_REPO_ROOT/assignment2/deployed_version.json" ]]; then
  echo "Run this script from the frozen release under $ASS2_ROOT/releases." >&2
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Export OPENAI_API_KEY in this shell before submission." >&2
  exit 1
fi
if [[ ! -f "$ASS2_ROOT/results/api_spending_ledger.json" ]]; then
  echo "Historical cumulative spending ledger is missing." >&2
  exit 1
fi
export ASS2_CAMPAIGN_ROOT="$ASS2_ROOT/results/$RELEASE_NAME"
export ASS2_PHASE=campaign
export ASS2_MIN_FREE_GB=5
mkdir -p "$ASS2_ROOT/logs"
# A repeated invocation must not silently launch a second paid experiment.
mkdir "$ASS2_CAMPAIGN_ROOT"
printf 'release=%s\nresults=%s\n' "$ASS2_REPO_ROOT" "$ASS2_CAMPAIGN_ROOT" > "$ASS2_CAMPAIGN_ROOT/submission.txt"
export ASS2_MODELS=qwen_7b_local
QWEN_JOB="$(sbatch --parsable --chdir="$ASS2_REPO_ROOT" --export=ALL \
  --output="$ASS2_ROOT/logs/v2-qwen-%j.out" --error="$ASS2_ROOT/logs/v2-qwen-%j.err" \
  "$ASS2_REPO_ROOT/assignment2/scripts/run_pipeline.sbatch")"
QWEN_JOB="${QWEN_JOB%%;*}"
printf 'qwen_job=%s\n' "$QWEN_JOB" | tee -a "$ASS2_CAMPAIGN_ROOT/submission.txt"
export ASS2_MODELS=openai_mini
OPENAI_JOB="$(sbatch --parsable --chdir="$ASS2_REPO_ROOT" --export=ALL \
  --output="$ASS2_ROOT/logs/v2-openai-%j.out" --error="$ASS2_ROOT/logs/v2-openai-%j.err" \
  "$ASS2_REPO_ROOT/assignment2/scripts/run_pipeline.sbatch")"
OPENAI_JOB="${OPENAI_JOB%%;*}"
printf 'openai_job=%s\n' "$OPENAI_JOB" | tee -a "$ASS2_CAMPAIGN_ROOT/submission.txt"
ANALYSIS_JOB="$(sbatch --parsable --chdir="$ASS2_REPO_ROOT" --export=ALL \
  --dependency="afterok:$QWEN_JOB:$OPENAI_JOB" --kill-on-invalid-dep=yes \
  --output="$ASS2_ROOT/logs/v2-analysis-%j.out" --error="$ASS2_ROOT/logs/v2-analysis-%j.err" \
  "$ASS2_REPO_ROOT/assignment2/scripts/analyze_campaign.sbatch")"
printf 'analysis_job=%s\n' "$ANALYSIS_JOB" | tee -a "$ASS2_CAMPAIGN_ROOT/submission.txt"
echo "Results: $ASS2_CAMPAIGN_ROOT"
echo "Monitor: squeue -u $(id -un)"
