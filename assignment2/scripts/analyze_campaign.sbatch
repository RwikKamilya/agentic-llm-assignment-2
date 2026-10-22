#!/usr/bin/env bash
#SBATCH --job-name=assignment2-analysis
#SBATCH --partition=L40s_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside Slurm}"
: "${ASS2_ROOT:?}"
: "${ASS2_REPO_ROOT:?}"
: "${ASS2_CAMPAIGN_ROOT:?}"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$ASS2_ROOT/cache/xdg"
cd "$ASS2_REPO_ROOT"
"$ASS2_ROOT/envs/balrog-prebuilt/bin/python" -m assignment2.analysis \
  "$ASS2_CAMPAIGN_ROOT/qwen_7b_local/final" "$ASS2_CAMPAIGN_ROOT/openai_mini/final" \
  --phase final --output "$ASS2_CAMPAIGN_ROOT/analysis"
printf 'ASSIGNMENT2_ANALYSIS_OK\n' > "$ASS2_CAMPAIGN_ROOT/ANALYSIS_COMPLETE"
"$ASS2_ROOT/envs/balrog-prebuilt/bin/python" -m assignment2.package_submission "$ASS2_CAMPAIGN_ROOT"
printf 'ASSIGNMENT2_CAMPAIGN_OK\n' > "$ASS2_CAMPAIGN_ROOT/COMPLETE"
echo "ASSIGNMENT2_CAMPAIGN_OK results=$ASS2_CAMPAIGN_ROOT"
