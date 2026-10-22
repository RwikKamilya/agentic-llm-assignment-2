#!/usr/bin/env bash
# Optional offline harness check. The final campaign already runs these checks.
#SBATCH --job-name=assignment2-check
#SBATCH --partition=L40s_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Must run inside Slurm}"
ASS2_ROOT="${ASS2_ROOT:-/data/s4822285/assignment2}"
REPO_ROOT="${ASS2_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
EVAL_PYTHON="$ASS2_ROOT/envs/balrog-prebuilt/bin/python"
cd "$REPO_ROOT"
"$EVAL_PYTHON" -m unittest discover -s tests -p 'test_assignment2*.py' -q
"$EVAL_PYTHON" -m assignment2.preflight
echo ASSIGNMENT2_OFFLINE_DIAGNOSTIC_OK
