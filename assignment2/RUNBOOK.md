# Assignment 2: OpenAI + local Qwen execution

Use the current deployment receipt and allocated diagnostic logs as evidence.
Historical preparation reports do not establish current server readiness. Never run Python, dependency checks, GPU checks, model serving, or
benchmark episodes on calcium. This task does not authorize experiment
submission or paid requests; the commands below are for the user's later use.

## Before any submission

The target is `/data/s4822285/assignment2/repo`, partition `L40s_students`,
resource `gpu:nvidia_l40s:1`, four CPUs, 32G RAM, and a 12-hour experiment limit.
Historical setup claims must be checked against current logs and allocations.

From the Mac, every SSH command must use the existing shared socket:

```bash
ssh -S "/tmp/ass2-$(id -un).sock" -o BatchMode=yes l40s
```

On the gateway, use only filesystem/Git/Slurm checks:

```bash
export ASS2_ROOT=/data/s4822285/assignment2
cd "$ASS2_ROOT/repo"
bash assignment2/scripts/inspect_remote.sh
```

The inspection leaves Git refs unchanged and prints no credential values. Confirm
that `hostname -f` identifies calcium. Compare all source/config hashes with the
prepared local copy. Inspect existing jobs before editing source used by them.
Do not deploy while an existing job is using this checkout.

Before reconciling server edits, back up only the three reviewed modified files
and their scoped diff outside the repository. Do not copy SECRETS or bundle all
untracked files. Preserve the timestamped pre-deployment backup.

If the manifest is zero bytes, first inspect backup names, sizes, hashes, and
`git log -- assignment2/config/experiment.yaml`. Compare a recovered candidate
from the verified Git commit with the local manifest. Preserve the empty file
and all backups. Recover only after this comparison; do not blindly pull,
reset, clean, or use `rsync --delete`. Conflicting server edits need explicit
reconciliation before a scoped patch can pass `git apply --check`. Preserve
SECRETS, environments, caches, logs, results and ledger in place. Exclude
SECRETS from source transfers and submission archives.

Record the local source commit, server checkout SHA and deployed file hashes.
`assignment2/deployed_version.json` contains the source commit and allowlisted
hashes. The runner verifies these hashes before using that source commit and
records the actual checkout SHA separately; a stale Git HEAD is never relabeled. Leave that execution version unchanged through
smoke, pilot, final and resumes. Resume rejects changed source, package versions,
model snapshot, model selection, settings or expected episode identities.

## Allocated diagnostics and ledger migration

All Python commands in this section belong inside a Slurm allocation. The
prepared diagnostic job runs imports, package/cache/GPU checks, offline tests,
and saved-result summaries, explicitly migrates the reviewed historical ledger,
and checks offline Qwen startup using GET /v1/models only. It resets/closes
both benchmark environments without taking actions. It performs no inference
or paid requests. It installs matplotlib only if that dependency is absent. Submit only after its source has been deployed:

```bash
export ASS2_ROOT=/data/s4822285/assignment2
mkdir -p "$ASS2_ROOT/logs"
sbatch --chdir="$ASS2_ROOT/repo" \
  --output="$ASS2_ROOT/logs/diagnostic-%j.out" \
  --error="$ASS2_ROOT/logs/diagnostic-%j.err" \
  assignment2/scripts/diagnostic.sbatch
```

Require `ASSIGNMENT2_DIAGNOSTIC_OK`, `ASSIGNMENT2_DIAGNOSTIC_EXIT exit_code=0`,
the diagnostic status file, and actual quota
headroom. Shared-filesystem free space is not an individual quota guarantee.
The diagnostic requires both profiles' prerequisites because both are planned.
A cloud-only experiment launcher requires only the evaluation environment;
it does not require vLLM, a Qwen snapshot, or a model server.

Setup remains in `assignment2/scripts/setup.sbatch`; use it only if environments
are missing. It installs prebuilt Python 3.11.11 and separate
`envs/balrog-prebuilt` and `envs/vllm-prebuilt`,
prepares the Qwen cache, installs plotting support, and records package versions.
Do not rerun setup blindly over a prepared environment. Repair a missing
package in an allocation, then repeat diagnostics and freeze versions again.

The cumulative paid ledger is:
`/data/s4822285/assignment2/results/api_spending_ledger.json`.
The configured ceiling is $60. A $10 historical ledger is deliberately rejected
until explicitly migrated. From an existing interactive allocation:

```bash
cd /data/s4822285/assignment2/repo
export ASSIGNMENT2_BUDGET_LEDGER_PATH=/data/s4822285/assignment2/results/api_spending_ledger.json
/data/s4822285/assignment2/envs/balrog-prebuilt/bin/python -m assignment2.migrate_budget
```

The migration locks the same ledger as evaluation, writes an exclusive,
timestamped backup of its original bytes, and preserves every historical call,
charge and outstanding reservation. The new OpenAI prices apply only to new
calls. It refuses any existing lock and never resets spending. A missing ledger
needs no migration; the first paid run creates it under the paid-run lock.
Inspect its prior spend/reservations before concluding the sequence fits $60.

## Four dependent experiment stages (user launches later)

Keep `OPENAI_API_KEY` in the submitting environment without logging its value.
Never source an unreviewed SECRETS file or enable shell tracing. Set the
quota-aware minimum free-space threshold after checking quota. The following
commands require these two values to have been set already:

```bash
set -e
export ASS2_ROOT=/data/s4822285/assignment2
export ASS2_CLEANUP_RESERVE_MINUTES=30
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY privately in this submitting shell.}"
: "${ASS2_MIN_FREE_GB:?Set a positive integer based on confirmed quota headroom.}"
export OPENAI_API_KEY ASS2_MIN_FREE_GB
unset ASS2_RUN_DIRECTORY ASS2_RESUME_DIRECTORY
cd "$ASS2_ROOT/repo"
mkdir -p "$ASS2_ROOT/logs"

export ASS2_PHASE=smoke ASS2_MODELS=openai_mini
SMOKE_OPENAI=$(sbatch --parsable --chdir="$ASS2_ROOT/repo" \
  --output="$ASS2_ROOT/logs/run-%j.out" --error="$ASS2_ROOT/logs/run-%j.err" \
  assignment2/scripts/run_pipeline.sbatch)
SMOKE_OPENAI=${SMOKE_OPENAI%%;*}

export ASS2_PHASE=smoke ASS2_MODELS=qwen_7b_local
SMOKE_QWEN=$(sbatch --parsable --dependency="afterok:$SMOKE_OPENAI" \
  --chdir="$ASS2_ROOT/repo" --output="$ASS2_ROOT/logs/run-%j.out" \
  --error="$ASS2_ROOT/logs/run-%j.err" assignment2/scripts/run_pipeline.sbatch)
SMOKE_QWEN=${SMOKE_QWEN%%;*}

export ASS2_PHASE=pilot ASS2_MODELS='openai_mini qwen_7b_local'
PILOT=$(sbatch --parsable --dependency="afterok:$SMOKE_QWEN" \
  --chdir="$ASS2_ROOT/repo" --output="$ASS2_ROOT/logs/run-%j.out" \
  --error="$ASS2_ROOT/logs/run-%j.err" assignment2/scripts/run_pipeline.sbatch)
PILOT=${PILOT%%;*}

export ASS2_PHASE=final ASS2_MODELS='openai_mini qwen_7b_local'
FINAL=$(sbatch --parsable --dependency="afterok:$PILOT" \
  --chdir="$ASS2_ROOT/repo" --output="$ASS2_ROOT/logs/run-%j.out" \
  --error="$ASS2_ROOT/logs/run-%j.err" assignment2/scripts/run_pipeline.sbatch)
FINAL=${FINAL%%;*}
printf 'OpenAI smoke=%s Qwen smoke=%s pilot=%s final=%s\n' \
  "$SMOKE_OPENAI" "$SMOKE_QWEN" "$PILOT" "$FINAL"
```

Use a shell with `set -e` so a failed submission stops the sequence. Each job
captures its own exported phase/model values at submission time. The expected
validated completions are 6, 6, 48 and 138 respectively. Final contains 120
main episodes and 18 variability episodes; variability is analyzed separately.
An incomplete, corrupt, misidentified, duplicate or infrastructure-invalid
record makes the stage fail. `afterok` therefore blocks downstream stages.
Final combined analysis runs only after the full identity validation succeeds.

Smoke has 32 actor steps per episode. Pilot/final use BabyAI 128 and Crafter
250; both models share the agent prompts and limits. Report Crafter as a
**250-step evaluation, not a reproduction of the original 1000-step setting**.
One worker runs episodes sequentially. A failed game is still valid evidence
if the infrastructure checks pass. Invalid actions/parser failures are logged;
a wholly unusable actor or absent expected critic/usage/trace fails validation.

## Monitor and resume

Gateway commands (no Python). Slurm accounting storage is disabled, so do not
rely on sacct. Require both the phase marker and exit_code=0; a disappeared job
alone does not establish success:

```bash
squeue -u s4822285 -o '%.18i %.20j %.10T %.12M %.20R'
scontrol show job "$FINAL"  # Available only while Slurm retains the job.
cat "$ASS2_ROOT/logs/run-$FINAL.status"
rg 'ASSIGNMENT2_PHASE_OK|ASSIGNMENT2_JOB_EXIT' "$ASS2_ROOT/logs/run-$FINAL.out"
tail -f "$ASS2_ROOT/logs/run-$FINAL.out"
```

New directories are `results/PHASE_slurm-JOB_ID`. Keep the same phase and model
list on resume. For an interrupted final job, with `FINAL` set to its original
job ID:

```bash
export ASS2_PHASE=final ASS2_MODELS='openai_mini qwen_7b_local'
export ASS2_RESUME_DIRECTORY="$ASS2_ROOT/results/final_slurm-$FINAL"
unset ASS2_RUN_DIRECTORY
RESUMED=$(sbatch --parsable --chdir="$ASS2_ROOT/repo" \
  --output="$ASS2_ROOT/logs/resume-%j.out" --error="$ASS2_ROOT/logs/resume-%j.err" \
  assignment2/scripts/run_pipeline.sbatch)
printf '%s\n' "$RESUMED"
```

For smoke/pilot use their phase, original model list and directory instead.
Compatible completed episodes are skipped; incomplete attempts are preserved
and retried under a new attempt filename. No ledger balance or unresolved
reservation is released by resume. Timeout/SIGTERM saves partial evidence and
returns nonzero. Cleanup reserves 30 minutes plus a 120-second termination grace.
A SIGKILL may leave a lock; inspect `api_spending_ledger.lock/owner`, the owning
Slurm job, and the recorded host/PID before any manual recovery. Never clear an
active lock. Run directories also have their own execution lock.

If a prerequisite stage fails, already queued dependent jobs may remain blocked
on its old ID; resuming it does not automatically rewire them. Inspect/cancel
only your blocked dependent jobs and resubmit with `afterok:RESUMED_JOB_ID`.
Do not cancel a running job as part of recovery.

## Revalidate, analyze, retrieve

Inside an allocation, to independently revalidate before analysis:

```bash
cd "$ASS2_ROOT/repo"
/data/s4822285/assignment2/envs/balrog-prebuilt/bin/python -m assignment2.analysis \
  "$ASS2_ROOT/results/final_slurm-$FINAL" --phase final \
  --output "$ASS2_ROOT/results/final_slurm-$FINAL/analysis"
```

For split final model jobs, pass both complete run directories. The analysis
rejects synthetic data, phase mixing, duplicate completed identities, and
incompatible source/prompt settings. Do not combine pilot and final evidence.

From the local checkout after the jobs finish:

```bash
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" l40s:/data/s4822285/assignment2/results/ ./assignment2-results/
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" l40s:/data/s4822285/assignment2/logs/ ./assignment2-logs/
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" l40s:/data/s4822285/assignment2/setup/ ./assignment2-setup-evidence/
```

No deletion flags are used. Preserve raw records, the run manifest, validation,
ledger backups, package versions, cached snapshot identity and Slurm logs.
Complete the report and archives only with actual evidence and an honest
AI-assistance disclosure. Personal reflections must be written by the student.
