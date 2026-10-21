# Assignment 2: BabyAI 64-step and Crafter 128-step evaluation

Both models and all three agents retain the existing final seeds and all 138
final episodes (69 per model, including nine variability episodes per model).
Skip development and pilot. These are shorter-horizon evaluations; do not claim
reproduction of longer-horizon benchmark scores.

The manifest uses 4096 input tokens including all instructions, observations,
history, state and chat framing, 256 output tokens, at most two total attempts,
zero SDK retries, and reflection before every fifth actor decision (every four
completed steps). Both models use the same limits. Oldest history is removed
before submission; game instructions, the complete current observation and
required state/response-format instructions remain intact. Actor/lesson replies
are explicitly concise. Prompt bounds and trimming counts are logged per call.

## Budget and integration checks

The cumulative ledger is `/data/s4822285/assignment2/results/api_spending_ledger.json`.
The ceiling is $15 including historical charges and unresolved reservations.
Official GPT-4o mini standard prices checked 2026-10-21 are $0.15/M input and
$0.60/M output: https://developers.openai.com/api/docs/models/gpt-4o-mini . The
snapshot is `gpt-4o-mini-2024-07-18`. It is marked deprecated in the current docs;
account access is unverified until the user runs the integration checks.
No cached-input discount is assumed. Local Qwen requests do not spend API money.

Six short smoke episodes per model use three steps for naive/planning and five
for reflective planning. This exercises the critic and a following actor. The
maximum OpenAI logical calls are 24 for smoke and 6841 for final, including all
variability actors and critics. With two full-cost attempts per call:

`(24 + 6841) * 2 * (4096 * 0.15 + 256 * 0.60) / 1e6 = $10.544640`.

Add actual historical spending and outstanding reservations. The preflight
requires the entire remaining paid plan to fit while leaving $1 for recovery.
No additional paid diagnostic is configured. Every new or resumed paid stage
rechecks the real ledger before sending requests; smoke also reserves capacity
for the complete final. These are conservative token-cost estimates, not an
account billing reconciliation.

The explicit migration CLI has no arguments:

```bash
# Inside an allocation only; never Python on calcium.
export ASSIGNMENT2_BUDGET_LEDGER_PATH=/data/s4822285/assignment2/results/api_spending_ledger.json
/data/s4822285/assignment2/envs/balrog-prebuilt/bin/python -m assignment2.migrate_budget
```

It locks the ledger, backs up exact original bytes with a timestamp, and preserves
all historical calls, charges and reservations without repricing them. It accepts
reviewed schema-1 $10/$15/$60 ledgers, changes only ceiling/policy plus migration
metadata, and refuses commitments above $15. It never clears an existing lock.

## Deployment and allocated diagnostics

On the Mac every connection uses the existing shared socket:

```bash
ssh -S "/tmp/ass2-$(id -un).sock" -o BatchMode=yes l40s
```

Server root: `/data/s4822285/assignment2`; repository: `$ASS2_ROOT/repo`.
Evaluation Python: `$ASS2_ROOT/envs/balrog-prebuilt/bin/python`.
Serving Python: `$ASS2_ROOT/envs/vllm-prebuilt/bin/python`.
Keep package/model/temp caches under this project root. The serving bin directory
must be on PATH for ninja. Setup uses prebuilt Python 3.11.11 and evaluation
setuptools 80.9.0. Do not rerun a full setup over working environments.

Use a reviewed file allowlist for deployment, excluding SECRETS, runtime data,
environments, caches and unrelated files. Never use whole-repository rsync or
`--delete`. Preserve the three original server edits and their scoped diff outside
the repository. The deployed receipt `assignment2/deployed_version.json` contains
the frozen local source commit and allowlisted hashes. The runner verifies them
and records actual server Git HEAD separately. Keep this version fixed through
checks, final and resume; incompatible source/settings/packages/snapshots fail.

Submit diagnostic only after source hash verification:

```bash
export ASS2_ROOT=/data/s4822285/assignment2
cd "$ASS2_ROOT/repo"
sbatch --chdir="$ASS2_ROOT/repo" --output="$ASS2_ROOT/logs/diagnostic-%j.out" \
  --error="$ASS2_ROOT/logs/diagnostic-%j.err" assignment2/scripts/diagnostic.sbatch
```

All Python, dependency/GPU checks and model serving run inside allocations.
The diagnostic imports both environments, resets/closes them without actions,
checks real reset-observation prompts with offline responses, runs offline tests,
audits historical records, validates/migrates the ledger, checks cached shards,
and starts offline vLLM. It requests only GET /v1/models, no inference or paid
requests. It installs only missing matplotlib. Require the diagnostic success
marker, exit marker and status file; a historical setup marker alone is insufficient.

## User launch commands

No experiment is launched by deployment. From the gateway after diagnostics,
set the key privately in this submitting shell (never display or paste its value
into chat). A 5 GiB visible-filesystem guard below is not a personal quota claim;
check any exposed quota separately. Shared storage can fill between checks.
The experiment resources remain one L40S GPU, four CPUs, 32G and 12 hours.
Cold serving imports/startup were slow on this machine. The launcher allows
a 30-minute readiness deadline, sleeping two seconds between checks; each HTTP
check also has a five-second timeout (at most one polling cycle beyond the deadline).
Startup time is deducted from the same
12-hour allocation before reserving cleanup time. The full diagnostic allows
one hour; it is separate from experiment jobs.

```bash
set -euo pipefail
export ASS2_ROOT=/data/s4822285/assignment2
export ASS2_MIN_FREE_GB=5 ASS2_CLEANUP_RESERVE_MINUTES=30
: "${OPENAI_API_KEY:?Set privately in this submitting shell first.}"
export OPENAI_API_KEY
unset ASS2_RUN_DIRECTORY ASS2_RESUME_DIRECTORY
cd "$ASS2_ROOT/repo"
submit() {
  export ASS2_PHASE="$1" ASS2_MODELS="$2"
  shift 2
  sbatch --parsable --chdir="$ASS2_ROOT/repo" \
    --output="$ASS2_ROOT/logs/run-%j.out" --error="$ASS2_ROOT/logs/run-%j.err" \
    "$@" assignment2/scripts/run_pipeline.sbatch
}
OPENAI_CHECK=$(submit smoke openai_mini); OPENAI_CHECK=${OPENAI_CHECK%%;*}
QWEN_CHECK=$(submit smoke qwen_7b_local); QWEN_CHECK=${QWEN_CHECK%%;*}
OPENAI_FINAL=$(submit final openai_mini --dependency="afterok:$OPENAI_CHECK")
QWEN_FINAL=$(submit final qwen_7b_local --dependency="afterok:$QWEN_CHECK")
OPENAI_FINAL=${OPENAI_FINAL%%;*}; QWEN_FINAL=${QWEN_FINAL%%;*}
printf 'checks=%s,%s finals=%s,%s\n' "$OPENAI_CHECK" "$QWEN_CHECK" "$OPENAI_FINAL" "$QWEN_FINAL"
```

Each final depends only on its own model's validated check. Qwen is independent
of any OpenAI failure or paid-budget exhaustion. The launcher requires one model
per job. Validated expected counts are 6, 6, 69 and 69; final totals 138. Game
failure may still be valid evidence, while invalid infrastructure/usage/trace or
missing episode identities fail the stage. Each complete final runs its own
analysis. Cross-model analysis accepts both final directories inside an allocation.

## Monitor, resume and retrieve

No sacct: accounting storage is disabled. Gateway monitoring uses:

```bash
squeue -u s4822285 -o '%.18i %.20j %.10T %.12M %.20R'
scontrol show job "$OPENAI_FINAL"  # Only while retained by Slurm.
tail -f "$ASS2_ROOT/logs/run-$OPENAI_FINAL.out"
cat "$ASS2_ROOT/logs/run-$OPENAI_FINAL.status"
rg 'ASSIGNMENT2_PHASE_OK|ASSIGNMENT2_JOB_EXIT' "$ASS2_ROOT/logs/run-$OPENAI_FINAL.out"
```

Require the phase marker, validated count and exit_code=0; disappearance from
squeue alone is not proof. Run directories are `results/PHASE_slurm-JOB_ID`.

Resume the same directory/model/settings manually, only after the original job
has stopped. Completed compatible episodes are skipped; partial attempts and
charges remain. The current environment cannot restore a game mid-episode, so
an incomplete episode restarts once on explicit resume. At most two attempts per
episode across the directory are permitted; there is no automatic episode retry
loop. The paid preflight includes full remaining episodes plus all spent/reserved
money and retains $1 headroom. Review additional recovery manually if the limit
is reached. Never change fingerprints to make old results look compatible.

```bash
export ASS2_RESUME_DIRECTORY="$ASS2_ROOT/results/final_slurm-$OPENAI_FINAL"
RESUMED=$(submit final openai_mini)
unset ASS2_RESUME_DIRECTORY
printf '%s\n' "$RESUMED"
```

For Qwen use its original job ID and `qwen_7b_local`; for smoke use `smoke` and
its original model. No paid-ledger lock is needed by Qwen. Timeout saves partial
evidence and exits nonzero, with 30 minutes plus a 120-second termination grace
reserved. SIGKILL can leave a lock. Inspect its recorded host/PID/Slurm owner
before recovery; never clear active locks. Resuming a failed check does not rewire
queued dependent jobs; resubmit only the affected final with the resumed check ID.

Combined analysis, submitted by the user after both finals pass:

```bash
sbatch --partition=L40s_students --gres=gpu:nvidia_l40s:1 --cpus-per-task=4 \
  --mem=32G --time=00:30:00 --chdir="$ASS2_ROOT/repo" \
  --output="$ASS2_ROOT/logs/analysis-%j.out" --error="$ASS2_ROOT/logs/analysis-%j.err" \
  --wrap="$ASS2_ROOT/envs/balrog-prebuilt/bin/python -m assignment2.analysis $ASS2_ROOT/results/final_slurm-$OPENAI_FINAL $ASS2_ROOT/results/final_slurm-$QWEN_FINAL --phase final --output $ASS2_ROOT/results/final-analysis"
```

From the Mac, after jobs finish:

```bash
cd '/Users/rwikkamilya/MSc/Sem 3/ALLM/agentic-llm-assignment-2'
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" \
  l40s:/data/s4822285/assignment2/results/ ./assignment2-results/
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" \
  l40s:/data/s4822285/assignment2/logs/ ./assignment2-logs/
rsync -av -e "ssh -S /tmp/ass2-$(id -un).sock -o BatchMode=yes" \
  l40s:/data/s4822285/assignment2/deployment/ ./assignment2-deployment/
```

Preserve raw results, manifests, validation, ledger/backups, deployment receipts,
package evidence and logs. No synthetic diagnostic response is experiment evidence.
Complete reports from real results and disclose AI assistance under course policy.
