# Assignment 2: final corrected campaign

**Completed-run handoff:** see [report_materials/PARTNER_GUIDE.md](report_materials/PARTNER_GUIDE.md)
for verified results and [evidence/README.md](evidence/README.md) for offline audit
and packaging commands. Both model jobs completed. The original combined job
failed only during ZIP creation (a pre-1980 receipt timestamp); its early COMPLETE
marker did not mean packaging succeeded. Repaired ZIPs are stored separately under
`submission_packages/`. The launch instructions below document the executed
workflow; do not submit another experiment just to write the report.

This is protocol `outcome_feedback_v2`. Read `FINAL_EXPERIMENT.md` for the research
rationale and report guidance. The earlier experiment is preserved. Do not mix
its results with this protocol or overwrite its deployment.

## What the launch does

One submission starts two independent 12-hour evaluation jobs, one per model.
Each job runs the offline regression tests and real-game reset/prompt checks,
then six small smoke episodes, then its 69 final episodes. Qwen starts its cached
server once. A small CPU analysis job depends on successful completion of both
model jobs and creates the combined tables, figures, trace extracts and `COMPLETE`
marker, and packages/Ass2-1.zip through Ass2-3.zip. The student report still needs to be added after review. No manual smoke-to-final decision or prompt tuning is needed.

The smoke gate checks infrastructure, recording and accounting. Winning the game
is not a gate. Empty/invalid model replies are retained as behavioral failures;
API errors, missing usage, inconsistent records or incorrect resets stop the job.

Existing environments and cached weights are reused. Do not rerun setup or the
old diagnostic workflow. All Python, model inference and analysis on LIACS run
inside Slurm allocations; calcium is used only for transfer and scheduling.

## Deploy from your Mac

Run from the repository. The builder needs only standard Python and Git. It
copies an explicit set of source, configuration, tests and documentation, excludes
results/secrets/caches, and creates an archive plus SHA256 sidecar. The release
receipt records the base Git commit, whether the working tree was dirty, and exact
file hashes. The content hash, not a claim that uncommitted code is committed,
identifies this release.

```bash
cd '/Users/rwikkamilya/MSc/Sem 3/ALLM/agentic-llm-assignment-2'
python3 -m assignment2.prepare_deployment
```

Use the exact release name printed above (or the specific name in the handoff).
The existing SSH configuration uses `l40s` for calcium via the LIACS gateway.
If the shared SSH socket has expired, run `ssh l40s` normally and authenticate;
then omit the `-o ControlPath=...` options in the commands below.

```bash
RELEASE=ass2-v2-REPLACE_WITH_PRINTED_HASH
ASS2_SOCKET="/tmp/ass2-$(id -un).sock"
ssh -o ControlPath="$ASS2_SOCKET" l40s 'mkdir -p /data/s4822285/assignment2/releases'
scp -o ControlPath="$ASS2_SOCKET" \
  "assignment2/deployment/$RELEASE.tar.gz" \
  "assignment2/deployment/$RELEASE.tar.gz.sha256" \
  l40s:/data/s4822285/assignment2/releases/
ssh -o ControlPath="$ASS2_SOCKET" l40s
```

On calcium, paste the entire block below. Strict settings stay inside the
parenthesized subshell: enabling `set -u` in calcium's interactive login shell
can make its prompt hook fail with `PROMPT_START: unbound variable` and disconnect
SSH. This is a shell-integration issue, not an experiment failure. The frozen
bundle remains usable; this correction does not require rebuilding it.

```bash
RELEASE=ass2-v2-REPLACE_WITH_PRINTED_HASH
(
set -euo pipefail
cd /data/s4822285/assignment2/releases
sha256sum -c "$RELEASE.tar.gz.sha256"
if [[ ! -e "$RELEASE" ]]; then
  tar -xzf "$RELEASE.tar.gz"
fi
cd "$RELEASE"
# If already exported in this shell, keep the existing key. Otherwise enter it privately:
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp 'OpenAI API key: ' OPENAI_API_KEY
  printf '\n'
fi
export OPENAI_API_KEY
unset ASS2_RUN_DIRECTORY ASS2_RESUME_DIRECTORY
bash assignment2/scripts/submit_final.sh
)
```

The submitter refuses to launch the same release twice. If submission stops after
one job is scheduled, read `submission.txt` and `squeue` before doing anything
else; do not delete the campaign directory and resubmit blindly. A failed job
leaves all its records and charges intact. It cannot silently replace a failed
phase with zero scores.

## Storage, spending and run counts

- Release: `/data/s4822285/assignment2/releases/ass2-v2-HASH`.
- New results: `/data/s4822285/assignment2/results/ass2-v2-HASH`.
- Existing cumulative ledger: `/data/s4822285/assignment2/results/api_spending_ledger.json`.
- Evaluation Python: `/data/s4822285/assignment2/envs/balrog-prebuilt/bin/python`.
- Serving Python: `/data/s4822285/assignment2/envs/vllm-prebuilt/bin/python`.
- Cached Qwen weights: the existing Hugging Face cache under the project root.
- Logs: `/data/s4822285/assignment2/logs/v2-{qwen,openai,analysis}-JOB_ID.{out,err}`.

The new output cap is 192 tokens and the conservative input bound is 4096 tokens
(UTF-8 bytes plus framing). The cloud job automatically backs up and migrates the
existing ledger's reservation policy; it preserves historical calls, spending
and unresolved reservations. It refuses to create a fresh allowance if that
ledger is absent. At most two request attempts, no SDK retries, no automatic
episode retries. A manual episode restart is limited to one extra attempt.

There are 24 maximum logical OpenAI smoke calls and 8505 final calls, including
critics and repeats. With two maximum-cost attempts each, the new cost bound is:

`8529 × 2 × (4096 × 0.15 + 192 × 0.60) / 1,000,000 = $12.4455168`.

Using the previously inspected historical estimate of $0.9520108 and no unresolved
reservations gives a cumulative bound of $13.3975276. That leaves $1.6024724,
including the required $1 recovery reserve, below the $15 ceiling. This is a
conservative estimate, not the provider's invoice. Every paid stage rechecks the
live ledger and stops before requests if the remaining plan no longer fits.

Final counts per model: 45 main episodes (30 BabyAI + 15 Crafter), plus 24 extra
samples of four existing settings, total 69. Across two models: 90 main + 48 repeat
observations = 138 final episodes. Smoke adds 12 episodes, kept separate.

A 5 GiB free-space check is a filesystem guard, not proof of individual quota.
Startup is limited to 30 minutes and charged against the same 12-hour job window.
The runner reserves 30 minutes plus termination grace for saving and cleanup.
There are no new dependency installs or model downloads in this workflow.

## Monitor and retrieve

```bash
squeue -u s4822285 -o '%.18i %.24j %.10T %.12M %.30R'
cat /data/s4822285/assignment2/results/$RELEASE/submission.txt
# Replace JOB_ID with the corresponding ID printed by submit_final.sh:
tail -f /data/s4822285/assignment2/logs/v2-openai-JOB_ID.out
cat /data/s4822285/assignment2/logs/run-JOB_ID.status
```

For success require `ASSIGNMENT2_PHASE_OK`, exit code 0, and 69/69 final identities
for each model. The cross-model analysis writes `COMPLETE` only after both final
directories validate and initial paired observations match. Disappearing from
`squeue` alone is not success. `sacct` was unavailable on this cluster.

From the Mac after completion:

```bash
mkdir -p assignment2/results
scp -r -o ControlPath="$ASS2_SOCKET" \
  "l40s:/data/s4822285/assignment2/results/$RELEASE" assignment2/results/
```

Read `analysis/performance_matrix.csv`, `task_breakdown.csv`,
`paired_differences.csv`, `repeated_run_variability.csv`, and `selected_traces.md`.
Exact requests/responses, transitions, source hashes, configurations and usage are
in each model's `final/episodes/*.json`; do not submit only the summary CSV.

## Recovery only if infrastructure fails

Do not edit code, prompts, seeds, limits or package versions within this release.
Keep completed game failures. Review the error, wait for the original job to stop,
then explicitly resume the same phase/model directory in an allocation. A resume
starts an incomplete episode over; it cannot restore a game mid-episode. It
preserves both attempts and all charges. It will fail if source/configuration or
budget is incompatible. After a recovery, the canceled dependent analysis job
must be submitted again with the repaired job dependencies.

For example, from the same frozen release on calcium:

```bash
export ASS2_ROOT=/data/s4822285/assignment2
cd "$ASS2_ROOT/releases/$RELEASE"
export ASS2_REPO_ROOT="$PWD"
export ASS2_CAMPAIGN_ROOT="$ASS2_ROOT/results/$RELEASE"
export ASS2_MODELS=openai_mini ASS2_PHASE=final ASS2_MIN_FREE_GB=5
export ASS2_RESUME_DIRECTORY="$ASS2_CAMPAIGN_ROOT/openai_mini/final"
sbatch --chdir="$PWD" --export=ALL \
  --output="$ASS2_ROOT/logs/v2-recovery-%j.out" --error="$ASS2_ROOT/logs/v2-recovery-%j.err" \
  assignment2/scripts/run_pipeline.sbatch
```

This is a recovery command, not part of the normal one-submission workflow.
