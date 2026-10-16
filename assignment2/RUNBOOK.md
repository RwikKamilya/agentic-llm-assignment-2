# Assignment 2 LIACS deployment runbook

This document prepares Slurm execution only. No SSH connection, submission,
package installation, model download, API request, or benchmark episode was
made while updating it.

## Confirmed deployment facts

- `calcium` is a gateway/head node: do not run setup, a model server, Python
  evaluation, or benchmarks there.
- Use partition `L40s_students` with one node, task, and `gpu:nvidia_l40s:1`.
  The accepted starting request is 4 CPUs, 32G RAM, and 12 hours. No account
  option is required.
- An allocation on `ceratanium` exposed one NVIDIA L40S with 46068 MiB and
  driver 615.71.09. The CUDA version displayed by `nvidia-smi` is not treated
  as an installed CUDA toolkit.
- Only Python 3.9.25 was on `PATH`; no usable Python module, conda, or uv was
  found. Setup therefore builds an isolated Python 3.11.11 below the project
  root. Current vLLM GPU documentation supports Python 3.11–3.14 and ships
  precompiled CUDA binaries, so setup does not assume a system CUDA toolkit.
  [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- `/home`, `/local`, and `/tmp` are too small for environments or model data.
  Use `/data/s4822285/assignment2`, while treating its write permission,
  quota, and retention as checks still required before setup.

The repository location is fixed as:

```text
/data/s4822285/assignment2/repo
```

Both batch scripts require this environment value:

```bash
export ASS2_ROOT=/data/s4822285/assignment2
```

## 1. One-time preflight on the gateway

These are filesystem checks only; do not create environments or run Python on
the gateway.

```bash
ssh l40s
export ASS2_ROOT=/data/s4822285/assignment2
test -d "$ASS2_ROOT/repo" && test -f "$ASS2_ROOT/repo/assignment2/scripts/setup.sbatch"
test -w "$ASS2_ROOT"
mkdir -p "$ASS2_ROOT/logs"
df -h "$ASS2_ROOT"
```

Confirm the project-specific quota before choosing the free-space guard below.
The shared `/data` free-space figure is not a quota guarantee.

```bash
export ASS2_MIN_FREE_GB=REPLACE_WITH_CONFIRMED_QUOTA_HEADROOM
```

The scripts intentionally refuse to run until this value is a positive
integer. It is a guardrail, not a claim that the user has that quota.

## 2. Submit setup only

`assignment2/scripts/setup.sbatch` has no `--account`, `--output`, `--error`,
or `--chdir` directives. Supply those at submission time. It requests one
allocated L40S, builds Python 3.11.11 under `/data/s4822285/assignment2`, then
creates separate `envs/balrog` and `envs/vllm` environments.

It sends build files, temporary files, pip files, Hugging Face data, Torch,
Triton, and vLLM caches below the project root. It installs BALROG and its
assets in the evaluation environment, installs vLLM in the serving environment,
downloads the configured Qwen model snapshot, checks imports and GPU
visibility, and saves package manifests below `setup/setup-JOB_ID/`.

It does **not** use `GEMINI_API_KEY`, call Gemini, start vLLM, or run a
benchmark episode.

With the log paths chosen here as an example, submit:

```bash
sbatch \
  --chdir=/data/s4822285/assignment2/repo \
  --output=/data/s4822285/assignment2/logs/setup-%j.out \
  --error=/data/s4822285/assignment2/logs/setup-%j.err \
  assignment2/scripts/setup.sbatch
```

Record the submitted job ID as `SETUP_JOB_ID`. Monitor without running work on
the gateway:

```bash
squeue -j SETUP_JOB_ID
sacct -j SETUP_JOB_ID --format=JobID,State,Elapsed,ExitCode
```

After Slurm reports completion, setup is successful only if both checks pass:

```bash
grep -Fx ASSIGNMENT2_SETUP_OK \
  /data/s4822285/assignment2/logs/setup-SETUP_JOB_ID.out
test -s /data/s4822285/assignment2/setup/setup-SETUP_JOB_ID/model-id.txt
```

If setup fails, inspect the matching `.err` and `.out` files. Do not attempt
to repair the system Python, install system packages, or move environments to
`/home`.

## 3. Submit an experiment job

`assignment2/scripts/run_pipeline.sbatch` has the same verified resource
request and likewise has no account/log/chdir directives. It uses only the
setup environments and persistent paths below `ASS2_ROOT`. At runtime it
checks `timeout`, free space, the installed environments, and the already
prepared local model snapshot before starting vLLM on `127.0.0.1` inside the
allocated job.

Use a 30-minute cleanup reserve unless you intentionally set another positive
value. The launcher subtracts both that reserve and a two-minute termination
grace period from the 12-hour limit before evaluation begins.

```bash
export ASS2_CLEANUP_RESERVE_MINUTES=30
export ASS2_PHASE=smoke
export ASS2_MODELS="qwen_7b_local"
unset ASS2_RESUME_DIRECTORY
sbatch \
  --chdir=/data/s4822285/assignment2/repo \
  --output=/data/s4822285/assignment2/logs/run-%j.out \
  --error=/data/s4822285/assignment2/logs/run-%j.err \
  assignment2/scripts/run_pipeline.sbatch
```

For a Gemini phase, set `GEMINI_API_KEY` only in the submitting shell, without
printing it or enabling shell tracing. The launcher never writes that variable
to a log. Do not submit two Gemini-containing jobs concurrently: it uses an
exclusive budget lock to protect the shared spending ledger.

The persistent ledger is:

```text
/data/s4822285/assignment2/results/api_spending_ledger.json
```

It has a USD 10 ceiling from `assignment2/config/experiment.yaml`. Reservations
are saved before each Gemini request and include retries. Consequently smoke,
pilot, final, and resumed runs all consume one cumulative budget; a terminated
job's unresolved reservation remains charged rather than being reset.

Monitor a run with:

```bash
squeue -j RUN_JOB_ID
sacct -j RUN_JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -f /data/s4822285/assignment2/logs/run-RUN_JOB_ID.out
```

## 4. Find and resume an interrupted run

For a new job, the result directory is deterministic:

```text
/data/s4822285/assignment2/results/PHASE_slurm-RUN_JOB_ID
```

List completed or partial directories without changing them:

```bash
find /data/s4822285/assignment2/results -maxdepth 1 -mindepth 1 -type d -name 'final_slurm-*' -print
```

Resume the exact selected directory via a new Slurm job. Keep the same phase
and model list so only compatible completed episodes are skipped. The global
USD 10 ledger is reused automatically; do not delete or replace it.

```bash
export ASS2_CLEANUP_RESERVE_MINUTES=30
export ASS2_PHASE=final
export ASS2_MODELS="gemini_flash_lite qwen_7b_local"
export ASS2_RESUME_DIRECTORY=/data/s4822285/assignment2/results/final_slurm-RUN_JOB_ID
sbatch \
  --chdir=/data/s4822285/assignment2/repo \
  --output=/data/s4822285/assignment2/logs/resume-%j.out \
  --error=/data/s4822285/assignment2/logs/resume-%j.err \
  assignment2/scripts/run_pipeline.sbatch
```

Do not remove `api_spending_ledger.json`. If a previous cloud job was killed
so abruptly that its budget-lock directory remains, inspect its recorded owner
and Slurm status before deciding how to clear it:

```bash
cat /data/s4822285/assignment2/results/api_spending_ledger.lock/owner
sacct -j PREVIOUS_JOB_ID --format=JobID,State,Elapsed,ExitCode
```

## 5. Final analysis and retrieve outputs

A final job containing both model profiles analyses its one result directory
before cleanup. If models were run separately, combine their two directories
inside a Slurm allocation with `assignment2/analysis.py`; never combine smoke,
pilot, synthetic, or incomplete data as final evidence.

To retrieve persistent results and logs to the local machine after jobs finish:

```bash
rsync -av l40s:/data/s4822285/assignment2/results/ ./assignment2-results/
rsync -av l40s:/data/s4822285/assignment2/logs/ ./assignment2-logs/
```

## 6. Local review and commit

Run these locally, not on calcium:

```bash
git diff --check
python3 -m unittest discover -s tests -v
git add balrog/agents/__init__.py balrog/client.py balrog/config/config.yaml assignment2 tests
git commit -m "Prepare Assignment 2 L40S deployment"
```

Do not claim setup or experiments succeeded until their Slurm logs and
persisted records provide the evidence.
