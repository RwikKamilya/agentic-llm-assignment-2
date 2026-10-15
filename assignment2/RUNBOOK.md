# Assignment 2 remote runbook

This runbook prepares future execution only. No API request, remote connection,
model download, model server, benchmark run, or scheduler submission was made
while preparing this repository.

The commands marked **not yet remotely validated** depend on the actual LIACS
allocation, scheduler, operating system, and installed modules. Do not run
inference on a login node or assume that a visible GPU belongs to you.

## 1. Review and push local changes

Run locally after reviewing the diff and tests:

```bash
git status
git diff --check
python3 -m unittest discover -s tests -v
git add balrog/agents/__init__.py balrog/config/config.yaml assignment2 tests
git commit -m "Prepare Assignment 2 planning and reflection experiments"
git push origin assignment-2
```

## 2. Connect and inspect resources (**not yet remotely validated**)

The existing alias is already defined by the user. Connect without changing SSH
configuration:

```bash
ssh l40s
```

On the remote host, first inspect the site and allocation context. This script
only reports information and discovers scheduler commands; it does not submit a
job:

```bash
cd /path/to/agentic-llm-assignment-2
bash assignment2/scripts/inspect_remote.sh
```

If the session is a login node or no allocated GPU is visible, stop there.
Ask the site documentation or administrator which scheduler/allocation command
to use. Only then choose a scheduler-specific interactive or batch workflow.

## 3. Install the project (**not yet remotely validated**)

Inside the allocated environment, use a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
balrog-post-install
python -m pip install matplotlib vllm
python -m unittest discover -s tests -v
```

`balrog-post-install` and vLLM/model setup may download required assets. Check
available disk quota first; do not install into a nearly full shared filesystem.

## 4. Configure secrets safely (**not yet remotely validated**)

Do not edit or commit `SECRETS`. In the allocated shell, supply the cloud key
only through the environment:

```bash
export GEMINI_API_KEY='paste-key-here'
```

Confirm that the variable is set without printing its value:

```bash
test -n "$GEMINI_API_KEY" && echo "GEMINI_API_KEY is set"
```

## 5. Start the local model endpoint (**not yet remotely validated**)

In a confirmed GPU allocation, start the planned 7B model:

```bash
source .venv/bin/activate
bash assignment2/scripts/start_vllm.sh
curl -fsS http://127.0.0.1:8080/v1/models
```

The vLLM script records one PID under `assignment2/server_logs/`. Stop that
specific process when finished:

```bash
bash assignment2/scripts/stop_vllm.sh
```

## 6. Smoke tests (**not yet remotely validated**)

First run one local-model episode per benchmark and agent condition. This calls
the local endpoint and writes a separate `smoke` result directory:

```bash
source .venv/bin/activate
python assignment2/run_experiment.py --phase smoke --models qwen_7b_local
```

Then, only after consciously allowing cloud use within the budget, run the
Gemini smoke test:

```bash
python assignment2/run_experiment.py --phase smoke --models gemini_flash_lite
```

Inspect every completed record before proceeding. A smoke pass is not final
evidence and must not be mixed with pilot or final analysis.

## 7. Pilot, freeze settings, and final evaluation (**not yet remotely validated**)

Use the pilot to validate model availability, action parsing, episode duration,
usage estimates, and whether the conservative USD 1 pilot ceiling is practical:

```bash
python assignment2/run_experiment.py --phase pilot --models gemini_flash_lite
python assignment2/run_experiment.py --phase pilot --models qwen_7b_local
```

Before final runs, record any justified changes to model identifiers, prices,
limits, or manifest in a commit. Do not silently alter tasks, seeds, or sample
size. Run the fixed final manifest one model at a time:

```bash
python assignment2/run_experiment.py --phase final --models gemini_flash_lite
python assignment2/run_experiment.py --phase final --models qwen_7b_local
```

To resume an interrupted run, reuse its exact result directory. Completed
records with a matching fingerprint are skipped; incomplete records remain as
separate attempts:

```bash
python assignment2/run_experiment.py --phase final --resume assignment2/results/<run-directory>
```

## 8. Analysis, report, and archives

Analyse only one phase at a time:

```bash
python assignment2/analysis.py assignment2/results/<final-run-directory> \
  --phase final \
  --output assignment2/analysis/<final-run-directory>
```

Read the generated CSV files, figures, and selected traces. Then complete the
evidence placeholders in `assignment2/report_outline.md`; do not claim a
reflection benefit unless the matched comparisons support it.

Finally create the three archives:

```bash
python assignment2/package_submission.py
```

Each archive deliberately contains `ARCHIVE_STATUS.json` marked **NOT
SUBMISSION READY** until real results and required written answers have been
added and reviewed.

## Remote smoke-test acceptance checklist

- [ ] Actual host, scheduler, allocation method, GPU count, GPU memory, CPU,
  RAM, disk quota, and wall-time limit are recorded from the allocated context.
- [ ] The environment installs and BALROG imports on the remote host.
- [ ] `nvidia-smi` confirms an allocated GPU before vLLM starts.
- [ ] The local `/v1/models` endpoint responds and BALROG's vLLM client reaches it.
- [ ] `GEMINI_API_KEY` is present without appearing in logs or source files.
- [ ] One BabyAI and one Crafter smoke episode complete for each intended agent.
- [ ] Records contain raw outputs, parse outcomes, plans/reflections, seeds,
  model settings, progression, calls, token availability, estimates, and paths.
- [ ] Malformed actions and injected/mock model failures become explicit
  invalid/incomplete records rather than successful zero scores.
- [ ] Resume skips only compatible complete records and preserves incomplete attempts.
- [ ] Pilot estimates remain below the chosen guardrail before final settings freeze.
