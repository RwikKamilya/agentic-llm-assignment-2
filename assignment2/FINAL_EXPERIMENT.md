# Corrected experiment: rationale and report handoff

**Historical pre-run protocol.** The campaign has now finished. Read
[the completed partner guide](report_materials/PARTNER_GUIDE.md) for actual results,
remaining parser/fallback limitations, and the previous-run comparison. Statements
below about evidence still to collect describe the plan before evaluation.

## What question can this experiment answer?

For two fixed language models on selected BabyAI missions and short Crafter
worlds, does an explicit maintained plan improve progression over action-only
prompting? Does adding scheduled feedback-based critique improve that planner,
and what does it cost in model calls, tokens and time?

The assignment asks for a baseline, your reasoning agent, and a self-reflective
agent, their model-by-benchmark matrices, concrete failure analysis, and a
considered assessment of stability and suitability. Better scores are useful,
but a well-supported negative result also answers those questions. This campaign
is designed to make the comparison interpretable. It cannot guarantee an
improvement or establish that reflection helps language models in general.

## What was wrong, and what changed?

| Earlier limitation | Correction in protocol v2 | What remains to check in the results |
|---|---|---|
| Different action cleanup for naive and structured agents | One narrow normalizer for case, whitespace, enclosing quotes and terminal punctuation; raw, normalized and applied actions all saved | Remaining invalid actions are substantive commands or format failures; report normalization counts too |
| BabyAI rules described pickup incorrectly | Shared rules say pickup acts on the front tile; toggle operates on a front door/object | Verify traces use the right interaction, not merely a valid command |
| Agent history contained requested actions even when the environment executed a fallback | Every condition sees applied action, invalid flag, reward and literal observation changes | Invalid actions still use BALROG's existing fallback; this policy is common and disclosed |
| Plans were retained without an explicit prediction check | Five-field actor response starts with OUTCOME_CHECK, then plan, subgoal, next prediction and action | These checks are model claims; manually assess whether they match visible evidence |
| Critic lacked the actor's plan, prediction and reliable execution feedback | Critic sees the same transition evidence, actor state and prior lessons | Inspect whether corrections are supported and whether subsequent actions use them |
| Critic was pressed to invent a correction | CHANGE, NO_CHANGE and UNCERTAIN are valid; uncertainty/invalid critique preserves earlier lessons | Lessons can remain stale or be wrong; there is no oracle verifier |
| Longer state crowded out more environment history | Common 1536-byte state reserve for all conditions; choose transition history before adding any agent-specific state | Full current observation and last execution feedback are always retained; older full transition records can be dropped |
| Parser accepted duplicate/empty sections and could partly update state after an invalid response | Unique ordered nonempty fields; initial UNCHANGED rejected; no partial state update | Structured formatting remains part of the planning method's practical burden |
| Some model mistakes could make an episode fail infrastructure validation | Empty/malformed actor/critic responses remain measured behavior; no score-based retry | Genuine API/usage/record failures still halt a phase |
| One BabyAI setting was the only repeat evidence | Two BabyAI settings and two Crafter worlds are each sampled three times | Three samples per setting characterize limited variability, not dependable universal stability |
| Crafter horizon was very short | Increase 128 to 192 actions | Still an early-achievement evaluation, not the original 1000-step benchmark |
| Previous final results had been inspected while redesigning | New final seeds, fresh results directory, frozen release | Earlier results are exploratory evidence for redesign; no clean causal v1-v2 claim |
| Trace selection and aggregate scores hid mechanisms | Deterministic best/worst cases per cell, task breakdown, all three paired contrasts, behavioral and cost columns | Read complete raw traces when excerpts are insufficient; do not cherry-pick only favorable cases |

## The three conditions, from scratch

**Action-only baseline:** one model call sees the public game rules, current text
observation and bounded executed-action history and returns one action. This
preserves BALROG's naive decision style but uses the corrected common prompt,
feedback and normalizer; describe it as an adapted baseline, not byte-for-byte
upstream naive.

**Planning:** one model call checks whether the last predicted outcome is
supported, contradicted or uncertain, then updates a short plan, immediate
subgoal, next prediction and action. Plans are capped at 24 words/160 UTF-8 bytes;
subgoals and predictions at 12 words/90 bytes. Invalid structured replies leave
the plan unchanged and submit an invalid action to the documented common
fallback. If a fallback was executed, the next step does not treat the rejected
action's prediction as its outcome prediction.

**Reflective planning:** the identical planner also receives a critic call after
every four completed actions, before deciding steps 4, 8, 12, etc. (zero-based).
The critic can replace up to three lessons, each at most 16 words/100 bytes, or
retain them with NO_CHANGE/UNCERTAIN. An invalid critique is logged, previous
lessons remain, and the actor continues. All state resets at each episode.

These are inference-time prompts and memory, not weight training. Reflection is
within-episode and Reflexion-inspired; it is not the original cross-trial
Reflexion algorithm. The critic uses the same model as its actor, so correlated
mistakes remain possible. No hidden game state, success labels or solutions are
fed to either agent. Reward and execution feedback come only from completed steps.

## Fixed protocol and denominators

| Item | Setting |
|---|---|
| Cloud model | `gpt-4o-mini-2024-07-18`, official OpenAI endpoint |
| Self-hosted model | `Qwen/Qwen2.5-7B-Instruct`, cached revision `a09a35458c702b33eeacc393d103063234e8bc28` on one L40S, BF16, no quantization flag |
| Sampling | vLLM server seed 0; temperature 0.2 for both; repeats use fresh requests, not a forced identical sampling seed |
| Input/output limits | 4096 conservative input-token bound, 192 output tokens; framing included |
| Primary BabyAI sample | 10 task/seed settings per model/agent; two each of goto, pickup, open, putnext, pick_up_seq_go_to; seeds 5101–5110 |
| Primary Crafter sample | Five worlds per model/agent; seeds 5401–5405 |
| Episode action limits | BabyAI 64, Crafter 192; natural termination can end earlier |
| Repeated settings | BabyAI goto/5101, pickup/5102; Crafter 5401, 5402 |
| Repetitions | Main observation plus two additional samples for every model/agent in each repeated setting |
| Totals | 90 primary + 48 extra repeat observations = 138 final episodes; 12 smoke episodes separately |
| Schedule | Fixed rotating agent order within each setting; single worker per model; critic every four actions |
| Spending | Shared historical ledger, cumulative $15 ceiling, $1 recovery reserve, up to two request attempts |

The old main seeds were 3101–3110 and 3401–3410. Do not add those observations to
this new final matrix. Repeated observations deliberately reuse four new main
settings: the independent setting counts remain 10 BabyAI and 5 Crafter, not 14
and 9. Repetition summaries include the original observation and its two extras.

The primary contrasts are planning minus naive and reflective planning minus
planning, paired by model, benchmark, task and seed. Reflection minus naive is
also exported as a useful secondary contrast. The pipeline checks initial visible
missions/observations match across agents, repetitions and models. A seed alone
would not be enough if reset behavior had changed.

## How to interpret the output

BabyAI progression is binary mission completion; its mean is a success fraction.
Crafter progression is distinct achievements divided by 22; multiply a mean by
22 to obtain the average achievement count. Do not average these benchmarks into
one overall score. The Crafter `success_rate` field means all 22 achievements;
it is not the useful primary metric for this short horizon.

`performance_matrix.csv` provides main sample counts, means, sample SD, SEM,
action counts, invalid-action rate, format errors, normalization counts, token
usage, critic calls, estimated API cost and elapsed time. Unchanged-observation
rate is a descriptive signal: unchanged text does not prove the entire hidden
state was unchanged. Output-limit stops expose possible truncation. Qwen API
cost is zero, not zero compute cost.

`paired_differences.csv` contains every paired difference, wins/ties/losses,
mean differences and exploratory percentile-bootstrap intervals. With five
Crafter worlds or ten BabyAI settings, intervals are coarse. Identical observed
outcomes can produce a zero-width bootstrap interval; that does not prove
population equivalence. These are selected task types, not a random sample of
all BALROG tasks. There is no confirmatory multiple-comparison claim.

`task_breakdown.csv` prevents two easier task types from hiding failure on harder
ones. With two BabyAI settings per task, use it descriptively. Repeated-setting
results include score SD, step SD, distinct applied-action sequences and invalid
actions for each repeat: equal final scores do not imply equal behavior.

The extra critic computation is not matched by a separate additional-call
control. Therefore any advantage belongs to the **reflection procedure as a whole**,
including its extra inference, rather than proving that criticism itself beats
all alternative uses of the same compute. Report actual calls/tokens/time beside
performance. Likewise the planner has a longer output and structured-format
burden than the action-only baseline; this is a comparison of practical agent
procedures, not a controlled measurement of an invisible reasoning ability.

## Trace inspection rubric for your partner

For each model/benchmark, read at least a success (if one exists) and a failure,
and compare the agents on the same seed. The exported mechanical cases are a
starting point; do not infer frequency from best/worst selections.

1. Did the requested action parse, normalize and execute as intended? If not,
   was the subsequent agent aware of the actual fallback?
2. Does the plan fit the current observation and mission? Is a changed relative
   direction explained by a turn? Does inventory support a crafting claim?
3. Does OUTCOME_CHECK correctly classify the visible transition? Mark unsupported
   certainty separately from a reasonable UNCERTAIN judgment.
4. Did a failed prediction lead to a useful change, or does a repeated turn/move
   cycle persist? A valid action is not necessarily an effective action.
5. For each reviewed critic call, is its EVIDENCE real? Are lessons actionable and
   supported? Did later actions follow them, and did visible progress follow?
   Following a lesson does not by itself establish causality.
6. Do the same failure patterns recur across settings or repeated samples?
   Keep a small annotated table with exact episode filename and step numbers.

Use only observed evidence for these answers. No score has been produced for v2
by writing this document, and no improvement is assumed.

## Report wording and structure

Exercise 1: justify cloud/local models and selected benchmarks; describe the
adapted baseline, horizons and progression definitions; insert its 2×2 matrix.
Discuss actual navigation, action-grounding and resource-interaction failures.

Exercise 2 two-sentence method: “The agent maintains a short plan, immediate
subgoal and predicted outcome, and compares the previous prediction with the
latest observed transition before selecting its next action. Its generic
planning prompt operates on benchmark-specific public rules and a common bounded
history, with all state reset between episodes.” Insert its matrix, primary
paired contrasts, sample sizes and observed brittleness.

Exercise 3 two-sentence method: “Every four completed actions, the same model
reviews the planner's state against executed actions, rewards and visible changes,
and may retain or revise up to three short lessons. The planner uses those lessons
on subsequent decisions, while invalid or uncertain critiques are recorded and
do not force a fabricated correction.” Insert its matrix, reflection-versus-plan
and reflection-versus-naive differences, repeats and suitability assessment.

If planning/reflection still loses, report that plainly and explain supported
mechanisms. If it improves only one cell, make a cell-specific claim. Do not
turn three identical repeated scores into a statement of reliable stability.
Use the earlier experiment as a clearly labeled development/redesign narrative;
because prompts, seeds and horizons changed, a higher v2 score is not an isolated
causal estimate of the effect of any one repair.

State the frozen source content ID, effective configs, model snapshot, package
versions, smoke/final distinction and exact commands. Include honest AI-assistance
disclosure under the course policy. Personal reflections must come from the
students; this document does not invent them. Finish the required report and
three exercise archives after inspecting real outputs.

## Files to read

1. `config/experiment.yaml`: the fixed design and budget.
2. `protocol.py`, `prompts.py`: shared game rules, normalization and evidence selection.
3. `agents.py`: actor/critic formats, bounded state, feedback and model calls.
4. `runner.py`, `validation.py`: episodes, provenance, completion and failure handling.
5. `analysis.py`: denominators, paired comparisons, repeats and trace selection.
6. `RUNBOOK.md`, `scripts/submit_final.sh`: deployment and the one-submission workflow.

The local regression suite uses synthetic model responses. The allocated
preflight additionally exercises the real game engines and paired resets without
API calls; the following smoke tests exercise actual models. Passing local tests
is not a claim that the remote campaign has already run.
