# Assignment 2: partner guide to the implementation, results, and report

Prepared from the assignment, executed source, and downloaded evidence on **22 October 2026**. This is a report-writing handoff, not a finished student report. Numbers are measured unless explicitly described as a bound, interpretation, or proposed future experiment.

## 1. Read this first

The assignment asks whether and how LLM agents can reason and self-reflect while playing games. It values a careful experiment and an honest explanation of failures; it does not require us to beat the benchmark.

We completed two versions of a two-model, two-benchmark, three-agent experiment. The current version, **v2**, has **138 completed final episodes: 90 primary observations and 48 extra repeated observations**, plus 12 separate smoke episodes. Both model jobs exited successfully. An independent audit validates every final identity, confirms that paired v2 settings started with the same visible task, and reproduces the saved performance matrices.

**The defensible finding is mixed performance and substantial harness brittleness, not a general success of planning or self-reflection.** GPT-4o mini's planner ties its action-only baseline on both benchmarks. Qwen's planner and reflective planner score higher than its baseline on BabyAI, but one success in each structured condition is produced by a fallback action after a rejected reply. Qwen's action-only baseline is best on Crafter. Most critic responses are rejected by a mismatch between the prompt's loose instructions and the parser's stricter contract.

The work supports a useful assignment report about the interaction between model behavior, parsing, memory, fallback actions, and evaluation. It is a **limited comparison of complete agent procedures**. It is a weak test of whether successful self-reflection itself improves reasoning, because most scheduled critiques never update the agent's memory.

There was also a post-evaluation packaging failure: the frozen deployment receipt had a 1970 timestamp, which the original ZIP writer rejected. That failure occurred after evaluation and analysis. The post-run packaging fix clamps ZIP timestamps, validates the original source and evidence, and writes complete archives atomically. It does not change experimental records or rerun any model. The original failure log and original deployment remain preserved.

Start with sections 6, 8, 9, and 12 if you already know the project. Read sequentially for the explanation from scratch.

## 2. What exactly was asked?

The authoritative source is [Ass2Reasoning.pdf](../../references/Ass2Reasoning.pdf), especially pages 3–8. The book is [Agentic_LLM_Book2.pdf](../../references/Agentic_LLM_Book2.pdf). The preparation asks for Parts 2 and 3, Chapters 6–9; the assignment draws on Chapter 10.

| Requirement | What our report should contain | Evidence or qualification |
|---|---|---|
| Exercise 1.1 | `Ass2-1.zip`, with scripts, prompts, code, and README | Repaired evidence archive; add the reviewed student report if required by the submission arrangement |
| Exercise 1.2 | Which models and games were tried, and why; cloud versus self-hosted | GPT-4o mini and Qwen2.5-7B; BabyAI and Crafter; feasibility and contrasting task demands |
| Exercise 1.3 | Model × benchmark performance matrix and exact metric | Action-only matrix in section 6; metric definitions in section 4 |
| Exercise 1.4 | What failed, what the games measure, what the models do well, reasoning claims, improvement ideas | Sections 8–12; students add their own genuine experience |
| Exercise 2.1–2.2 | `Ass2-2.zip`, README, and planning matrix | Same preserved evidence, planning condition identified |
| Exercise 2.3 | Two-sentence idea; generic or benchmark/model-specific; best case | Section 13 provides checked wording and qualified answers |
| Exercise 2.4 | Models/games, statistical run count, brittleness, implementation experience, future changes, suitability | Sections 5, 9–12; distinguish episodes from independent settings |
| Exercise 3.1–3.4 | `Ass2-3.zip`, README, reflective matrix, two-sentence mechanism, model/game focus and best case | Sections 3, 6, and 13 |
| Exercise 3.5 | Failures, stability, and whether self-reflection suits reasoning tasks | Repeated-run evidence plus critic-adoption failures; avoid a universal conclusion |
| Exercise 3.6 | Optional, ungraded philosophy in fewer than five sentences | Students write their own position; it is not an empirical result |
| Evaluation hints | Repeated runs, trace inspection, careful interpretation of progression | Four repeated v2 settings, complete raw traces, and explicit uncertainty |

The assignment begins by asking students to try standard BALROG evaluation. **Our reported baseline is adapted**, and our small task subsets and short horizons do not reproduce the paper's evaluation. Describe the work as a scoped BALROG-based study, not a reproduction of the published leaderboard. Do not claim few-shot experiments, extra model comparisons, or exact standard-baseline measurements without matching records. Reading a tutorial does not mean we ran that experiment.

## 3. What is an agent here, and how does our code work?

### 3.1 Basic vocabulary

An **LLM** generates text. An **agent** is the surrounding program that decides what information to send, interprets the reply, executes an action, and remembers information for the next decision. A better model and a better agent program are different things.

An **environment** is the game. An **observation** is the text the game exposes at the current step. An **episode** starts from a reset and ends at success, death, another game termination, or the action limit. A **setting** is a benchmark, task type, and environment seed. Different agents are compared on the same setting within a version.

A **seed** controls environment generation, not the entire experiment. Model sampling, serving behavior, and different action sequences can still yield different trajectories. A **repeat** uses the same setting again with fresh model requests. Repeated observations are not additional independently sampled game settings.

There is no neural-network training here. The model weights stay fixed. “Learning” means changing short text state within an episode. State is reset between episodes. The model-generated plan is an observable output, not privileged access to the model's internal reasoning or proof that its verbal explanation caused its action.

### 3.2 Common execution loop

1. Load the fixed task/seed/model/agent specification and verify the release.
2. Construct the requested game and reset it. Save the mission, seed, and initial observation.
3. Build a prompt from public game rules, the full current text observation, recent executed transitions, and the condition's instructions/state.
4. Ask the model for its next reply. The reflective condition may first ask the same model to act as a critic.
5. Parse the reply, normalize the action text, and ask BALROG whether the normalized action is in its action vocabulary.
6. Execute the accepted action or BALROG's fallback; record the action actually applied, reward, termination, and next observation.
7. Save the episode atomically after each step. Continue until the episode ends.
8. Validate the complete record and aggregate only compatible results.

The raw JSON separates the actor's complete `raw_response`, the parser's `parse.errors`, `requested_action` returned by the agent, `normalized_action`, and `applied_action`. This distinction is essential: after a parse failure, `requested_action` can be empty even if the raw response contained a plausible `ACTION` line.

### 3.3 Action-only baseline

`LoggedNaiveAgent` makes one actor call per step and asks for exactly one action. It receives the same public rules and common evidence-selection policy as the other conditions. It has no plan or lesson memory. This preserves the naive action-only idea but is not the unmodified upstream naive implementation.

### 3.4 Planning agent

`PlanningAgent` requests these five ordered fields:

```text
OUTCOME_CHECK: <FIRST, SUPPORTED, CONTRADICTED or UNCERTAIN>; <evidence>
PLAN: <plan, or UNCHANGED if a plan already exists>
SUBGOAL: <immediate objective>
EXPECTED_OUTCOME: <prediction for the next action>
ACTION: <one allowed action>
```

The intention is to compare the previous prediction with the visible outcome before choosing the next action. The stored plan is limited to 24 words/160 UTF-8 bytes; subgoal and prediction to 12 words/90 bytes each. The model is asked to stay below 85 words per reply; the actual API output ceiling is 192 tokens.

The parser requires each heading once, in order, with nonempty contents. `OUTCOME_CHECK` needs a status, semicolon, and nonempty evidence. `UNCHANGED` is rejected when no plan exists. **Any structured parsing error rejects the whole action and prevents all state updates**, even if the action line itself would have been legal. This conservative choice becomes a major experimental limitation in section 8.

### 3.5 Reflective planning agent

`ReflectivePlanningAgent` uses the same actor and adds a critic after every four completed actions, before the next actor call. For an episode of `T` actions, there are `floor((T - 1) / 4)` critic calls: no critique is made after the episode has already ended.

The critic sees the planner's state, retained lessons, current observation, and recent executed transitions. It must return:

```text
STATUS: <CHANGE, NO_CHANGE or UNCERTAIN>
EVIDENCE: <visible evidence>
LESSONS: <NONE or short bullet lessons>
```

A valid `CHANGE` replaces the retained lessons with up to three lessons, each limited to 16 words/100 bytes. Valid `NO_CHANGE` or `UNCERTAIN` leaves the previous lessons unchanged. An invalid critique also leaves memory unchanged and allows the actor to continue. The parser additionally requires **`LESSONS: NONE` for `NO_CHANGE` and `UNCERTAIN`**, a dependency the actual critic prompt did not state clearly enough.

This is within-episode, feedback-based critique inspired by verbal-memory approaches. It is **not an implementation of cross-trial Reflexion**: our state does not carry between trials. ReAct interleaves reasoning and acting; Reflexion uses feedback and episodic text memory across trials. Cite these as background, not algorithms reproduced exactly. See the primary papers in section 16.

### 3.6 What was shared, and what was not?

All conditions share corrected public rules, text observations, the legal action list, a narrow normalizer, applied-action feedback, maximum action counts, and a maximum of three recent transitions. A common 1,536-byte reserve is set aside before adding agent-specific instructions/state. Evidence is selected first; longer planning text cannot itself evict extra history. All agents preserve the full current observation and latest feedback or fail before a request if these cannot fit.

This is the same **selection rule**, not identical observations at every step: once agents take different actions, their trajectories and observations differ. The code still updates BALROG's history builder, but v2's actual requests use the compact shared transition representation. Do not describe it as simply “six full observations of history.”

The normalizer accepts case differences, whitespace, enclosing quotes/backticks, and specified terminal punctuation. It does not invent actions or extract a command from arbitrary prose. Invalid actions use existing BALROG defaults: **`go forward` in BabyAI and `Noop` in Crafter**. A legal action can still be ineffective, such as moving into a wall.

The planner has longer output and a harder response format than the baseline. The critic adds calls and tokens. We did not include an equal-compute control. Therefore the comparison concerns entire procedures, including their formatting and cost burden.

## 4. What do the games and scores measure?

**BabyAI** tests following small language instructions in a grid world: navigating, picking up objects, opening doors, placing objects beside others, and carrying out a short sequence. Observations describe positions relative to the agent's facing. Turning changes those relative descriptions. The wrapper sets progression to 1 after a positive mission-completion reward, otherwise 0. Thus:

$$
\text{BabyAI mean progression}=\frac{\text{successful missions}}{\text{evaluated settings}}.
$$

Five successes in ten primary settings means 50%, not 50% completion within each game. The primary score ignores how quickly a successful mission was completed; step counts are a separate measure. The one-step success on seed 5101 is much easier than a multi-stage locked-door task.

**Crafter** tests exploration, survival, resource interaction, and crafting prerequisites. Its wrapper records the number of distinct achievements completed, divided by 22:

$$
\text{Crafter progression}=\frac{\text{distinct achievements completed}}{22}.
$$

A mean of 0.10 means an average of 2.2 achievements, not that 10% of episodes were “won.” The `success_rate` column for Crafter tests progression equal to 1 (all 22 achievements) and is zero throughout these experiments; mean progression is the useful short-horizon outcome. One achievement is about 4.55 percentage points for an individual episode.

**Do not average BabyAI and Crafter into one headline score.** Their normalized numbers represent different constructs. Our 192-step Crafter evaluation is an early-achievement task, not the original long-horizon benchmark. These scores measure achieved behavior under a particular interface and budget; they do not isolate abstract reasoning ability.

## 5. Exact experimental design and the meaning of “number of runs”

| Item | Current v2 setting |
|---|---|
| Cloud model | OpenAI `gpt-4o-mini-2024-07-18` |
| Open, self-hosted model | `Qwen/Qwen2.5-7B-Instruct` on an allocated NVIDIA L40S; vLLM 0.31.0, BF16, no quantization |
| Qwen snapshot | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Model request settings | Temperature 0.2; at most 192 output tokens; conservative 4,096 input-token bound calculated from UTF-8 bytes and framing |
| Serving details | Qwen server seed 0, maximum sequence length 4,352, port 18085; one evaluation worker per model |
| BabyAI primary settings | Seeds 5101–5110; two settings each of `goto`, `pickup`, `open`, `putnext`, and `pick_up_seq_go_to` |
| Crafter primary settings | Five worlds, seeds 5401–5405 |
| Episode limits | BabyAI 64 actions; Crafter 192 actions; natural termination may occur earlier |
| Repeated settings | BabyAI `goto`/5101 and `pickup`/5102; Crafter worlds 5401 and 5402 |
| Repetitions per repeated setting | Primary observation plus two extras, for every model and agent |
| Order | Fixed rotation of agent order within each setting; benchmarks are not fully interleaved |
| Memory | Reset every episode; no cross-trial transfer |
| Retries | At most two total attempts per model request; no retries actually occurred in these final runs |

Qwen's server log also reports model-default `top_p=0.8`, `top_k=20`, and `repetition_penalty=1.05`. Requests explicitly set temperature 0.2; they do not explicitly equalize these other sampling parameters across providers. Thus “same temperature” is not “identical decoding.” The log also lists a model-default temperature of 0.7; do not substitute it for the explicit 0.2 in the saved request configuration. Record both the client settings and this serving qualification.

For each model/agent there are **10 primary BabyAI observations and 5 primary Crafter observations**. Across two models and three agents that is `2 × 3 × (10 + 5) = 90` primary observations. The four repeated settings add `2 × 3 × 4 × 2 = 48` extra observations, making **138 final episodes**. Each model runs 69 final episodes. Twelve short smoke episodes are separate and never enter the performance matrices.

The underlying primary environment sample is **15 distinct settings**, reused across agents and models. Within a model/benchmark comparison, agents are paired on the setting. Do not call 90 primary agent observations 90 independent game settings. Similarly, do not call the four repeated settings 12 independently sampled tasks.

The earlier v1 experiment also had 138 final episodes, but allocated them differently: **120 primary observations** (10 BabyAI and 10 Crafter settings per model/agent) plus **18 extra observations** (three samples of BabyAI `goto` seed 3901 for each model/agent). Seed 3901 was separate from the v1 primary set. V1 had no Crafter repeat evidence. Both versions used 64 BabyAI actions; Crafter increased from 128 to 192 actions in v2.

V1 primary seeds were BabyAI 3101–3110 and Crafter 3401–3410. V2's primary seeds are disjoint. The v1 results informed the redesign, so present v1 as the earlier exploratory version and v2 as the later fixed evaluation. This was not a preregistered, randomly sampled confirmatory study.

## 6. Current results: the three required performance matrices

All entries below are **mean progression × 100**, from primary observations only. BabyAI uses (n=10) settings per cell; Crafter uses (n=5). Use these denominators in every figure caption. Exact means, SD, SEM, steps, calls, tokens, and costs are in [v2_performance_matrix.csv](v2_performance_matrix.csv).

### Exercise 1: adapted action-only baseline

| Model | BabyAI success | Crafter mean progression |
|---|---:|---:|
| GPT-4o mini | 50.0% (5/10) | 10.00% (2.2 achievements) |
| Qwen2.5-7B-Instruct | 10.0% (1/10) | 13.64% (3.0 achievements) |

### Exercise 2: planning

| Model | BabyAI success | Crafter mean progression |
|---|---:|---:|
| GPT-4o mini | 50.0% (5/10) | 10.00% (2.2 achievements) |
| Qwen2.5-7B-Instruct | 20.0% (2/10) | 8.18% (1.8 achievements) |

### Exercise 3: reflective planning

| Model | BabyAI success | Crafter mean progression |
|---|---:|---:|
| GPT-4o mini | 40.0% (4/10) | 8.18% (1.8 achievements) |
| Qwen2.5-7B-Instruct | 30.0% (3/10) | 4.55% (1.0 achievement) |

The cloud model performs better on these BabyAI settings under every condition. Qwen's action-only condition has the highest Crafter mean, while GPT-4o mini has the higher Crafter means under planning and reflection. Neither model is uniformly better across all conditions.

Planning ties GPT-4o mini's baseline in both benchmarks, but matching means do not imply the same episodes were solved. On BabyAI, planning gains the door-opening mission at seed 5103 and loses the pickup mission at 5102. All six model/agent conditions fail both `putnext` primary settings. The task breakdown has only two settings per BabyAI task type, so avoid generalizing a task-specific percentage.

The original [matrix figure](v2_performance_matrix.png) and [paired-difference figure](v2_paired_differences.png) are included, with PDF versions. The matrix has crowded horizontal labels; use the tables above or re-typeset from the CSV for the submitted report. The matrix uses a common 0–1 color scale for display; that is not a justification for combining the benchmark scores.

## 7. How does this compare with the previous run?

| Model | Benchmark | Agent | v1 | v2 | Change, percentage points |
|---|---|---|---:|---:|---:|
| GPT-4o mini | BabyAI | Action-only | 40.00% | 50.00% | +10.00 |
| GPT-4o mini | BabyAI | Planning | 0.00% | 50.00% | +50.00 |
| GPT-4o mini | BabyAI | Reflective | 20.00% | 40.00% | +20.00 |
| GPT-4o mini | Crafter | Action-only | 9.09% | 10.00% | +0.91 |
| GPT-4o mini | Crafter | Planning | 3.18% | 10.00% | +6.82 |
| GPT-4o mini | Crafter | Reflective | 7.27% | 8.18% | +0.91 |
| Qwen | BabyAI | Action-only | 20.00% | 10.00% | -10.00 |
| Qwen | BabyAI | Planning | 20.00% | 20.00% | 0.00 |
| Qwen | BabyAI | Reflective | 0.00% | 30.00% | +30.00 |
| Qwen | Crafter | Action-only | 11.82% | 13.64% | +1.82 |
| Qwen | Crafter | Planning | 0.00% | 8.18% | +8.18 |
| Qwen | Crafter | Reflective | 1.82% | 4.55% | +2.73 |

Source: [previous_vs_current.csv](previous_vs_current.csv). V1 has (n=10) in every cell; v2 has (n=10) BabyAI and (n=5) Crafter. Means improved in 10 cells, tied in one, and fell in one. This is a descriptive count, not a significance test or an overall performance metric.

**Do not write “the fix caused a 50-point improvement.”** The versions differ simultaneously in task seeds, rules, prompt content, action normalization, parsing, evidence selection, output ceiling (256 → 192), repeat allocation, and Crafter sample size/horizon. V1 and v2 are not paired trials on the same settings. A longer Crafter horizon alone offers more opportunity for achievements. The new seeds can be easier or harder.

What can we attribute directly to code and trace evidence?

| Earlier issue | V2 change | What the data now say |
|---|---|---|
| Different cleanup for naive and structured actions | One narrow common normalizer | V1 OpenAI Crafter planning had 221 invalid commands; 193 strings become legal under v2 normalization. This diagnoses lexical rejection, not a counterfactual score improvement. |
| Incorrect BabyAI pickup description | Shared rules correctly describe front-tile interaction | The rule is corrected, but models can still choose the wrong interaction, as section 9 shows. |
| Requested actions could appear in history when a fallback actually ran | Explicit applied-action feedback | V2 can expose fallback-driven success and failure. |
| Richer agent state squeezed out history | Evidence selected before agent state, with a common reserve | V1 OpenAI Crafter trimming affected 172/966 baseline calls, 713/1,048 planning calls, and 1,009/1,392 reflective actor-plus-critic calls. V2 uses a fairer selection policy, though trajectories still differ. |
| No explicit prediction check | `OUTCOME_CHECK` and `EXPECTED_OUTCOME` | More interpretable records, but the new strict format causes major Qwen failures. |
| Critic lacked the necessary actor context | Critic receives plan, prediction, lessons, and executed feedback | Better information access; most replies still fail the conditional lesson contract. |
| Repeat evidence only on one BabyAI setting | Repeats on two BabyAI and two Crafter settings | Broader stability evidence, still only three observations per setting. |
| Limited request-level evidence | Exact v2 messages, raw replies, metadata, and source fingerprints | Stronger auditability; this does not make the underlying agent effective. |

V1's structured parser reported zero format failures under its simpler rules. That does not make it better: validly parsed commands could still be illegal, and the critic followed a different format. Compare parser behavior in the context of the actual contract, not just the error count.

## 8. What did we still get wrong in v2?

### 8.1 The actor parser created persistent failure loops

In the primary Qwen BabyAI data, planning has **323 rejected actor replies out of 536 calls (60.26%)**; reflection has **263/495 (53.13%)**. Every one of those rejected replies leads to an invalid requested action and the default `go forward` action.

The common sequence is concrete: Qwen first writes `OUTCOME_CHECK: FIRST;` with nothing after the semicolon. The reply is rejected, so its otherwise plausible plan is not stored. Next it writes `PLAN: UNCHANGED`, which the parser rejects because no plan exists. This repeats. Of the primary Qwen BabyAI planning errors, 318 report “UNCHANGED requires an existing plan” and 5 report missing outcome evidence; the reflective counts are 259 and 4.

The agent is told to use `FIRST` when no prior prediction exists, but still must supply evidence after the semicolon. We failed to make that first-step contract robust for the actual model, for example with an unambiguous valid example. The all-or-nothing parser then magnifies one formatting error into repeated behavior failure.

This is a harness/model interaction, not simply evidence that Qwen cannot plan. Rejecting uncertain state updates is defensible, but coupling every state-format error to rejection of an otherwise legal action needs to be treated as a design choice and evaluated.

### 8.2 Most critic calls did not deliver a usable memory update

Primary observations only:

| Model / benchmark | Critic calls | Rejected replies | Rejected fraction | Accepted `CHANGE` calls |
|---|---:|---:|---:|---:|
| GPT-4o mini / BabyAI | 106 | 77 | 72.64% | 29 |
| GPT-4o mini / Crafter | 117 | 106 | 90.60% | 11 |
| Qwen / BabyAI | 115 | 107 | 93.04% | 6 |
| Qwen / Crafter | 189 | 174 | 92.06% | 11 |

All these rejected primary critiques violate the same rule: they select `NO_CHANGE` or `UNCERTAIN` but still provide lessons. The prompt says to retain earlier lessons for these statuses, and its generic schema permits “NONE, or ... bullet lessons”; it does not clearly spell out the parser's conditional requirement. The parser rejects the reply and retains memory.

For GPT-4o mini all accepted primary critiques were `CHANGE`. Qwen also had two valid retain-status critiques in BabyAI and four in Crafter. These are valid replies but not new lesson updates. Accepted text is still not guaranteed to be factually correct.

**Do not describe all 527 primary critic calls as 527 successful reflections.** There were 464 rejected critiques, 57 accepted `CHANGE` calls, and 6 accepted retain-status calls. Calling a critic is an intervention; successfully integrating useful reflection is a stronger claim that these aggregate scores do not establish.

### 8.3 Fallback can both harm and artificially help apparent performance

The fallback is shared across agents but triggered at very different rates. In BabyAI, it actively moves forward rather than doing nothing. On the easy `goto` seed 5101, this move completes the mission despite a rejected Qwen actor reply. Both Qwen planning and reflective planning receive a success in this way.

Among primary BabyAI observations, successes whose **terminal action was validly requested** are Qwen baseline 1/10, planning 1/10, and reflection 2/10. The official pipeline scores remain 1/10, 2/10, and 3/10. This diagnostic explains the fallback contribution; it is **not a replacement score or an estimate of what would happen under another fallback policy**. Earlier invalid actions may also affect trajectories. Do not delete, relabel, or rerun these episodes after seeing their scores.

### 8.4 Smoke and tests were necessary but not sufficient

Both jobs passed 43 offline tests. Both real smoke phases validated. However, Qwen's 22 smoke actor calls included 9 parser failures, while OpenAI's included 2. The smoke gate checked infrastructure, accounting, recording, and resets, not whether the structured protocol was practical for the model.

That separation correctly prevented poor game scores from being silently discarded. It also meant we entered the final run with a visible response-contract problem. A future workflow should use a disjoint development pilot to diagnose parsing and critic adoption before freezing the final experiment. It should not choose final seeds or rerun final episodes based on favorable outcomes.

### 8.5 The packaging marker overstated completion

Jobs 22881 (Qwen) and 22882 (OpenAI) exited 0. Their 69/69 final identities validate. Job 22883 completed combined analysis but failed in ZIP creation with `ValueError: ZIP does not support timestamps before 1980`. The old launcher had already written `COMPLETE`, so that marker alone did not mean the submission packages were ready. Only a partial `Ass2-1.zip` was downloaded.

The repaired packager supports a separate frozen source root, clamps ZIP timestamps, checks archive integrity, and only replaces a target after the ZIP is finished. The updated launcher writes `ANALYSIS_COMPLETE` before packaging and `COMPLETE` only after packaging. These are **post-run delivery repairs**, not changes that generated the reported results. Never claim the remote analysis job exited successfully.

## 9. Concrete traces your partner should inspect

Step indices below are zero-based. Use [episode_index.csv](episode_index.csv) to locate a full record after extracting the evidence bundle. V2 records contain exact `model_call_traces[].messages`; check those rather than reconstructing prompts from today's code.

| Case | Episode key / setting | What is observed | What it supports |
|---|---|---|---|
| Fallback-assisted success | Qwen planning `b960e66e2f20dfb2bcec`, BabyAI goto/5101, step 0 | Mission: go to yellow ball, visible two steps forward. `FIRST;` lacks evidence; reply rejected; applied fallback `go forward`; mission succeeds in one action. | A success can be produced by the harness despite invalid output. The raw action line also says `go forward`, so do not claim the model proposed a different move in this example. |
| Parser failure loop | Qwen planning `febbfd71b6ea4993d17d`, pickup/5102, steps 0, 1, 63 | First reply proposes `turn right` but is rejected; fallback moves forward. Next `UNCHANGED` refers to a plan never stored. All 64 actions are fallbacks and the episode fails. | Parser state and fallback compound an initial formatting error. |
| Valid format, poor grounded control | OpenAI planning `d152a1e11e24c594c29d`, pickup/5102, steps 0–3 and 63 | Alternates right/left early. At step 63 the green box is one step in front, but it chooses `go forward` instead of pickup. Zero invalid actions; fails at the 64-step limit. | Syntactic compliance is not task competence; a correct-looking plan does not guarantee the right action. |
| A real planning success | OpenAI planning `aabfa60a96cd588e9eda`, open/5103, steps 56–57 | Holds a yellow key facing a locked yellow door. A forward action leaves the view unchanged; the next action is `toggle`, and the mission succeeds in 58 actions. | A grounded interaction can succeed; the preceding `SUPPORTED` claim is not itself evidence of a correct prediction check. |
| Reflection present but causality uncertain | Qwen reflective `a009ef0f48ae183b791e`, pickup/5107, steps 4, 8, 12, 16, 18 | Four critic calls; two accepted `CHANGE`, two rejected `UNCERTAIN` with lessons. At step 8 the critic correctly says a box three steps left is not within reach, but the actor still tries `pick up`. Eventually pickup succeeds at step 18. | Accepted critique need not be immediately followed, and eventual success does not prove that critique caused it. |

For the last case, the critic sometimes says the box “moved” or its position “changed.” In BabyAI, relative coordinates change when the agent turns or moves; that wording is not evidence the object moved in world coordinates. This is precisely why critic evidence must be checked against the environment's observation semantics.

The original mechanical best/worst trace selection remains in the raw campaign's `analysis/selected_traces.md`. Its absolute paths refer to calcium; use the index to map to the extracted files. These illustrations were selected after inspecting results and must be described as case studies, not an unbiased estimate of failure frequency. The error tables provide the systematic counts.

## 10. Paired comparisons, uncertainty, and stability

The primary within-v2 contrasts pair each task/seed across agents. Positive differences favor the first named method. Entries are **percentage-point differences**, not relative percentage increases:

| Model / benchmark | Contrast | Mean difference | Exploratory bootstrap 95% interval | Wins / ties / losses |
|---|---|---:|---:|---:|
| OpenAI / BabyAI | Planning − baseline | 0.00 | [-30.00, 30.00] | 1 / 8 / 1 |
| OpenAI / Crafter | Planning − baseline | 0.00 | [-7.27, 6.36] | 2 / 2 / 1 |
| Qwen / BabyAI | Planning − baseline | +10.00 | [-20.00, 40.00] | 2 / 7 / 1 |
| Qwen / Crafter | Planning − baseline | -5.45 | [-11.82, -0.91] | 0 / 2 / 3 |
| OpenAI / BabyAI | Reflection − planning | -10.00 | [-30.00, 0.00] | 0 / 9 / 1 |
| OpenAI / Crafter | Reflection − planning | -1.82 | [-6.36, 1.82] | 1 / 2 / 2 |
| Qwen / BabyAI | Reflection − planning | +10.00 | [0.00, 30.00] | 1 / 9 / 0 |
| Qwen / Crafter | Reflection − planning | -3.64 | [-11.82, 4.55] | 1 / 1 / 3 |

These are 2,000-sample percentile bootstraps of matched settings, with RNG seed 0. There is no correction for the multiple comparisons, the seeds were selected rather than randomly sampled from all possible tasks, and (n=5) Crafter is very small. Intervals are exploratory descriptions, not a license for broad significance claims. Touching zero does not prove a benefit; equal observed means do not prove equivalence.

The strongest consistent adverse pattern is Qwen reflection versus its baseline in Crafter: reflection scores lower in **all five worlds**, with a mean difference of -9.09 points. The percentile interval is [-13.64, -5.45], but a simple two-sided sign test on five non-tied pairs would give (2(1/2)^5=0.0625). That post-hoc calculation illustrates the fragility of strong small-sample claims; it is not an additional confirmatory test.

Repeated sampling addresses a different question: can the same procedure behave differently on the same initial task? Selected examples, with the primary observation listed first:

| Model / agent / setting | Three scores | Interpretation |
|---|---|---|
| OpenAI planning / BabyAI pickup 5102 | 0, 0, 1 | Same setting can fail twice and succeed once. |
| OpenAI reflective / BabyAI pickup 5102 | 0, 1, 0 | Reflection does not make this setting reliably successful. |
| Qwen planning / BabyAI pickup 5102 | 0, 0, 0 | Stable failure, with identical applied-action sequences; not reliable competence. |
| Qwen planning and reflective / BabyAI goto 5101 | 1, 1, 1 in each condition | Stable score can conceal repeated fallback-assisted success. |
| OpenAI baseline / Crafter 5401 | 1/22, 6/22, 4/22 | Large variability in achievements on the same world. |
| Qwen baseline / Crafter 5402 | 4/22, 4/22, 4/22 | Equal scores but three distinct action sequences. |

All 24 model/agent/repeated-setting rows are in [v2_repeated_run_variability.csv](v2_repeated_run_variability.csv). Three observations are enough to show that variability exists in these examples, but not to estimate a reliable long-run success probability. The assignment's suggestion about five identical observations is guidance, not evidence we produced five repetitions. We produced three.

Do not report standard errors across all 138 episodes as if they were independent samples. Keep benchmarks separate, use the primary setting denominator, and report within-setting repeat variation separately.

## 11. Cost, calls, and runtime

These rows include all 69 final episodes per model, including repeats. They exclude smoke and server startup. Durations are manifest start-to-completion elapsed time, including local bookkeeping, not pure GPU compute time.

| Version / model | Actor calls | Critic calls | Total calls | Estimated API cost | Final-phase elapsed |
|---|---:|---:|---:|---:|---:|
| v1 / OpenAI | 5,369 | 450 | 5,819 | $0.74637 | 6 h 15 min |
| v1 / Qwen | 5,670 | 462 | 6,132 | $0 API charge | 2 h 35 min |
| v2 / OpenAI | 5,572 | 381 | 5,953 | $0.77559 | 5 h 15 min |
| v2 / Qwen | 6,526 | 484 | 7,010 | $0 API charge | 1 h 47 min |

V2 OpenAI smoke cost an estimated $0.00258105. V2 smoke plus final therefore cost **$0.77817120**. The historical cumulative ledger ends at **$1.730182**, with no outstanding reservation, under the $15 cap. These are token-based estimates using the recorded pricing policy, not a provider invoice. Local Qwen has compute and allocation costs even though no external inference API charge is recorded.

The pre-run maximum new cost of about $12.45 reserved full input/output ceilings and two attempts for every allowed call. It was a safety bound, not the expected bill. Actual responses were shorter and the final runs had no retries, empty responses, or output-limit stops.

Reflective episodes can have fewer total calls than planning episodes when they terminate earlier. This does not show a cheaper reflection mechanism: it can reflect death, failure, or another short trajectory. Compare scores, steps, actor/critic calls, tokens, and runtime together. Cross-model wall times also include API latency, serialization, and different trajectories; they are not a clean hardware speed benchmark.

## 12. Is the result good, and does it support proper research?

**Execution quality:** good enough to support an auditable descriptive report. Both model jobs finished; final records validate; complete raw data are preserved; no failed game was silently excluded for a low score; paired initial observations match; and the saved matrices are reproducible. The source snapshots match the recorded execution hashes. The packaging failure was real but occurred after measurement and can be repaired without inference.

**Task performance:** modest and mixed. No procedure solves the whole BabyAI sample, all fail `putnext`, and Crafter achievements remain few. Planning does not generally improve the baseline. Reflection is not a general improvement and is especially poor for Qwen in Crafter. A negative result can be scientifically useful and satisfies the assignment's invitation to analyze failure.

**Research validity:** sufficient for a narrow case study of these implementations, insufficient for a strong claim about the value of reasoning or reflection in general. The main qualifications are:

| Limitation | Consequence for the report |
|---|---|
| Most critic outputs rejected; strict actor-state failure loops | The intended mechanism often was not delivered successfully. Report critic adoption, not just the condition label. |
| Fallback actions influence scores | Results describe model plus harness; some successes do not demonstrate accepted model decisions. |
| Small, selected sample | Rankings may change with seeds; two BabyAI settings per task and five Crafter worlds are weak population evidence. |
| Three repetitions on four settings | Limited stability evidence; no temperature sweep or universal repeatability claim. |
| Extra compute and different format burdens | Cannot isolate the effect of reasoning text or criticism from additional tokens/calls and parsing difficulty. |
| Adapted baseline and shortened horizons | Not a reproduction of published BALROG results. |
| Multiple simultaneous v1→v2 changes | Numerical improvement is not a causal estimate of one repair. |
| Same model acts and critiques | Errors can be correlated; the critic is not an independent ground-truth evaluator. |
| Bounded text state and relative-coordinate observations | Memory loss, spatial misunderstanding, and state compression can dominate performance. |
| Different provider/serving defaults | Same temperature does not equal identical sampling; hosted model internals are not controlled. |
| No formal human annotation study | Case descriptions explain observed mechanisms but do not provide inter-rater reliability or a complete hallucination rate. |

An appropriate answer to “are LLMs good at reasoning?” is: these systems sometimes translate observations into successful multi-step behavior, but performance depends strongly on model, task, and interface; verbal plans and critique do not guarantee better grounded decisions. This experiment cannot establish or refute general reasoning ability or human-like self-awareness.

For a future experiment, first simplify and explicitly specify the actor/critic contracts on disjoint development settings. Make `FIRST; no prior action` unambiguous. Decide in advance whether a uniquely legal action can execute when only the state fields fail; log the distinction. For retain-status critiques, either explicitly require `NONE` in the prompt or accept-and-ignore proposed lessons under a predefined policy. Measure parser recovery and actual lesson adoption before freezing the next evaluation.

Then choose one change to test on held-out matched settings, keep horizons and decoding fixed, include an equal-budget extra-call control, and broaden independent worlds before adding more repeats of a few easy cases. Evaluate fallback behavior explicitly. These are **proposals**, not changes applied to the reported run. Offline re-parsing can diagnose syntax sensitivity, but cannot reconstruct the game trajectories that different executed actions would have produced.

## 13. Ready-to-adapt report wording and writing plan

### Suggested abstract

We compared action-only prompting, a stateful planning agent, and a planner with periodic self-critique using GPT-4o mini and self-hosted Qwen2.5-7B-Instruct on selected BabyAI missions and short-horizon Crafter worlds. The later evaluation comprised 90 primary agent–setting observations and 48 additional repeated observations, with ten primary BabyAI settings and five Crafter worlds per model and agent. Planning tied the OpenAI baseline on both benchmarks, while Qwen's apparent BabyAI improvements were partly affected by fallback actions; reflection reduced Qwen's mean Crafter progression from 13.64% to 4.55% relative to its baseline. Trace analysis revealed actor parsing failure loops and rejection of most critic responses, showing that the practical interaction between model output and agent harness can dominate the intended reasoning mechanism. The results support a limited, auditable comparison of these procedures rather than a general claim that planning or self-reflection improves LLM reasoning.

### Exercise 2: exactly two method sentences

The agent maintains a short plan, immediate subgoal, and predicted observable outcome, and asks the model to compare the previous prediction with executed-action feedback before selecting its next action. A shared bounded-history policy supplies game evidence, while a structured parser updates the state only after a valid reply and all state resets between episodes.

This is generic at the algorithm level across the two supported games and both models, with game-specific public rules and action vocabularies. It does not demonstrate generality to every BALROG environment. By absolute progression the strongest planning cell is OpenAI BabyAI (50%); by measured improvement over its own baseline the positive cell is Qwen BabyAI (+10 points), but its fallback-assisted success makes a reasoning-benefit interpretation weak. Say which meaning of “best” you use.

### Exercise 3: exactly two method sentences

After every four completed actions, the same model is prompted as a critic to compare the planner's state with recent executed actions, rewards, and visible observations. A valid change critique replaces up to three short lessons used in subsequent decisions, while retain-status or invalid critiques leave memory unchanged and all lessons reset at the next episode.

The most favorable within-model reflection-versus-planning difference is Qwen BabyAI (+10 points, one additional successful setting). OpenAI BabyAI is higher in absolute reflective success (40% versus Qwen's 30%). Neither observation establishes robust self-reflection benefit: failures of the critic contract, sample size, extra compute, and case-specific fallback effects remain.

### Suggested report structure

1. **Introduction and research question:** explain why maintained plans and critique might help interactive reasoning; define the practical comparison.
2. **Methods:** models, cloud/self-hosted setup, benchmarks, metrics, three conditions, prompt/state limits, pairing, seed sets, repeats, and budget.
3. **Exercise 1:** adapted baseline matrix, rationale, and concrete task failures.
4. **Exercise 2:** two-sentence planning method, matrix, paired differences, parser/state brittleness, and scope of the harness.
5. **Exercise 3:** two-sentence reflection method, matrix, critic adoption/rejection counts, stability, and qualified suitability answer.
6. **Redesign comparison:** v1 versus v2 as an exploratory engineering narrative; state every major simultaneous change.
7. **Discussion and limitations:** distinguish task performance, correct formatting, usable critique, and claims about reasoning.
8. **Reproducibility and disclosure:** exact frozen source, data, analysis commands, cost, repaired packaging, and truthful AI assistance.

The students must provide their own names, course details, personal experiences, and judgments. Do not write “we manually checked every trace” unless you actually do so. A suitable disclosure starting point is: “We used OpenAI Codex to assist with implementation, debugging, data auditing, and preparation of report-writing materials.” Add the students' actual verification and writing contributions; do not claim unaided authorship or invent manual review. The supplied assignment PDF does not establish a detailed AI-use policy; check the course's applicable rules before submission.

## 14. Files to read and what each proves

| File | Purpose |
|---|---|
| [agents.py](../agents.py) | Actor/critic prompts, state updates, response contracts, and failure behavior |
| [protocol.py](../protocol.py) | Public game rules, exact actions, and normalization |
| [prompts.py](../prompts.py) | Shared evidence selection and byte-based input bound |
| [experiment.yaml](../config/experiment.yaml) | Fixed v2 model/task/seed/repeat configuration |
| [experiment_plan.py](../experiment_plan.py) | Episode counts, ordering, and conservative call bounds |
| [runner.py](../runner.py) | Environment interaction, applied actions, atomic records, provenance, and phase gating |
| [validation.py](../validation.py) | What infrastructure validation checks and what it deliberately does not judge |
| [analysis.py](../analysis.py) | Means, SD/SEM, paired bootstrap, repeat summaries, and mechanical trace selection |
| [report_audit.py](../report_audit.py) | Independent post-run table regeneration and diagnostics; no model calls |
| [package_submission.py](../package_submission.py) | Repaired packaging using the separately verified evaluated source |
| [audit_summary.json](audit_summary.json), [validation.csv](validation.csv) | Verified counts, budget totals, and successful record checks |
| [behavior_diagnostics.csv](behavior_diagnostics.csv), [parser_error_counts.csv](parser_error_counts.csv) | Primary-only failure counts; error strings can overlap in one reply |
| [episode_index.csv](episode_index.csv) | All 276 final records across versions, portable paths, scores, terminal fallback flags, and SHA256 hashes |
| [cost_and_runtime.csv](cost_and_runtime.csv) | All-final costs and manifest durations; repeats included |
| [evaluated_source_hashes.json](evaluated_source_hashes.json) | Exact execution-file hashes for both versions and models |
| [evidence README](../evidence/README.md) | How to extract all raw records, logs, and both source snapshots |

The current checkout includes post-run audit and packaging changes. It is not byte-for-byte the evaluated source. For exact reproduction, use the original source archives in `assignment2/evidence/`. V1 matches 68 execution-file hashes; v2 matches 73 execution-file hashes and all 88 files in its deployment receipt. V2's base Git commit is `9c9bd18c0a65759a288034c293bb5920f02ddb07`, but the evaluated working tree was dirty; the release content hash and file receipt identify it precisely. Do not cite that Git commit alone as the complete v2 source.

## 15. Reproduce the report tables without spending anything

From the repository root, using Python 3.10 or newer:

```bash
mkdir -p assignment2/evidence/unpacked
tar -xzf assignment2/evidence/experiment-records.tar.gz \
  -C assignment2/evidence/unpacked
python3 -m assignment2.report_audit assignment2/evidence/unpacked
```

This audit uses the standard library and the repository's local modules. It validates the four final directories and both v2 smoke directories, checks v2 initial-state pairing, reproduces saved per-model matrices, and regenerates the CSV/JSON report tables. It does not install models, contact an API, modify raw data, or rebuild figures. Original figures are preserved separately.

For each source archive and the evidence bundle, verify the digests in `assignment2/evidence/SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` on macOS, run from that directory). Individual evidence hashes are in `record_inventory.csv`. The evidence archive expands to `results/` and `logs/`, matching the portable paths in `episode_index.csv`.

The repaired ZIPs in `assignment2/submission_packages/` contain the evaluated v2 source, raw v2 evidence, analysis, and this handoff. Their `ARCHIVE_STATUS.json` explicitly states that student-report review is still required. They do not turn the handoff into the students' own report. Preserve the frozen source and original results when adding the final report.

Before submitting, verify: all three requested matrices are present; metric units and sample sizes are explicit; v1 is not pooled with v2; the fallback and critic-contract problems are disclosed; citations and AI disclosure are checked; and all three ZIPs contain the required README, scripts, prompts, and code. Complete any formatting, length, naming, and submission instructions given separately in the course/LMS.

## 16. Sources and citation guidance

Use the assignment and repository evidence for claims about this study. Cite the following primary papers for background; none is evidence that our implementation succeeded:

1. **Course assignment:** *Agentic LLM, Assignment 2: Reasoning*, Leiden University, 2 June 2026, supplied PDF, pages 3–8 for questions and evaluation guidance.
2. **Course book:** supplied *Agentic LLM* book; use its title page for the author/edition required by your citation style. Do not invent missing bibliographic metadata.
3. **Paglieri et al.** *BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games*. [arXiv:2411.13543](https://arxiv.org/abs/2411.13543), original 2024; inspected version revised April 2025. Background on evaluating interactive agents across games.
4. **Yao et al.** *ReAct: Synergizing Reasoning and Acting in Language Models*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629), 2022 preprint / ICLR 2023. Background on interleaving reasoning traces and actions.
5. **Shinn et al.** *Reflexion: Language Agents with Verbal Reinforcement Learning*. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366), 2023 arXiv version. Background on feedback-driven text memory across trials. The assignment uses a proceedings-year citation; follow the bibliographic version you actually cite.
6. **Executed code and data:** the two source archives, `v2_deployment_receipt.json`, `record_inventory.csv`, saved manifests, and raw episode JSON. The vLLM serving configuration is evidenced by `logs/vllm-22881/vllm.log` in the evidence bundle.

BibTeX starting entries for the three research papers are in [references.bib](references.bib). The short literature descriptions here are background only; the findings, failure rates, and report conclusions are derived from our recorded experiment.
