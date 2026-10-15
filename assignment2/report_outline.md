# Assignment 2 report outline

All bracketed evidence placeholders must be replaced only after real runs.

## Abstract

- Research question: Does a short maintained plan improve game-playing
  progression, and does feedback-driven reflection add benefit relative to its
  extra calls and cost?
- [State models, benchmarks, final sample sizes, and results only after final analysis.]

## Introduction and research question

- Motivate long-horizon, observation-action reasoning in BALROG.
- State the planned comparison: naive vs planning vs reflective planning.

## Background

- BALROG and the two selected benchmarks.
- Planning, bounded memory, and Reflexion-inspired verbal feedback.
- Clarify that the reflective agent is not an exact reproduction of Reflexion.

## Experimental setup and metrics

- Models: [actual Gemini version] and [actual local model/version/quantization].
- Hosted versus local execution; actual resource allocation.
- BabyAI progression is binary mission completion. Crafter progression is the
  achieved count divided by 22. They are not combined into one average.
- Matched task/seed lists, temperature, prompt fingerprint, one-worker setup,
  and pilot/final separation.
- Cost accounting and the reflection extra-compute confound.

## Exercise 1: BALROG naive baseline

- Required two-sentence method description: [pending actual final settings].
- LLM and benchmark rationale: [pending pilot feasibility].
- Performance matrix: [insert final table].
- Selected traces/failure cases: [insert inspected evidence].
- Reflection: [pending real evidence].

## Exercise 2: planning agent

- Two-sentence method description: The planning agent maintains a short plan,
  immediate subgoal, and expected observable outcome. At each step it updates
  those fields from the current observation and bounded action-observation
  history, then emits one action.
- Generic scope: the same parser and prompt are used for BabyAI and Crafter;
  no benchmark-specific reference solution is supplied.
- Performance matrix and paired planning-minus-naive differences: [pending].
- Brittleness, sample count, failures, and harness reflection: [pending].

## Exercise 3: reflective planning agent

- Two-sentence method description: The reflective variant is the same planning
  agent plus a fixed-interval critic call over observed evidence. The critic
  stores at most three short corrective lessons for later decisions and all
  plan/reflection state resets per episode.
- Performance matrix and paired reflection-minus-planning differences: [pending].
- Repeated-run stability on the fixed task/seed subset: [pending].
- Failure cases and suitability conclusion: [pending; do not infer benefit
  merely because reflection was enabled].

## Discussion, limitations, and reproducibility

- Small-sample uncertainty, task variation versus repeated-run variability,
  API availability/pricing, local resource allocation, and reflection's
  extra-compute confound.
- Exact commit, effective configuration, prompt fingerprints, archives, and
  run commands.
- AI assistance disclosure, following the course policy: [complete honestly].

## References and appendices

- BALROG, Reflexion, ReAct, course book, and actual tool/model documentation.
- Effective config, task/seed manifest, complete matrices, traces, and archive
  inventory.
