# Assignment 2 requirements map

This implementation map follows the numbered questions in `Ass2Reasoning.pdf`.
It distinguishes code prepared now from evidence that must be generated during
future real runs.

| Assignment item | Implementation component | Evidence to collect later | Report location |
|---|---|---|---|
| Preparation: read BALROG, agents, evaluation, few-shot material | `RUNBOOK.md`, this map, and the preserved BALROG documentation | Installation and smoke-test notes | Methods / appendix |
| Exercise 1.1: Ass2-1 archive and README | `package_submission.py` | Archive contents and README completed with actual commands | Appendix / submitted archive |
| Exercise 1.2: LLMs and benchmarks tried, and why | Fixed Gemini and Qwen profiles in `config/experiment.yaml`; BabyAI and Crafter manifest | Availability, pilot feasibility, final model versions | Exercise 1 |
| Exercise 1.3: LLM x benchmark performance matrix and metric | `analysis.py`, `performance_matrix.csv`, figures | Completed final records only | Exercise 1 |
| Exercise 1.4: reflection on failures, benchmarks, reasoning, agent idea | `report_outline.md` prompts; selected traces | Actual failures, traces, and outcomes | Exercise 1 discussion |
| Exercise 2.1: Ass2-2 archive and README | `package_submission.py` | Archive contents and README completed with actual evidence | Appendix / submitted archive |
| Exercise 2.2: performance matrix | Same analysis pipeline, agent `planning` | Matched completed episodes | Exercise 2 |
| Exercise 2.3: two-sentence idea, generic/specific scope, best cell | `agents.py` and report placeholder | Measured best cell only | Exercise 2 |
| Exercise 2.4: LLMs/benchmarks, run count, brittleness, harness reflection | Fixed manifests, paired analysis, variability subset, trace export | Pilot/final records and qualitative inspection | Exercise 2 discussion |
| Exercise 3.1: Ass2-3 archive and README | `package_submission.py` | Archive contents and README completed with actual evidence | Appendix / submitted archive |
| Exercise 3.2: performance matrix | Same analysis pipeline, `reflective_planning` | Matched completed episodes | Exercise 3 |
| Exercise 3.3: two-sentence self-reflection description | `agents.py`; report placeholder | Description checked against final implementation | Exercise 3 |
| Exercise 3.4-3.5: focus, best benchmark, failures, stability, suitability | Repeated-run subset, paired comparisons, selected traces | Actual results and cautious conclusion | Exercise 3 discussion |
| Evaluation hints: progression, multiple runs, trace inspection | Fixed task/seed manifest, atomic episode records, analysis and trace export | Progression definitions, variance, manually inspected records | Methods / appendices |

## Design decisions

- **Baseline:** unmodified BALROG `naive` agent.
- **Planning condition:** one actor call per step, producing a bounded plan,
  subgoal, expected observable outcome, and one action.
- **Reflection condition:** the same planner plus an evidence-only critic call
  every four actor steps. It keeps at most three short lessons and resets all
  state for each episode. This is *Reflexion-inspired within-episode
  reflection*, not a reproduction of Reflexion's cross-trial episodic memory.
- **Benchmarks:** BabyAI uses BALROG's binary mission progression. Crafter
  uses its final count of achievements divided by 22. They are reported in
  separate rows and are never averaged together.
- **Comparison:** ten matched task/seed episodes are planned per
  model × benchmark × agent final cell. The small fixed-seed repeat subset
  quantifies sampling variation but is not treated as additional task samples.
- **Cost comparison:** reflection makes extra critic calls. The report must
  present calls, tokens, time, and estimated cost alongside progression. An
  equal-call-budget ablation is optional and intentionally not required for
  the main evaluation.

## Future model choice

The local profile proposes `Qwen/Qwen2.5-7B-Instruct` served by vLLM. It is a
7B instruction model, fits the requested 3-8B range, uses BALROG's existing
OpenAI-compatible vLLM client, and is a conservative fit for one allocated
48 GB L40S. This is a deployment plan, not a claim that the model is installed
or that its quality/availability has been validated on LIACS.
