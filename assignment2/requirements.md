# Assignment 2 requirements map

This implementation map follows the numbered questions in `Ass2Reasoning.pdf`.
The campaign is now complete: [report_materials/PARTNER_GUIDE.md](report_materials/PARTNER_GUIDE.md)
maps every question to the verified evidence, explains the remaining validity
limitations, and provides the three required matrices. The table below is the
original implementation/evidence plan; its future-tense entries are superseded
by that completed-run guide. The student report still requires human authorship
and review.

| Assignment item | Implementation component | Evidence to collect later | Report location |
|---|---|---|---|
| Preparation: read BALROG, agents, evaluation, few-shot material | `RUNBOOK.md`, this map, and the preserved BALROG documentation | Installation and smoke-test notes | Methods / appendix |
| Exercise 1.1: Ass2-1 archive and README | `package_submission.py` | Archive contents and README completed with actual commands | Appendix / submitted archive |
| Exercise 1.2: LLMs and benchmarks tried, and why | Fixed OpenAI mini and Qwen profiles in `config/experiment.yaml`; BabyAI and Crafter manifest | Availability, pilot feasibility, final model versions | Exercise 1 |
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

## Final design

The current, frozen protocol is described in `FINAL_EXPERIMENT.md` and
`config/experiment.yaml`. It uses an adapted action-only baseline with shared
corrected rules and normalization; an outcome-checking planner; and the same
planner with a critic every four completed actions. Reflection is within-episode,
not cross-trial learning.

Each model/agent has ten primary BabyAI settings at 64 steps and five primary
Crafter worlds at 192 steps. Four existing settings have two additional samples,
yielding three observations each; these extras never inflate primary sample size.
The two models are GPT-4o mini and local Qwen2.5-7B-Instruct. All six main cells per
model, repeats, source/configuration records, costs and uncertainty must be reported.

The earlier completed experiment informed this redesign. Keep it labeled as the
earlier exploratory version. New final evidence must come from the new release's
results directory. Model availability and runtime integration are checked inside
the allocated campaign, not inferred from a test using synthetic responses.

An equal-compute ablation is outside this budget. Report the extra calls/tokens/time
of reflection, and interpret results as comparisons of whole agent procedures.
Actual results and appropriately qualified claims are now available in the
completed-run guide; no general improvement from reflection is established.
