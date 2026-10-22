# Assignment 2: outline for the student report

The completed [partner guide](report_materials/PARTNER_GUIDE.md) is the main
handoff. It contains the exact matrices, trace evidence, two-sentence method
descriptions, sample-size accounting, references, and answers to the assignment's
questions. This outline organizes that evidence; it is not a finished report.

## Abstract and research question

Ask whether a maintained plan improves an action-only agent, and whether periodic
critique adds a useful benefit. State the two models, two benchmarks, and three
procedures. Summarize mixed results and the response-contract failures rather
than claiming a general improvement in reasoning.

## Methods

- GPT-4o mini (`gpt-4o-mini-2024-07-18`) and self-hosted Qwen2.5-7B-Instruct.
- BabyAI mission success and Crafter distinct achievements divided by 22;
  keep these metrics separate.
- Adapted action-only baseline, stateful planner, and planner with the same
  model acting as critic every four actions. No neural training or cross-trial memory.
- V2: ten primary BabyAI settings and five Crafter worlds per model/agent,
  totaling 90 primary observations; four settings have three observations each,
  adding 48 extras. Twelve smoke episodes are excluded from final scores.
- Shared rules, evidence-selection policy, action normalization, and fallback;
  64 BabyAI actions and 192 Crafter actions; temperature 0.2, output limit 192 tokens.
- Matched seeds within v2; state reset each episode; extra critic calls are an
  uncontrolled compute difference. Include recorded Qwen sampling defaults.

## Exercise 1: adapted baseline

Use the first matrix in guide section 6: OpenAI BabyAI 50%, Crafter 10%; Qwen
BabyAI 10%, Crafter 13.64%. Explain the cloud/self-hosted comparison and the
contrast between instruction-following grid missions and survival/crafting.
Discuss failures and what the scores can actually say about reasoning.
Disclose that this is not an exact reproduction of the BALROG paper's protocol.

## Exercise 2: planning

Use the exactly two-sentence method description in guide section 13. Insert the
planning matrix and paired planning-minus-baseline comparisons. OpenAI ties its
baseline on both benchmarks; Qwen gains 10 percentage points on BabyAI and loses
5.45 points on Crafter. One of Qwen's two primary BabyAI successes ends on a
fallback action. Discuss parser-state failure loops and validly formatted but
poorly grounded actions using the indexed traces.

## Exercise 3: reflective planning

Use the second two-sentence method description in guide section 13. Insert the
reflective matrix and comparisons with planning. Across primary observations,
464 of 527 critic replies are rejected; only 57 are accepted change critiques.
Explain why this weakens a test of the intended reflection mechanism. Contrast
Qwen's one additional BabyAI success with its lower Crafter scores, and describe
the three-observation repeat evidence without overstating statistical stability.

## Previous-run comparison

Use `report_materials/previous_vs_current.csv` and guide section 7. Ten of twelve
cell means increase, one ties, and one decreases. Seeds, prompts, response
contracts, history selection, and the Crafter horizon changed together; sample
allocation also changed. Describe this as an exploratory engineering comparison,
not a causal estimate of any one repair. Never pool v1 and v2 as one experiment.

## Discussion and limitations

Discuss small selected samples, fallback effects, strict parsing, critic adoption,
extra compute, short horizons, correlated actor/critic errors, and provider
sampling differences. Report exploratory paired intervals with their denominators;
do not treat the 48 extra repeated observations as independent new settings.
Negative results and harness failures are substantive findings for this assignment.

## Reproducibility, references, and student reflection

Point to frozen source hashes, raw records, audit commands, and repaired archives.
V2 OpenAI smoke plus final estimated cost is $0.77817120; local compute is not free
merely because its external API charge is zero. Explain that both model jobs
completed but the remote packaging step failed and was repaired after the run.
Use `report_materials/references.bib` for primary-paper starting entries.

Add the students' real experiences, names, course details, verification work, and
honest AI-assistance disclosure under the applicable course policy. Check any
separate LMS formatting and submission requirements. Review the report before
adding it to the exercise ZIPs; the supplied handoff is not a claim of unaided
student authorship.
