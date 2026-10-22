# Report handoff

Read **[PARTNER_GUIDE.md](PARTNER_GUIDE.md)** first. It explains the assignment,
implementation, results, previous-run comparison, failure mechanisms, uncertainty,
and how to write each required report section.

- `v1_*` and `v2_*` CSVs keep the two versions separate. Performance matrices use
  primary observations; repeat tables do not increase the primary sample size.
- `previous_vs_current.csv` is an unpaired descriptive comparison, not a causal
  estimate of any one repair.
- `behavior_diagnostics.csv` and `parser_error_counts.csv` quantify primary-only
  formatting, critic-adoption, fallback, and history-trimming behavior. Blank
  v1 `CHANGE` fields are not applicable to its different critic contract.
- `episode_index.csv` indexes all 276 final records across versions, including
  repeated observations. `primary` identifies the records used for the matrices.
- `audit_summary.json`, `validation.csv`, and `evaluated_source_hashes.json`
  preserve the independently checked counts and provenance.
- `cost_and_runtime.csv` includes all final episodes and repeats, excluding smoke.
- PDF/PNG figures are copied unchanged from the successful combined analysis.
- `references.bib` provides primary-paper bibliography entries to check against
  the course's required citation style.

Raw data, logs, budgets, and exact sources are in [../evidence/](../evidence/README.md).
Repaired exercise archives are in [../submission_packages/](../submission_packages/README.md).
The student report, genuine personal reflection, and final disclosure still need
to be written and reviewed by the students.
