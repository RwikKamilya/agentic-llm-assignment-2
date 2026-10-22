# Handoff verification, 22 October 2026

The post-run review completed these checks without model requests:

- All 276 final episode identities validate across v1 and v2; both v2 smoke
  directories validate (12 more episodes).
- All v2 paired settings have matching initial observations and missions.
- Regenerated per-model performance matrices agree with the saved numerical
  means, counts, step counts, and estimated costs.
- The offline audit also succeeds with system Python 3.13 using no third-party
  packages. Its CSV/JSON outputs are supplied alongside this file.
- All 44 assignment unit tests pass in the prepared Python 3.10 test environment,
  including the regression for an epoch-dated deployment receipt and rejection
  of mismatched source. This is an offline test result; it does not claim that
  another real game integration run was performed during this review.
- All assignment shell scripts pass `bash -n`; `git diff --check` passes.
- The compressed raw evidence inventory matches 418 original files byte for byte.
- V1 source matches 68 execution hashes. V2 source matches 73 execution hashes
  and all 88 deployment-receipt hashes.
- All three repaired ZIPs pass CRC checks. Their archived execution source,
  deployment receipt, results, and report-material files match their input bytes.

Reproduce the table audit using the commands in [PARTNER_GUIDE.md](PARTNER_GUIDE.md).
In an environment with the project's test dependencies, run the offline tests:

```bash
python -m unittest discover -s tests -p 'test_assignment2*.py' -q
```

The original runtime logs establish the real run separately. Both model jobs
exited 0; the remote analysis/packaging job failed during ZIP creation. Passing
record validation and offline tests establishes evidence consistency, not that
model explanations are correct or that reflection improved task performance.
