# Preserved experimental evidence

Start with [the partner guide](../report_materials/PARTNER_GUIDE.md). These archives
preserve actual observations, including unsuccessful games. They contain no model
weights or API credentials. The original downloaded backup remains local and is
not duplicated wholesale in Git.

| File | Contents |
|---|---|
| `experiment-records.tar.gz` | 418 original files: both v1 final runs, their smoke runs, the complete v2 campaign except its failed partial ZIP, downloaded logs, and cumulative budget ledger/backups |
| `v1-source-9c9bd18.tar.gz` | Git-tracked `assignment2/`, `balrog/`, and assignment tests from the v1 source commit, under `v1-source/` |
| `ass2-v2-09afb9fbcc46.tar.gz` | Original, unchanged frozen v2 deployment; extracted root has this release name |
| `v2_deployment_receipt.json` | Original receipt extracted from the v2 tar; base commit, dirty-tree flag, content ID, and 88 file hashes |
| `record_inventory.csv` | Relative path, byte count, and SHA256 of every file in the evidence bundle |
| `SHA256SUMS` | SHA256 digests for all three archives |

The v1 source archive matches all 68 execution hashes in its final manifests.
The v2 source archive matches all 73 execution hashes in its final manifests and
all 88 files in its deployment receipt. File content, not archive timestamps,
identifies the evaluated code. The current repository also contains clearly
separated post-run audit and packaging repairs.

From this directory on macOS, verify compressed archives:

```bash
shasum -a 256 -c SHA256SUMS
```

From the repository root, extract and recompute report tables without any API calls:

```bash
mkdir -p assignment2/evidence/unpacked
tar -xzf assignment2/evidence/experiment-records.tar.gz -C assignment2/evidence/unpacked
python3 -m assignment2.report_audit assignment2/evidence/unpacked
```

Python 3.10+ and the local repository modules are sufficient for this audit;
it does not need the game engines, model SDKs, or matplotlib. Extracted data are
ignored by Git to avoid duplicating the compressed evidence.

Inspect source separately:

```bash
tar -xzf assignment2/evidence/v1-source-9c9bd18.tar.gz -C assignment2/evidence/unpacked
tar -xzf assignment2/evidence/ass2-v2-09afb9fbcc46.tar.gz -C assignment2/evidence/unpacked
```

To rebuild the submission ZIPs using an existing BALROG Python environment
(the packager imports the project's OmegaConf dependency):

```bash
python -m assignment2.package_submission \
  assignment2/evidence/unpacked/results/ass2-v2-09afb9fbcc46 \
  --source-root assignment2/evidence/unpacked/ass2-v2-09afb9fbcc46 \
  --output assignment2/submission_packages \
  --report-materials assignment2/report_materials
```

Never use a newly modified experiment source tree as if it produced the old
results. The packager rejects source-hash mismatches. These commands perform
post-processing only; they do not restart or resume a paid experiment.

## Completion and packaging qualification

The preserved logs show both model jobs exited 0. The analysis job failed during
ZIP creation because the receipt timestamp predates 1980. The original `COMPLETE`
marker was written before that failure. It establishes neither that all three
ZIPs existed nor that the remote job succeeded. The failed partial ZIP is excluded
from this curated bundle; it remains in the unmodified local calcium backup.

Repaired ZIPs are separately located in `assignment2/submission_packages/`.
Their status still requires the students' report review. Neither the original
logs nor the experimental scores have been rewritten to hide the failure.
