# Repaired exercise archives

These are the three code/evidence packages requested by the assignment:

- [Ass2-1.zip](Ass2-1.zip): action-only baseline.
- [Ass2-2.zip](Ass2-2.zip): planning agent.
- [Ass2-3.zip](Ass2-3.zip): reflective planning agent.

Each includes the same frozen evaluated v2 source, its deployment receipt, all v2
raw final/smoke records, analysis, and report-writing materials. Its README and
`ARCHIVE_STATUS.json` identify the exercise condition. Including all conditions
allows the comparisons to be checked. The full cross-version data, logs, and
post-run audit tool are available in the Git repository's
[evidence](../evidence/README.md) and [report materials](../report_materials/README.md).

**The students still need to write and review their report.** Read
[PARTNER_GUIDE.md](../report_materials/PARTNER_GUIDE.md), complete the students'
own reflection and disclosure, and follow any separate course submission rules.
Preserve the original source and results if adding the final report to these ZIPs.

The remote packaging job failed because ZIP cannot represent the frozen receipt's
1970 timestamp. The post-run packager clamps ZIP timestamps to the supported range,
retains file contents, verifies the evaluated source, checks CRC integrity, and
replaces each output only after it is complete. No game or model call was rerun.
The original failure logs and source are preserved in the evidence bundle.

`SHA256SUMS` identifies these supplied archives. From this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

If you add the final report, the ZIP checksums will change; record new checksums
for the submitted copies. Rebuild commands are in the evidence README.
