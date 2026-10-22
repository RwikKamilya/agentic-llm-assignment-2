"""Package one verified campaign with matching source; the student report is still required."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from assignment2.prepare_deployment import release_files
from assignment2.runner import execution_source_hashes, source_provenance
from assignment2.validation import validate_run_directory


ROOT = Path(__file__).resolve().parents[1]


def create_archives(campaign, source_root=ROOT, output=None, report_materials=None):
    campaign = Path(campaign).resolve()
    source_root = Path(source_root).resolve()
    source_provenance(source_root)
    expected_source = execution_source_hashes(source_root)
    for model in ['openai_mini', 'qwen_7b_local']:
        run = campaign / model / 'final'
        validate_run_directory(run)
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['source_files'] != expected_source:
            raise ValueError('Archive source differs from the evaluated release; use the frozen release.')
    # Older campaigns wrote COMPLETE after analysis, before ZIP creation. Keep
    # that evidence readable, but new launchers use a distinct analysis marker.
    if not any((campaign / name).is_file() for name in ['ANALYSIS_COMPLETE', 'COMPLETE']):
        raise ValueError('Combined analysis has not written its COMPLETE marker.')
    output = Path(output).resolve() if output else campaign / 'packages'
    output.mkdir(parents=True, exist_ok=True)
    source = release_files(source_root)
    evidence = [path for path in campaign.rglob('*') if path.is_file() and output not in path.parents
                and campaign / 'packages' not in path.parents
                and (path.suffix in {'.json', '.csv', '.md', '.png', '.pdf', '.txt'}
                     or path.name in {'COMPLETE', 'ANALYSIS_COMPLETE'})]
    materials = Path(report_materials).resolve() if report_materials else None
    if materials is not None and not materials.is_dir():
        raise ValueError('Report materials directory is missing.')
    packaging_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    archives = []
    for exercise, agent in [(1, 'naive'), (2, 'planning'), (3, 'reflective_planning')]:
        target = output / f'Ass2-{exercise}.zip'
        temporary = target.with_suffix('.zip.tmp')
        # The frozen tar receipt has mtime=0. ZIP begins at 1980, so clamp the
        # archive timestamp while retaining every source/evidence byte.
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             strict_timestamps=False) as archive:
            for path in source:
                archive.write(path, path.relative_to(source_root))
            receipt = source_root / 'assignment2/deployed_version.json'
            if receipt.exists():
                archive.write(receipt, receipt.relative_to(source_root))
            for path in evidence:
                archive.write(path, Path('results') / path.relative_to(campaign))
            if materials is not None:
                for path in sorted(materials.rglob('*')):
                    if path.is_file():
                        archive.write(path, Path('assignment2/report_materials') / path.relative_to(materials))
            archive.writestr('README.md',
                f'# Assignment 2, Exercise {exercise}\n\n'
                f'The condition for this exercise is `{agent}`. All conditions are included for reproducibility.\n\n'
                'Read assignment2/RUNBOOK.md for commands, assignment2/FINAL_EXPERIMENT.md for methods, '
                'and results/analysis/ for verified tables and trace selections. Raw evidence and effective '
                'configuration are in results/MODEL/final/.\n\n'
                'If present, assignment2/report_materials/PARTNER_GUIDE.md explains the completed results and limitations. '
                'Its full cross-version evidence links and audit commands target the Git repository handoff.\n\n'
                'The student report and honest AI-assistance disclosure still require review and inclusion.\n')
            archive.writestr('ARCHIVE_STATUS.json', json.dumps({
                'status': 'VERIFIED_EVIDENCE_PACKAGED_REPORT_REVIEW_REQUIRED', 'exercise': exercise,
                'agent': agent, 'source_files': expected_source,
                'packaging_script_sha256': packaging_hash,
                'packaging_note': 'Post-run packaging may differ from frozen experimental code. ZIP timestamps are clamped; file contents are preserved.',
                'note': 'Verified real campaign, not a claim that the student report is complete.'}, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'ZIP integrity check failed: {temporary}')
        temporary.replace(target)
        archives.append(target)
        print(target)
    return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', help='One completed v2 campaign directory, never the entire historical results root.')
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Extracted frozen release that produced these records.')
    parser.add_argument('--output', type=Path, help='Separate output directory; leaves downloaded evidence untouched.')
    parser.add_argument('--report-materials', type=Path, help='Optional reviewed handoff directory to include.')
    args = parser.parse_args()
    create_archives(args.campaign, args.source_root, args.output, args.report_materials)


if __name__ == '__main__':
    main()
