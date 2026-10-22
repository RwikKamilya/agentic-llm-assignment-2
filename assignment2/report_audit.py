"""Recompute the partner's report tables from preserved records, without model calls.

Usage: python -m assignment2.report_audit assignment2/evidence/unpacked
The input contains results/ and logs/ copied from calcium. Output is separate
from the immutable downloaded records. No scores or experimental code are edited.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from assignment2.analysis import (load_records, paired_differences,
                                 repeated_run_variability, summary_rows, task_rows,
                                 validate_paired_resets, write_csv)
from assignment2.protocol import normalize_action, ACTIONS
from assignment2.validation import validate_run_directory


RELEASE = 'ass2-v2-09afb9fbcc46'
MODELS = ['openai_mini', 'qwen_7b_local']


def primary(records):
    return [record for record in records if 'variability_subset' not in record['spec']]


def cell(record):
    return tuple(record['spec'][key] for key in ['model', 'benchmark', 'agent'])


def behavior_tables(records, version):
    groups = defaultdict(list)
    for record in primary(records):
        groups[cell(record)].append(record)
    rows, error_rows = [], []
    for key, items in sorted(groups.items()):
        model, benchmark, agent = key
        steps = [step for record in items for step in record['steps']]
        actors = [step['agent_trace']['actor'] for step in steps]
        critics = [step['agent_trace']['reflection'] for step in steps
                   if step['agent_trace'].get('reflection')]
        calls = [call for record in items for call in record['usage']['calls']]
        successful = [record for record in items if record['outcome']['success']]
        fallback_successes = sum(record['steps'][-1]['invalid_action'] for record in successful)
        rows.append({
            'version': version, 'model': model, 'benchmark': benchmark, 'agent': agent,
            'scope': 'primary settings only', 'n_episodes': len(items), 'actor_calls': len(actors),
            'actor_parse_failures': sum(not actor['parse']['valid'] for actor in actors),
            'invalid_actions': sum(step['invalid_action'] for step in steps),
            'invalid_strings_accepted_by_v2_normalizer': sum(
                step['invalid_action'] and normalize_action(step['requested_action'], benchmark) in ACTIONS[benchmark]
                for step in steps),
            'critic_calls': len(critics),
            'critic_parse_failures': sum(not critic['parse']['valid'] for critic in critics),
            'accepted_CHANGE_critiques': sum(critic['parse']['valid'] and critic['parse'].get('status') == 'CHANGE'
                                           for critic in critics) if version == 'v2' else None,
            'episodes_with_any_accepted_CHANGE': sum(any(
                step['agent_trace'].get('reflection') and step['agent_trace']['reflection']['parse']['valid']
                and step['agent_trace']['reflection']['parse'].get('status') == 'CHANGE'
                for step in record['steps']) for record in items) if version == 'v2' else None,
            'successful_episodes': len(successful),
            'successful_episodes_ending_on_fallback': fallback_successes,
            'successful_episodes_ending_on_valid_action': len(successful) - fallback_successes,
            'calls_with_history_trimming': sum(call.get('prompt', {}).get('trimmed_history_messages', 0) > 0
                                             for call in calls),
            'calls_with_empty_response': sum(call['status'] == 'empty_response' for call in calls),
            'output_limit_stops': sum(call['status'] == 'length' for call in calls),
            'retries': sum(call.get('retries', 0) for call in calls),
        })
        for role, traces in [('actor', actors), ('critic', critics)]:
            counts = Counter(error for trace in traces for error in trace['parse'].get('errors', []))
            for error, count in sorted(counts.items()):
                error_rows.append({'version': version, 'model': model, 'benchmark': benchmark, 'agent': agent,
                                   'role': role, 'error': error, 'count': count,
                                   'note': 'Error counts can overlap within a reply; denominator is calls of this role.'})
    return rows, error_rows


def verify_saved_matrix(run, records):
    """Check regenerated numeric cells against the analysis saved on calcium."""
    path = run / 'analysis/performance_matrix.csv'
    saved = list(csv.DictReader(path.open()))
    rebuilt = summary_rows(records)
    keys = ['model', 'benchmark', 'agent']
    by_key = {tuple(row[key] for key in keys): row for row in saved}
    if len(saved) != len(rebuilt):
        raise ValueError(f'Matrix row count changed: {path}')
    for row in rebuilt:
        other = by_key[tuple(row[key] for key in keys)]
        for field in ['n_completed', 'mean_progression', 'mean_steps', 'estimated_cost_usd']:
            if not math.isclose(float(other[field]), float(row[field]), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'Saved matrix differs: {path}, {field}')


def audit(evidence_root, output):
    evidence_root, output = Path(evidence_root), Path(output)
    root = evidence_root / 'results'
    campaign = root / RELEASE
    runs_by_version = {
        'v1': [root / 'final_slurm-22876', root / 'final_slurm-22874'],
        'v2': [campaign / model / 'final' for model in MODELS],
    }
    all_records, all_rows, behavior, errors, validation_rows, costs = {}, {}, [], [], [], []
    source_hashes = {}
    for version, runs in runs_by_version.items():
        for run in runs:
            validation = validate_run_directory(run)
            validation_rows.append({'version': version, 'run': str(run.relative_to(evidence_root)),
                                    **{key: value for key, value in validation.items() if key != 'validated_paths'}})
        records, _ = load_records(runs, 'final')
        all_records[version] = records
        validate_paired_resets(records)
        all_rows[version] = summary_rows(records)
        write_csv(output / f'{version}_performance_matrix.csv', all_rows[version])
        write_csv(output / f'{version}_task_breakdown.csv', task_rows(records))
        paired = []
        for left, right in [('planning', 'naive'), ('reflective_planning', 'planning'),
                            ('reflective_planning', 'naive')]:
            paired.extend(paired_differences(records, left, right))
        write_csv(output / f'{version}_paired_differences.csv', paired)
        write_csv(output / f'{version}_repeated_run_variability.csv', repeated_run_variability(records))
        rows, error_rows = behavior_tables(records, version)
        behavior.extend(rows)
        errors.extend(error_rows)
        for run, model in zip(runs, MODELS):
            model_records = [record for record in records if record['spec']['model'] == model]
            verify_saved_matrix(run, model_records)
            manifest = json.loads((run / 'manifest.json').read_text())
            source_hashes[f'{version}/{model}'] = manifest['source_files']
            calls = [call for record in model_records for call in record['usage']['calls']]
            duration = (datetime.fromisoformat(manifest['completed_at'])
                        - datetime.fromisoformat(manifest['created_at'])).total_seconds()
            costs.append({'version': version, 'model': model, 'scope': 'all final episodes including repeats',
                          'episodes': len(model_records), 'calls': len(calls),
                          'actor_calls': sum(call['role'] == 'actor' for call in calls),
                          'critic_calls': sum(call['role'] == 'critic' for call in calls),
                          'input_tokens': sum(call.get('input_tokens') or 0 for call in calls),
                          'output_tokens': sum(call.get('output_tokens') or 0 for call in calls),
                          'estimated_api_cost_usd': sum(record['usage']['episode_estimated_cost_usd'] for record in model_records),
                          'final_phase_elapsed_seconds': duration,
                          'note': 'Provider invoice unavailable. Qwen API charge zero excludes hardware cost. Elapsed includes local bookkeeping.'})
    # Smoke remains separate from the final matrices.
    smoke = []
    for model in MODELS:
        run = campaign / model / 'smoke'
        validated = validate_run_directory(run)
        validation_rows.append({'version': 'v2-smoke', 'run': str(run.relative_to(evidence_root)),
                                **{key: value for key, value in validated.items() if key != 'validated_paths'}})
        smoke.extend(load_records([run], 'smoke')[0])
    write_csv(output / 'validation.csv', validation_rows)
    write_csv(output / 'behavior_diagnostics.csv', behavior)
    write_csv(output / 'parser_error_counts.csv', errors)
    write_csv(output / 'cost_and_runtime.csv', costs)
    previous = {(row['model'], row['benchmark'], row['agent']): row for row in all_rows['v1']}
    comparison = []
    for row in all_rows['v2']:
        old = previous[(row['model'], row['benchmark'], row['agent'])]
        comparison.append({key: row[key] for key in ['model', 'benchmark', 'agent']} | {
            'v1_n': old['n_completed'], 'v2_n': row['n_completed'],
            'v1_progression_percent': 100 * old['mean_progression'],
            'v2_progression_percent': 100 * row['mean_progression'],
            'descriptive_change_percentage_points': 100 * (row['mean_progression'] - old['mean_progression']),
            'v1_step_limit': old['evaluation_step_limit'], 'v2_step_limit': row['evaluation_step_limit'],
            'interpretation': 'Unpaired descriptive comparison: different seeds, prompts, parser, evidence, and Crafter horizon/sample size.'})
    write_csv(output / 'previous_vs_current.csv', comparison)
    identities = {}
    for version, records in all_records.items():
        identities[version] = {(record['spec']['benchmark'], record['spec']['task'], record['spec']['seed'])
                               for record in primary(records)}
    if identities['v1'] & identities['v2']:
        raise ValueError('Previous and current primary settings overlap.')
    ledger = json.loads((root / 'api_spending_ledger.json').read_text())
    overview = {
        'release': RELEASE, 'v1_final_episodes': len(all_records['v1']), 'v2_final_episodes': len(all_records['v2']),
        'v1_primary_episodes': len(primary(all_records['v1'])), 'v2_primary_episodes': len(primary(all_records['v2'])),
        'v2_smoke_episodes': len(smoke), 'v2_smoke_cost_usd': sum(r['usage']['episode_estimated_cost_usd'] for r in smoke),
        'v2_final_cost_usd': sum(r['usage']['episode_estimated_cost_usd'] for r in all_records['v2']),
        'cumulative_ledger_estimated_spend_usd': ledger['estimated_spent_usd'],
        'cumulative_ledger_outstanding_reservation_usd': ledger['reserved_usd'],
        'primary_setting_overlap_between_versions': 0, 'v2_initial_observation_pairing': 'passed',
        'saved_per_model_matrices_recomputed': 'passed',
        'warning': 'Infrastructure validation does not validate the truth of model reasoning or prove a beneficial reflection mechanism.',
    }
    (output / 'audit_summary.json').write_text(json.dumps(overview, indent=2) + '\n')
    (output / 'evaluated_source_hashes.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    index = []
    for version, records in all_records.items():
        for record in records:
            path = Path(record['_path'])
            spec = record['spec']
            index.append({'version': version, **spec, 'primary': 'variability_subset' not in spec,
                          'episode_key': record['episode_key'], 'progression': record['outcome']['progression'],
                          'steps': len(record['steps']), 'invalid_actions': record['outcome']['invalid_actions'],
                          'terminal_action_was_fallback': record['steps'][-1]['invalid_action'],
                          'path': str(path.relative_to(evidence_root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    # Include the optional repeat column even in the first, primary record.
    for row in index:
        row.setdefault('variability_subset', '')
    write_csv(output / 'episode_index.csv', index)
    print(json.dumps(overview, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('assignment2/report_materials'))
    arguments = parser.parse_args()
    audit(arguments.evidence_root, arguments.output)
