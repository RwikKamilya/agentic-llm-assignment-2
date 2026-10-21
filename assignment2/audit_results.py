"""Summarize saved evidence without exposing prompts, responses, or credentials."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def audit_results(root):
    root = Path(root)
    counts = Counter()
    timings = defaultdict(list)
    unreadable = []
    synthetic = 0
    completed_identities = set()
    duplicate_completed = 0
    paths = set(root.glob('**/episodes/*.json'))
    paths.update(root.glob('synthetic_results/*.json'))
    for path in sorted(paths):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            unreadable.append(str(path))
            continue
        if record.get('data_kind') == 'synthetic' or 'synthetic_results' in path.parts:
            synthetic += 1
            continue
        spec = record.get('spec', {})
        group = (record.get('phase', record.get('data_kind', 'unknown')), spec.get('model', 'unknown'),
                 spec.get('benchmark', 'unknown'), spec.get('agent', 'unknown'))
        status = record.get('status', 'unknown')
        counts[group + (status,)] += 1
        if status == 'completed':
            identity = record.get('episode_key') or json.dumps(spec, sort_keys=True)
            if identity in completed_identities:
                duplicate_completed += 1
                continue
            completed_identities.add(identity)
            elapsed = record.get('elapsed_seconds')
            if isinstance(elapsed, (int, float)) and elapsed >= 0:
                timings[group].append(elapsed)
    return {'root': str(root), 'episode_record_counts': [
                dict(zip(['phase','model','benchmark','agent','status'], group), count=count)
                for group, count in sorted(counts.items())],
            'explicit_synthetic_records': synthetic, 'unreadable_records': unreadable,
            'duplicate_completed_identities': duplicate_completed,
            'timings': [dict(zip(['phase','model','benchmark','agent'], group), n=len(values),
                             total_seconds=sum(values), mean_seconds=sum(values)/len(values))
                        for group, values in sorted(timings.items())]}


if __name__ == '__main__':
    print(json.dumps(audit_results(sys.argv[1]), indent=2))
