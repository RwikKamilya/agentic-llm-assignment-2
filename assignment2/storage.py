"""Small helpers for atomic episode records and compatible-resume checks."""

import json
import tempfile
from pathlib import Path


def _json_safe(value):
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def matching_completed_record(episodes_directory, episode_key, config_fingerprint):
    """Return a completed record only when it has the same effective config."""
    for record_path in Path(episodes_directory).glob(f"{episode_key}_attempt-*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("status") == "completed" and record.get("config_fingerprint") == config_fingerprint:
            return record_path
    return None


def next_attempt_path(episodes_directory, episode_key):
    attempts = []
    for path in Path(episodes_directory).glob(f"{episode_key}_attempt-*.json"):
        suffix = path.stem.rsplit("-", 1)[-1]
        if suffix.isdigit():
            attempts.append(int(suffix))
    return Path(episodes_directory) / f"{episode_key}_attempt-{max(attempts, default=0) + 1:02d}.json"
