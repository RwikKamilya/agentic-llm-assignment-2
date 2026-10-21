"""Validate infrastructure and every expected episode; game success is not required."""

import json
import math
from pathlib import Path


class IncompletePhaseError(RuntimeError):
    """A phase cannot be used as complete experimental evidence."""


def completed_record_problems(record, expected, phase):
    problems = []
    for field in ["episode_key", "config_fingerprint", "spec", "effective_config", "model_profile",
                  "source_commit", "source_files", "runtime_versions", "local_model_snapshot"]:
        if record.get(field) != expected.get(field):
            problems.append(f"mismatched {field}")
    if record.get("phase") != phase or record.get("data_kind") != phase:
        problems.append("wrong phase or synthetic data")
    if record.get("status") != "completed":
        return problems + ["episode is not completed"]
    steps = record.get("steps", [])
    if not steps or record.get("outcome", {}).get("steps") != len(steps):
        return problems + ["missing steps or outcome"]
    limit = expected["effective_config"]["evaluation"]["max_steps_per_episode"]
    if len(steps) > limit:
        problems.append("episode exceeds effective step limit")
    progression = record.get("outcome", {}).get("progression")
    if not isinstance(progression, (int, float)) or not math.isfinite(progression) or not 0 <= progression <= 1:
        problems.append("missing or invalid benchmark progression")
    calls = record.get("usage", {}).get("calls", [])
    actors = [call for call in calls if call.get("role") == "actor"]
    critics = [call for call in calls if call.get("role") == "critic"]
    if len(actors) != len(steps):
        problems.append("actor call count differs from persisted steps")
    interval = expected["effective_config"]["agent"]["reflection_interval"]
    expected_critics = (len(steps) - 1) // interval if expected["spec"]["agent"] == "reflective_planning" else 0
    if len(critics) != expected_critics:
        problems.append("critic calls do not match the reflection schedule")
    usable = 0
    reflections = 0
    for index, step in enumerate(steps):
        trace = step.get("agent_trace", {})
        actor = trace.get("actor", {})
        if step.get("step") != index or not actor.get("raw_response") or not actor.get("call"):
            problems.append("missing actor trace or nonsequential step")
        parsed = actor.get("parse", {})
        if parsed.get("valid") and str(step.get("requested_action", "")).strip():
            usable += 1
        reflection = trace.get("reflection")
        if reflection:
            reflections += 1
            if not reflection.get("raw_response") or not reflection.get("call"):
                problems.append("missing critic trace")
            if not reflection.get("parse", {}).get("valid"):
                problems.append("invalid critic response format")
    if reflections != expected_critics:
        problems.append("persisted critic traces do not match calls")
    if not usable:
        problems.append("no usable actor output in episode")
    paid = expected["model_profile"]["client_name"] == "openai"
    for call in calls:
        if not 1 <= call.get("attempts", 0) <= 2:
            problems.append("call attempt count outside 1..2")
        if paid and (not call.get("usage_available") or
                     not all(type(call.get(key)) is int and call[key] >= 0
                             for key in ["input_tokens", "output_tokens"])):
            problems.append("OpenAI token usage was not recorded")
        if paid and call.get("provider") != "openai":
            problems.append("OpenAI call missing paid provider identity")
        if paid and (call.get("input_tokens", 0) or 0) > expected["effective_config"]["client"]["max_input_tokens"]:
            problems.append("reported input tokens exceed enforced bound")
        if (call.get("output_tokens", 0) or 0) > expected["effective_config"]["client"]["generate_kwargs"]["max_tokens"]:
            problems.append("reported output tokens exceed bound")
        if str(call.get("status", "")).startswith(("error", "exception")):
            problems.append("failed call in completed episode")
    return sorted(set(problems))


def validate_run_directory(run_directory):
    directory = Path(run_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = manifest.get("expected_episodes")
    if not expected:
        raise IncompletePhaseError("Run manifest has no complete expected episode identities.")
    by_key = {item["episode_key"]: item for item in expected}
    if len(by_key) != len(expected):
        raise IncompletePhaseError("Duplicate expected episode identities.")
    found = {}
    errors = []
    invalid_actions = 0
    parser_failures = 0
    for path in sorted((directory / "episodes").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            errors.append(f"Unreadable record: {path.name}")
            continue
        key = record.get("episode_key")
        if key not in by_key:
            errors.append(f"Unexpected episode identity: {path.name}")
            continue
        if record.get("status") != "completed":
            continue  # An earlier interrupted attempt does not replace a completed retry.
        problems = completed_record_problems(record, by_key[key], manifest["phase"])
        if problems:
            errors.append(f"{path.name}: {', '.join(problems)}")
            continue
        if key in found:
            errors.append(f"Duplicate completed identity: {key}")
        found[key] = str(path)
        invalid_actions += sum(bool(step.get("invalid_action")) for step in record["steps"])
        parser_failures += sum(not step["agent_trace"]["actor"]["parse"].get("valid") for step in record["steps"])
    missing = sorted(set(by_key) - set(found))
    if missing:
        errors.append(f"Missing {len(missing)} of {len(expected)} expected completed episodes: {', '.join(missing)}")
    if errors:
        raise IncompletePhaseError("\n".join(errors))
    return {"expected": len(expected), "completed": len(found), "invalid_actions": invalid_actions,
            "parser_failures": parser_failures, "validated_paths": list(found.values())}
