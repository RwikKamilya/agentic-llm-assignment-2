"""Validate infrastructure and every expected episode; game success is not required."""

import json
import math
from pathlib import Path


class IncompletePhaseError(RuntimeError):
    """A phase cannot be used as complete experimental evidence."""


def completed_record_problems(record, expected, phase):
    problems = []
    protocol_v2 = expected.get("schema_version", 1) >= 2
    if protocol_v2:
        for field in ["schema_version", "protocol_id"]:
            if record.get(field) != expected.get(field):
                problems.append(f"mismatched {field}")
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
        raw_present = isinstance(actor.get("raw_response"), str) if protocol_v2 else bool(actor.get("raw_response"))
        if step.get("step") != index or not raw_present or not actor.get("call"):
            problems.append("missing actor trace or nonsequential step")
        parsed = actor.get("parse", {})
        if parsed.get("valid") and str(step.get("requested_action", "")).strip():
            usable += 1
        reflection = trace.get("reflection")
        if reflection:
            reflections += 1
            raw_present = isinstance(reflection.get("raw_response"), str) if protocol_v2 else bool(reflection.get("raw_response"))
            if not raw_present or not reflection.get("call"):
                problems.append("missing critic trace")
            if not protocol_v2 and not reflection.get("parse", {}).get("valid"):
                problems.append("invalid critic response format")
        if protocol_v2:
            from assignment2.protocol import ACTIONS, normalize_action
            benchmark = expected["spec"]["benchmark"]
            if step.get("normalized_action") != normalize_action(step.get("requested_action"), benchmark):
                problems.append("incorrect action normalization")
            if step.get("applied_action") not in ACTIONS[benchmark]:
                problems.append("unrecognized applied action")
            invalid = step.get("normalized_action") != step.get("applied_action")
            if step.get("invalid_action") != invalid:
                problems.append("incorrect invalid-action flag")
            feedback = trace.get("environment", {})
            for key in ["applied_action", "invalid_action", "reward", "terminated", "truncated"]:
                if feedback.get(key) != step.get(key):
                    problems.append("environment feedback differs from saved transition")
            if not isinstance(step.get("observation_after"), dict):
                problems.append("missing next observation")
            if index and step.get("observation") != steps[index - 1].get("observation_after"):
                problems.append("observation transition chain is broken")
    if reflections != expected_critics:
        problems.append("persisted critic traces do not match calls")
    if not protocol_v2 and not usable:
        problems.append("no usable actor output in episode")
    if protocol_v2:
        from assignment2.prompts import input_token_upper_bound
        from balrog.prompt_builder.history import Message
        traces = record.get("model_call_traces", [])
        if len(traces) != len(calls):
            problems.append("missing exact model requests")
        for trace, call in zip(traces, calls):
            if trace.get("call") != call or trace.get("role") != call.get("role"):
                problems.append("model request and usage records differ")
            messages = trace.get("messages", [])
            if not messages or input_token_upper_bound([Message(**message) for message in messages]) > expected["effective_config"]["client"]["max_input_tokens"]:
                problems.append("missing or oversized saved prompt")
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
