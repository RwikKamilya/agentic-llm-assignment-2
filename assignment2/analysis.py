"""Analysis for real Assignment 2 episode records.

This module intentionally reads only completed records from one data kind at a
time. It never fills missing or failed episodes with zero progression.
"""

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from assignment2.validation import validate_run_directory


PROGRESSION_DEFINITIONS = {
    "babyai": "Binary mission completion: BALROG records progression 1.0 after a positive reward, otherwise 0.0.",
    "crafter": "The fraction of 22 Crafter achievements completed at the end of the episode.",
}


def _mean(values):
    return sum(values) / len(values) if values else None


def _sample_sd(values):
    if len(values) < 2:
        return None
    average = _mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def _sem(values):
    spread = _sample_sd(values)
    return spread / math.sqrt(len(values)) if spread is not None else None


def load_records(input_directories, expected_data_kind=None):
    """Load episode JSON records and reject accidental phase/synthetic mixing."""
    records = []
    data_kinds = set()
    completed_identities = set()
    for directory in input_directories:
        for path in sorted(Path(directory).glob("episodes/*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Unreadable JSON record: {path}") from error
            record["_path"] = str(path)
            if record.get("status") == "completed" and record.get("episode_key"):
                identity = record["episode_key"]
                if identity in completed_identities:
                    raise ValueError(f"Duplicate completed episode identity: {identity}")
                completed_identities.add(identity)
            kind = record.get("data_kind")
            if kind:
                data_kinds.add(kind)
            records.append(record)
    if not records:
        raise ValueError("No episode records found under the supplied run directories.")
    if len(data_kinds) != 1:
        raise ValueError(f"Analysis refuses to mix data kinds: {sorted(data_kinds)}")
    data_kind = next(iter(data_kinds))
    if data_kind == "synthetic":
        raise ValueError("Synthetic fixtures cannot be analysed as experiment data.")
    if expected_data_kind and data_kind != expected_data_kind:
        raise ValueError(f"Expected {expected_data_kind!r} data, found {data_kind!r}.")
    return records, data_kind


def completed_records(records):
    return [record for record in records if record.get("status") == "completed" and "outcome" in record]


def summary_rows(records):
    """Aggregate completed records without treating incomplete attempts as results."""
    grouped = defaultdict(list)
    incomplete = 0
    repeated_subset_records = 0
    for record in records:
        if record.get("status") != "completed" or "outcome" not in record:
            incomplete += 1
            continue
        spec = record["spec"]
        if "variability_subset" in spec:
            repeated_subset_records += 1
            continue
        grouped[(spec["model"], spec["benchmark"], spec["agent"])].append(record)

    rows = []
    for (model, benchmark, agent), items in sorted(grouped.items()):
        step_limit = items[0].get("effective_config", {}).get("evaluation", {}).get("max_steps_per_episode")
        progressions = [item["outcome"].get("progression") for item in items]
        progressions = [value for value in progressions if isinstance(value, (int, float))]
        successes = [item["outcome"].get("success") for item in items if item["outcome"].get("success") is not None]
        call_records = [call for item in items for call in item.get("usage", {}).get("calls", [])]
        actor_calls = sum(call["role"] == "actor" for call in call_records)
        critic_calls = sum(call["role"] == "critic" for call in call_records)
        usage_available_calls = sum(bool(call.get("usage_available")) for call in call_records)
        steps = [step for item in items for step in item.get("steps", [])]
        actor_errors = sum(not step.get("agent_trace", {}).get("actor", {}).get("parse", {}).get("valid", True) for step in steps)
        critic_errors = sum(not step["agent_trace"]["reflection"].get("parse", {}).get("valid", True)
                            for step in steps if step.get("agent_trace", {}).get("reflection"))
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "agent": agent,
                "n_completed": len(items),
                "mean_progression": _mean(progressions),
                "sample_sd_progression": _sample_sd(progressions),
                "sem_progression": _sem(progressions),
                "success_rate": _mean([float(value) for value in successes]) if successes else None,
                "mean_steps": _mean([item["outcome"].get("steps", 0) for item in items]),
                "mean_invalid_actions": _mean([item["outcome"].get("invalid_actions", 0) for item in items]),
                "actor_format_errors": actor_errors,
                "critic_format_errors": critic_errors,
                "normalizations": sum(bool(step.get("normalization_changed")) for step in steps),
                "invalid_action_rate": _mean([float(bool(step.get("invalid_action"))) for step in steps]),
                "unchanged_observation_rate": _mean([float(step["observation"] == step["observation_after"])
                                                    for step in steps if "observation_after" in step]),
                "output_limit_stops": sum(call.get("status") == "length" for call in call_records),
                "actor_calls": actor_calls,
                "critic_calls": critic_calls,
                "retries": sum(call.get("retries", 0) for call in call_records),
                "input_tokens": sum(
                    call["input_tokens"] for call in call_records if isinstance(call.get("input_tokens"), int)
                ),
                "output_tokens": sum(
                    call["output_tokens"] for call in call_records if isinstance(call.get("output_tokens"), int)
                ),
                "calls_with_reported_token_usage": usage_available_calls,
                "calls_with_unavailable_token_usage": len(call_records) - usage_available_calls,
                "estimated_cost_usd": sum(
                    item.get("usage", {}).get("episode_estimated_cost_usd", 0.0) for item in items
                ),
                "provider_actual_cost_usd": None,
                "mean_runtime_seconds": _mean([item.get("elapsed_seconds", 0.0) for item in items]),
                "incomplete_records_excluded": incomplete,
                "repeated_subset_records_excluded": repeated_subset_records,
                "progression_definition": PROGRESSION_DEFINITIONS.get(
                    benchmark, "Read the environment implementation."
                ),
                "evaluation_step_limit": step_limit,
                "evaluation_setting": (f"{step_limit}-step evaluation; not the original 1000-step setting"
                                       if benchmark == "crafter"
                                       else f"{step_limit}-step evaluation" if step_limit else "Historical limit unavailable"),
            }
        )
    return rows


def paired_differences(records, left_agent, right_agent, bootstrap_samples=2000, seed=0):
    """Compare matched task/seed cells; repeated-run records stay separate."""
    by_key = defaultdict(dict)
    for record in completed_records(records):
        spec = record["spec"]
        if spec["agent"] not in {left_agent, right_agent}:
            continue
        if "variability_subset" in spec:
            continue
        progression = record["outcome"].get("progression")
        if not isinstance(progression, (int, float)):
            continue
        key = (spec["model"], spec["benchmark"], spec["task"], spec["seed"], spec.get("repeat_index", 0))
        by_key[key][spec["agent"]] = progression

    groups = defaultdict(list)
    for key, agent_scores in sorted(by_key.items()):
        if left_agent in agent_scores and right_agent in agent_scores:
            groups[key[:2]].append(agent_scores[left_agent] - agent_scores[right_agent])
    random_source = random.Random(seed)
    rows = []
    for (model, benchmark), differences in sorted(groups.items()):
        intervals = []
        if len(differences) > 1:
            for _ in range(bootstrap_samples):
                sample = [random_source.choice(differences) for _ in differences]
                intervals.append(_mean(sample))
            intervals.sort()
            lower = intervals[int(0.025 * (len(intervals) - 1))]
            upper = intervals[int(0.975 * (len(intervals) - 1))]
        else:
            lower = None
            upper = None
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "comparison": f"{left_agent} minus {right_agent}",
                "n_matched_pairs": len(differences),
                "mean_difference": _mean(differences),
                "sample_sd_difference": _sample_sd(differences),
                "bootstrap_95_ci_low": lower,
                "bootstrap_95_ci_high": upper,
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "interval_note": "Exploratory percentile bootstrap of matched settings; small n, no multiplicity correction. Degenerate intervals do not establish equivalence.",
                "individual_differences": differences,
            }
        )
    return rows


def repeated_run_variability(records):
    """Include the main observation plus its extra samples; never inflate main n."""
    def setting(record):
        spec = record["spec"]
        return (spec["model"], spec["benchmark"], spec["agent"], spec["task"], spec["seed"])
    repeated_keys = {setting(record) for record in completed_records(records)
                     if "variability_subset" in record["spec"]}
    grouped = defaultdict(list)
    for record in completed_records(records):
        if setting(record) in repeated_keys:
            grouped[setting(record)].append(record)
    rows = []
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["spec"].get("repeat_index", 0))
        values = [item["outcome"]["progression"] for item in items]
        sequences = {tuple(step["applied_action"] for step in item.get("steps", [])) for item in items}
        rows.append({"model": key[0], "benchmark": key[1], "agent": key[2], "task": key[3], "seed": key[4],
                     "n_repeats": len(values), "mean_progression": _mean(values),
                     "repeat_indices": [item["spec"].get("repeat_index", 0) for item in items],
                     "progressions": values,
                     "steps_per_repeat": [item["outcome"].get("steps", 0) for item in items],
                     "sample_sd_progression": _sample_sd(values), "min_progression": min(values),
                     "max_progression": max(values), "distinct_action_sequences": len(sequences),
                     "sample_sd_steps": _sample_sd([item["outcome"].get("steps", 0) for item in items]),
                     "invalid_actions_per_repeat": [item["outcome"].get("invalid_actions", 0) for item in items],
                     "interpretation": "Same setting sampled repeatedly, including main run if present; not independent tasks. Equal scores do not imply identical behavior."})
    return rows


def validate_paired_resets(records):
    """Seed matching is only meaningful if the initial visible task also matches."""
    initial = {}
    for record in completed_records(records):
        if record.get("schema_version", 1) < 2:
            continue
        spec = record["spec"]
        key = (spec["benchmark"], spec["task"], spec["seed"])
        value = (record["reset"].get("mission"), record["steps"][0]["observation"])
        if key in initial and initial[key] != value:
            raise ValueError(f"Paired setting has different initial mission/observation: {key}")
        initial[key] = value


def task_rows(records):
    groups = defaultdict(list)
    for record in completed_records(records):
        spec = record["spec"]
        if "variability_subset" not in spec:
            groups[(spec["model"], spec["benchmark"], spec["agent"], spec["task"])].append(record["outcome"]["progression"])
    return [{"model": key[0], "benchmark": key[1], "agent": key[2], "task": key[3],
             "n": len(values), "mean_progression": _mean(values), "individual_progressions": values}
            for key, values in sorted(groups.items())]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_traces(path, records):
    """Select best and worst main episode per cell, with deterministic seed ties."""
    groups = defaultdict(list)
    for record in completed_records(records):
        spec = record["spec"]
        if "variability_subset" not in spec:
            groups[(spec["model"], spec["benchmark"], spec["agent"])].append(record)
    lines = ["# Trace inspection", "", "Mechanical selection: lowest and highest progression per cell, "
             "breaking ties by seed. These cases illustrate mechanisms, not their frequency. Full prompts and "
             "all transitions remain in the linked raw records. Check critic evidence manually; its claims are not ground truth.", ""]
    for key, items in sorted(groups.items()):
        items.sort(key=lambda record: (record["outcome"]["progression"], record["spec"]["seed"]))
        selected = [items[0]] + ([items[-1]] if len(items) > 1 else [])
        for record in selected:
            spec = record["spec"]
            lines += [f"## {' | '.join(key)} | seed {spec['seed']}", "",
                      f"Task: {spec['task']}; progression: {record['outcome']['progression']}; raw: `{record.get('_path')}`", ""]
            steps = record.get("steps", [])
            indices = set(range(min(3, len(steps)))) | set(range(max(0, len(steps)-3), len(steps)))
            for predicate in [lambda step: step.get("invalid_action"),
                              lambda step: step.get("agent_trace", {}).get("reflection")]:
                found = next((i for i, step in enumerate(steps) if predicate(step)), None)
                if found is not None:
                    indices.update(range(max(0, found-1), min(len(steps), found+2)))
            for index in sorted(indices):
                step = steps[index]
                lines += [f"### Step {step['step']}", "", "```json", json.dumps(step, indent=2, ensure_ascii=False), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def create_figures(output_directory, rows, paired_rows):
    """Create a performance matrix and paired-difference dot plot as PDF and PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib before generating figures: pip install matplotlib") from error

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    matrix = defaultdict(dict)
    for row in rows:
        matrix[(row["model"], row["benchmark"])][row["agent"]] = row["mean_progression"]
    cells = sorted(matrix)
    agents = sorted({row["agent"] for row in rows})
    figure, axis = plt.subplots(figsize=(max(6, len(agents) * 1.8), max(3, len(cells) * 0.7 + 1)))
    values = [[matrix[cell].get(agent, float("nan")) for agent in agents] for cell in cells]
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(agents)), agents)
    axis.set_yticks(range(len(cells)), [f"{model}\n{benchmark}" for model, benchmark in cells])
    for row_index, row_values in enumerate(values):
        for column_index, value in enumerate(row_values):
            if not math.isnan(value):
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="Mean progression")
    axis.set_title("Performance matrix (completed episodes only)")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output_directory / f"performance_matrix.{extension}")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, max(3, len(paired_rows) * 0.8 + 1)))
    labels = []
    for index, row in enumerate(paired_rows):
        values = row["individual_differences"]
        axis.scatter(values, [index] * len(values), alpha=0.8)
        if row["mean_difference"] is not None:
            axis.scatter([row["mean_difference"]], [index], marker="D", color="black", zorder=3)
        labels.append(f"{row['model']} | {row['benchmark']}\n{row['comparison']}")
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Paired progression difference")
    axis.set_title("Matched task/seed differences; diamonds are means")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output_directory / f"paired_differences.{extension}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Analyse one compatible set of Assignment 2 run directories.")
    parser.add_argument("runs", nargs="+", help="Run directories containing an episodes/ directory.")
    parser.add_argument("--phase", choices=["development", "pilot", "final"], required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    for directory in arguments.runs:
        validate_run_directory(directory)
    records, _ = load_records(arguments.runs, expected_data_kind=arguments.phase)
    settings = set()
    benchmark_settings = {}
    model_settings = {}
    for record in completed_records(records):
        effective = record.get("effective_config", {})
        settings.add(json.dumps({"source": record.get("source_files"),
                                "runtime_versions": record.get("runtime_versions"), "agent": {
            key: value for key, value in effective.get("agent", {}).items() if key != "type"}}, sort_keys=True))
        environment = copy.deepcopy(effective["environment"])
        environment["env_kwargs"].pop("seed", None)
        environment["crafter_kwargs"].pop("seed", None)
        benchmark_key = record["spec"]["benchmark"]
        benchmark_value = {"environment": environment, "evaluation": effective["evaluation"]}
        if benchmark_key in benchmark_settings and benchmark_settings[benchmark_key] != benchmark_value:
            raise ValueError("Analysis refuses to mix environment settings or step limits within a benchmark.")
        benchmark_settings[benchmark_key] = benchmark_value
        model_key = record["spec"]["model"]
        model_value = {"profile": record["model_profile"], "snapshot": record.get("local_model_snapshot")}
        if model_key in model_settings and model_settings[model_key] != model_value:
            raise ValueError("Analysis refuses to mix profiles or snapshots within a model.")
        model_settings[model_key] = model_value
    if len(settings) != 1:
        raise ValueError("Analysis refuses to mix source versions or prompt/history settings.")
    validate_paired_resets(records)
    output = Path(arguments.output)
    rows = summary_rows(records)
    paired = paired_differences(records, "planning", "naive")
    paired += paired_differences(records, "reflective_planning", "planning")
    paired += paired_differences(records, "reflective_planning", "naive")
    write_csv(output / "performance_matrix.csv", rows)
    write_csv(output / "task_breakdown.csv", task_rows(records))
    write_csv(output / "paired_differences.csv", paired)
    write_csv(output / "repeated_run_variability.csv", repeated_run_variability(records))
    write_selected_traces(output / "selected_traces.md", records)
    create_figures(output, rows, paired)


if __name__ == "__main__":
    main()
