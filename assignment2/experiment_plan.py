"""Shared manifest expansion and conservative request/cost bounds; no execution."""


def planned_episodes(study_config, phase, model_names):
    phase_config = study_config.phases[phase]
    specs = []
    for model_name in model_names:
        for benchmark in phase_config.benchmarks:
            entries = study_config.benchmarks[benchmark].episodes[phase]
            for episode_index, episode in enumerate(entries):
                agents = list(phase_config.agents)
                offset = episode_index % len(agents)
                # Fixed rotation reduces a systematic agent-by-time confound.
                for agent_name in agents[offset:] + agents[:offset]:
                    specs.append(
                        {
                            "model": model_name,
                            "benchmark": benchmark,
                            "agent": agent_name,
                            "task": str(episode.task),
                            "seed": int(episode.seed),
                            "repeat_index": int(episode.get("repeat_index", 0)),
                        }
                    )
    for repeated in phase_config.get("repeated_runs", []):
        # A scheduled one-model job must not silently add another model's
        # repeated subset. A combined job still includes both profiles.
        configured_models = list(repeated.get("models", model_names))
        repeated_models = [model_name for model_name in configured_models if model_name in model_names]
        repeated_agents = list(repeated.get("agents", phase_config.agents))
        for model_name in repeated_models:
            for repeat_index in range(int(repeated.repeats)):
                offset = (repeat_index + 1) % len(repeated_agents)
                for agent_name in repeated_agents[offset:] + repeated_agents[:offset]:
                    specs.append(
                        {
                            "model": model_name,
                            "benchmark": str(repeated.benchmark),
                            "agent": agent_name,
                            "task": str(repeated.task),
                            "seed": int(repeated.seed),
                            "repeat_index": repeat_index + 1,
                            "variability_subset": str(repeated.name),
                        }
                    )
    return specs


def step_limit(study_config, phase, benchmark, agent=None):
    settings = study_config.phases[phase]
    if agent is not None and agent in settings.get("agent_step_limits", {}):
        return int(settings.agent_step_limits[agent])
    return int(settings.get("max_steps", {}).get(benchmark, study_config.benchmarks[benchmark].max_steps))


def logical_call_bound(study_config, phase, spec):
    steps = step_limit(study_config, phase, spec["benchmark"], spec["agent"])
    critics = (steps - 1) // int(study_config.agent.reflection_interval) if spec["agent"] == "reflective_planning" else 0
    return steps + critics


def cost_per_logical_call(study_config):
    budget = study_config.budget
    return int(study_config.execution.client_max_retries) * (
        int(budget.max_input_tokens_per_call) * float(budget.conservative_input_usd_per_million)
        + int(budget.max_output_tokens_per_call) * float(budget.conservative_output_usd_per_million)) / 1_000_000


def planned_budget(study_config, spent=0.0, reserved=0.0):
    calls = {}
    for phase in study_config.execution.planned_phases:
        specs = planned_episodes(study_config, phase, ["openai_mini"])
        calls[phase] = sum(logical_call_bound(study_config, phase, spec) for spec in specs)
    new_cost = sum(calls.values()) * cost_per_logical_call(study_config)
    total = float(spent) + float(reserved) + new_cost
    recovery = float(study_config.budget.recovery_reserve_usd)
    ceiling = float(study_config.budget.total_ceiling_usd)
    return {"logical_calls": calls, "max_attempts": int(study_config.execution.client_max_retries),
            "max_new_cost_usd": new_cost, "historical_spent_usd": spent,
            "outstanding_reserved_usd": reserved, "max_cumulative_usd": total,
            "recovery_reserve_usd": recovery, "remaining_after_plan_usd": ceiling - total,
            "fits_with_recovery_reserve": total + recovery <= ceiling}
