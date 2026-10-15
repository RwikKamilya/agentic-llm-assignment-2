"""Manifest-driven, one-worker evaluation runner for Assignment 2.

It reuses BALROG's environment factory, prompt builder, agent factory, action
validator, and progression metrics while adding fixed task/seed manifests and
atomic assignment records.
"""

import copy
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.agents import ModelCallError
from assignment2.budget import BudgetExceeded, BudgetLedger
from assignment2.storage import atomic_json_write, matching_completed_record, next_attempt_path
from balrog.agents import AgentFactory
from balrog.environments import make_env


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    """Convert numpy scalars and other environment values before JSON logging."""
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fingerprint(value):
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(repository_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


class ExperimentRunner:
    """Run only the cells specified in the assignment experiment manifest."""

    def __init__(self, repository_root, study_config, phase, resume_directory=None, model_names=None):
        self.repository_root = Path(repository_root)
        self.study_config = study_config
        self.phase = phase
        self.phase_config = study_config.phases.get(phase)
        if self.phase_config is None:
            raise ValueError(f"Unknown phase {phase!r}. Check assignment2/config/experiment.yaml.")
        self.model_names = list(model_names or self.phase_config.models)
        self._validate_manifest()

        if resume_directory:
            self.run_directory = Path(resume_directory)
            if not self.run_directory.is_dir():
                raise ValueError(f"Resume directory does not exist: {self.run_directory}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_directory = self.repository_root / self.study_config.output_root / f"{timestamp}_{phase}"
            self.run_directory.mkdir(parents=True, exist_ok=False)

        self.episodes_directory = self.run_directory / "episodes"
        self.episodes_directory.mkdir(exist_ok=True)
        self.base_config = OmegaConf.load(self.repository_root / "balrog/config/config.yaml")
        self.source_commit = git_commit(self.repository_root)
        self.run_id = self.run_directory.name
        self.run_ledger = self._build_ledger(phase)

    def _validate_manifest(self):
        for model_name in self.model_names:
            if model_name not in self.study_config.models:
                raise ValueError(f"Unknown model profile {model_name!r}.")
        if not self.phase_config.agents:
            raise ValueError("A phase must list at least one agent.")
        for benchmark in self.phase_config.benchmarks:
            if benchmark not in self.study_config.benchmarks:
                raise ValueError(f"Unknown benchmark {benchmark!r}.")
            episodes = self.study_config.benchmarks[benchmark].episodes.get(self.phase)
            if not episodes:
                raise ValueError(f"Benchmark {benchmark!r} has no episode list for phase {self.phase!r}.")
            for episode in episodes:
                if "task" not in episode or "seed" not in episode:
                    raise ValueError(f"Every {benchmark} episode must define both task and seed.")

    def _episode_specs(self):
        specs = []
        for model_name in self.model_names:
            for benchmark in self.phase_config.benchmarks:
                entries = self.study_config.benchmarks[benchmark].episodes[self.phase]
                for agent_name in self.phase_config.agents:
                    for episode in entries:
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
        for repeated in self.phase_config.get("repeated_runs", []):
            repeated_models = list(repeated.get("models", self.model_names))
            repeated_agents = list(repeated.get("agents", self.phase_config.agents))
            for model_name in repeated_models:
                for agent_name in repeated_agents:
                    for repeat_index in range(int(repeated.repeats)):
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

    def _effective_config(self, spec):
        config = copy.deepcopy(self.base_config)
        profile = self.study_config.models[spec["model"]]
        config.agent.type = spec["agent"]
        config.agent.max_text_history = int(self.study_config.agent.max_text_history)
        config.agent.max_image_history = int(self.study_config.agent.max_image_history)
        config.agent.max_cot_history = 0
        config.agent.max_plan_words = int(self.study_config.agent.max_plan_words)
        config.agent.max_subgoal_words = int(self.study_config.agent.max_subgoal_words)
        config.agent.max_recent_observations = int(self.study_config.agent.max_recent_observations)
        config.agent.reflection_interval = int(self.study_config.agent.reflection_interval)
        config.agent.max_reflection_lessons = int(self.study_config.agent.max_reflection_lessons)
        config.client.client_name = profile.client_name
        config.client.model_id = profile.model_id
        config.client.base_url = profile.get("base_url", config.client.base_url)
        config.client.generate_kwargs.temperature = float(profile.temperature)
        config.client.generate_kwargs.max_tokens = int(profile.max_output_tokens)
        config.client.max_retries = int(self.study_config.execution.client_max_retries)
        config.client.delay = float(self.study_config.execution.retry_delay_seconds)
        config.eval.num_workers = 1
        config.eval.max_steps_per_episode = self.study_config.benchmarks[spec["benchmark"]].get("max_steps", None)
        config.envs.names = spec["benchmark"]
        config.envs.env_kwargs.seed = spec["seed"]
        return config

    def _episode_key(self, spec, config):
        identity = {
            "phase": self.phase,
            "spec": spec,
            "model": self.study_config.models[spec["model"]],
            "agent": OmegaConf.to_container(config.agent, resolve=True),
            "client": OmegaConf.to_container(config.client, resolve=True),
            "source_commit": self.source_commit,
        }
        return fingerprint(identity)[:20], fingerprint(identity)

    def _matching_completed_record(self, episode_key, config_fingerprint):
        return matching_completed_record(self.episodes_directory, episode_key, config_fingerprint)

    def _next_attempt_path(self, episode_key):
        return next_attempt_path(self.episodes_directory, episode_key)

    def _build_ledger(self, phase_name):
        limit = self.study_config.budget.get(f"{phase_name}_ceiling_usd", self.study_config.budget.total_ceiling_usd)
        return BudgetLedger(
            ceiling_usd=limit,
            max_input_tokens=self.study_config.budget.max_input_tokens_per_call,
            max_output_tokens=self.study_config.budget.max_output_tokens_per_call,
            input_price_per_million=self.study_config.budget.conservative_input_usd_per_million,
            output_price_per_million=self.study_config.budget.conservative_output_usd_per_million,
        )

    def _record_header(self, spec, config, episode_key, config_fingerprint):
        profile = self.study_config.models[spec["model"]]
        return {
            "schema_version": 1,
            "status": "started",
            "run_id": self.run_id,
            "phase": self.phase,
            "data_kind": self.phase,
            "started_at": utc_now(),
            "episode_key": episode_key,
            "config_fingerprint": config_fingerprint,
            "prompt_config_fingerprint": fingerprint(OmegaConf.to_container(config.agent, resolve=True)),
            "source_commit": self.source_commit,
            "spec": spec,
            "model_profile": OmegaConf.to_container(profile, resolve=True),
            "effective_config": {
                "agent": OmegaConf.to_container(config.agent, resolve=True),
                "client": OmegaConf.to_container(config.client, resolve=True),
                "environment": OmegaConf.to_container(config.envs, resolve=True),
                "evaluation": OmegaConf.to_container(config.eval, resolve=True),
            },
            "steps": [],
            "usage": {"provider_actual_cost_usd": None},
        }

    def _run_episode(self, spec, config, record):
        profile = self.study_config.models[spec["model"]]
        ledger = self.run_ledger
        first_call_index = len(ledger.calls)
        agent = None
        environment = None
        start = time.monotonic()
        action = None
        episode_return = 0.0
        termination_reason = "max_steps"
        status = "completed"
        error = None
        try:
            if profile.client_name == "gemini" and not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError(
                    "GEMINI_API_KEY is required for the Gemini profile and must be set in the environment."
                )
            agent = AgentFactory(config).create_agent(budget_ledger=ledger)
            environment = make_env(spec["benchmark"], spec["task"], config)
            agent.reset()
            observation, _ = environment.reset(seed=spec["seed"])
            instructions = observation.get("mission") if spec["benchmark"] == "babyai" else None
            instruction_prompt = environment.get_instruction_prompt(instructions=instructions)
            agent.prompt_builder.update_instruction_prompt(instruction_prompt)
            maximum_steps = environment.max_steps
            if config.eval.max_steps_per_episode is not None:
                maximum_steps = config.eval.max_steps_per_episode

            for step in range(maximum_steps):
                before_text = dict(observation.get("text", {}))
                response = agent.act(observation, prev_action=action)
                applied_action = environment.check_action_validity(response.completion)
                invalid_action = applied_action != response.completion
                observation, reward, terminated, truncated, _ = environment.step(applied_action)
                episode_return += reward
                agent.record_environment_feedback(applied_action, invalid_action, reward, terminated, truncated)
                record["steps"].append(
                    {
                        "step": step,
                        "observation": before_text,
                        "requested_action": response.completion,
                        "applied_action": applied_action,
                        "invalid_action": invalid_action,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "agent_trace": agent.last_trace[-1],
                    }
                )
                action = response.completion
                if terminated:
                    termination_reason = "terminated"
                    break
                if truncated:
                    termination_reason = "truncated"
                    break
            stats = environment.get_stats()
            record["outcome"] = {
                "progression": stats.get("progression"),
                "success": stats.get("progression") == 1.0 if "progression" in stats else None,
                "environment_stats": stats,
                "steps": len(record["steps"]),
                "episode_return": episode_return,
                "termination_reason": termination_reason,
                "invalid_actions": len(environment.failed_candidates),
            }
        except BudgetExceeded as exc:
            status = "incomplete"
            termination_reason = "budget_ceiling"
            error = str(exc)
        except ModelCallError as exc:
            status = "incomplete"
            termination_reason = "model_error"
            error = str(exc)
        except Exception as exc:
            status = "incomplete"
            termination_reason = "infrastructure_error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if environment is not None and hasattr(environment, "close"):
                environment.close()

        record["status"] = status
        record["completed_at"] = utc_now()
        record["elapsed_seconds"] = time.monotonic() - start
        record["termination_reason"] = termination_reason
        record["error"] = error
        episode_calls = ledger.calls[first_call_index:]
        record["usage"] = {
            "episode_estimated_cost_usd": sum(call["estimated_cost_usd"] for call in episode_calls),
            "provider_actual_cost_usd": None,
            "calls": episode_calls,
            "run_budget_after_episode": ledger.snapshot(),
        }
        return record

    def run(self):
        manifest = {
            "run_id": self.run_id,
            "phase": self.phase,
            "created_at": utc_now(),
            "source_commit": self.source_commit,
            "study_config": OmegaConf.to_container(self.study_config, resolve=True),
            "execution_note": "One worker only. Existing complete, compatible episodes are skipped on resume.",
        }
        atomic_json_write(self.run_directory / "manifest.json", manifest)
        planned = self._episode_specs()
        completed = 0
        skipped = 0
        for index, spec in enumerate(planned, start=1):
            config = self._effective_config(spec)
            episode_key, config_fingerprint = self._episode_key(spec, config)
            existing = self._matching_completed_record(episode_key, config_fingerprint)
            if existing:
                skipped += 1
                print(f"[{index}/{len(planned)}] skip compatible {spec['benchmark']} {spec['agent']} -> {existing}")
                continue
            record_path = self._next_attempt_path(episode_key)
            record = self._record_header(spec, config, episode_key, config_fingerprint)
            atomic_json_write(record_path, record)
            record = self._run_episode(spec, config, record)
            atomic_json_write(record_path, record)
            completed += record["status"] == "completed"
            progression = record.get("outcome", {}).get("progression", "unavailable")
            print(
                f"[{index}/{len(planned)}] completed={completed} {record['status']} "
                f"{spec['benchmark']} {spec['agent']} "
                f"progression={progression} elapsed={record['elapsed_seconds']:.1f}s -> {record_path}"
            )
        print(
            f"Run complete: {completed} completed, {skipped} compatible episodes skipped, "
            f"output={self.run_directory}"
        )
        manifest["completed_at"] = utc_now()
        manifest["budget"] = self.run_ledger.snapshot()
        atomic_json_write(self.run_directory / "manifest.json", manifest)
        return self.run_directory
