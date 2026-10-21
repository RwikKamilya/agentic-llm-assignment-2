"""Manifest-driven, one-worker evaluation runner for Assignment 2.

It reuses BALROG's environment factory, prompt builder, agent factory, action
validator, and progression metrics while adding fixed task/seed manifests and
atomic assignment records.
"""

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.agents import ModelCallError
from assignment2.budget import BudgetExceeded, BudgetLedger, paid_run_lock
from assignment2.storage import atomic_json_write, next_attempt_path
from assignment2.validation import IncompletePhaseError, completed_record_problems, validate_run_directory
from balrog.agents import AgentFactory


def make_env(*args, **kwargs):
    # Configuration and offline validation do not need installed game engines.
    from balrog.environments import make_env as create_environment
    return create_environment(*args, **kwargs)


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


def execution_source_hashes(repository_root):
    """Identify executed code/config, including uncommitted patches and prompts."""
    root = Path(repository_root)
    paths = []
    # Historical backups and deployment receipts are evidence, not executed code.
    paths.extend((root / "assignment2").glob("*.py"))
    paths.extend((root / "assignment2/config").glob("*.yaml"))
    for pattern in ["*.sh", "*.sbatch"]:
        paths.extend((root / "assignment2/scripts").glob(pattern))
    for pattern in ["*.py", "*.yaml", "*.json"]:
        paths.extend((root / "balrog").rglob(pattern))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def source_provenance(repository_root):
    """Identify an allowlisted deployment without pretending Git HEAD changed."""
    root = Path(repository_root).resolve()
    checkout_commit = git_commit(root)
    receipt_path = root / "assignment2/deployed_version.json"
    if not receipt_path.exists():
        return checkout_commit, checkout_commit, None
    receipt = json.loads(receipt_path.read_text())
    commit = receipt.get("source_commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Invalid deployed source commit.")
    if not receipt.get("files"):
        raise ValueError("Deployment receipt has no verified files.")
    for relative, expected_hash in receipt["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or Path(relative).parts[0] not in {"assignment2", "balrog", "tests"}:
            raise ValueError("Deployment receipt contains an out-of-scope path.")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Deployed file differs from the frozen version: {relative}")
    return commit, checkout_commit, fingerprint(receipt)


def runtime_versions():
    versions = {"python": platform.python_version()}
    for name in ["openai", "omegaconf", "gym", "gymnasium", "minigrid", "crafter", "numpy", "scipy", "Pillow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


class ExperimentRunner:
    """Run only the cells specified in the assignment experiment manifest."""

    def __init__(self, repository_root, study_config, phase, resume_directory=None, run_directory=None, model_names=None):
        self.repository_root = Path(repository_root)
        self.study_config = study_config
        self.phase = phase
        self.phase_config = study_config.phases.get(phase)
        if self.phase_config is None:
            raise ValueError(f"Unknown phase {phase!r}. Check assignment2/config/experiment.yaml.")
        self.model_names = list(model_names or self.phase_config.models)
        self._validate_manifest()

        if resume_directory and run_directory:
            raise ValueError("resume_directory and run_directory cannot both be set.")
        if resume_directory:
            self.run_directory = Path(resume_directory)
            if not self.run_directory.is_dir():
                raise ValueError(f"Resume directory does not exist: {self.run_directory}")
        elif run_directory:
            self.run_directory = Path(run_directory)
            if self.run_directory.exists():
                raise ValueError(
                    f"New run directory already exists: {self.run_directory}. Use --resume to continue it."
                )
            self.run_directory.mkdir(parents=True, exist_ok=False)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_directory = self.repository_root / self.study_config.output_root / f"{timestamp}_{phase}"
            self.run_directory.mkdir(parents=True, exist_ok=False)

        self.episodes_directory = self.run_directory / "episodes"
        self.episodes_directory.mkdir(exist_ok=True)
        self.base_config = OmegaConf.load(self.repository_root / "balrog/config/config.yaml")
        self.source_commit, self.checkout_commit, self.deployment_receipt_hash = source_provenance(self.repository_root)
        self.source_files = execution_source_hashes(self.repository_root)
        self.runtime_versions = runtime_versions()
        self.run_id = self.run_directory.name
        self.api_ledger_path = self._api_ledger_path()
        self.run_ledger = None  # Load only after acquiring the paid-run lock.
        self.local_usage_ledger = BudgetLedger(
            ceiling_usd=0.0,
            max_input_tokens=self.study_config.budget.max_input_tokens_per_call,
            max_output_tokens=self.study_config.budget.max_output_tokens_per_call,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        )

    def _validate_manifest(self):
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("Select a nonempty, duplicate-free model list.")
        if not 1 <= int(self.study_config.execution.client_max_retries) <= 2:
            raise ValueError("At most two total client attempts are permitted.")
        if float(self.study_config.execution.request_timeout_seconds) <= 0:
            raise ValueError("A positive request timeout is required.")
        for model_name in self.model_names:
            if model_name not in self.study_config.models:
                raise ValueError(f"Unknown model profile {model_name!r}.")
            profile = self.study_config.models[model_name]
            if profile.client_name not in {"openai", "vllm"}:
                raise ValueError("This evaluation supports only the reviewed OpenAI and local vLLM profiles.")
            if int(profile.max_output_tokens) != int(self.study_config.budget.max_output_tokens_per_call):
                raise ValueError("Profile output limit differs from its ledger reservation.")
        if not self.phase_config.agents:
            raise ValueError("A phase must list at least one agent.")
        if len(set(self.phase_config.agents)) != len(self.phase_config.agents):
            raise ValueError("Duplicate phase agents.")
        if any(agent not in {"naive", "planning", "reflective_planning"} for agent in self.phase_config.agents):
            raise ValueError("Unknown Assignment 2 agent.")
        for benchmark in self.phase_config.benchmarks:
            if benchmark not in self.study_config.benchmarks:
                raise ValueError(f"Unknown benchmark {benchmark!r}.")
            episodes = self.study_config.benchmarks[benchmark].episodes.get(self.phase)
            if not episodes:
                raise ValueError(f"Benchmark {benchmark!r} has no episode list for phase {self.phase!r}.")
            for episode in episodes:
                if "task" not in episode or "seed" not in episode:
                    raise ValueError(f"Every {benchmark} episode must define both task and seed.")
            if self._step_limit(benchmark) < 1:
                raise ValueError("Episode step limits must be positive.")
        specs = self._episode_specs()
        if len({fingerprint(spec) for spec in specs}) != len(specs):
            raise ValueError("Duplicate episode identities in manifest.")

    def _step_limit(self, benchmark):
        return int(self.phase_config.get("max_steps", {}).get(
            benchmark, self.study_config.benchmarks[benchmark].max_steps))

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
            # A scheduled one-model job must not silently add another model's
            # repeated subset. A combined job still includes both profiles.
            configured_models = list(repeated.get("models", self.model_names))
            repeated_models = [model_name for model_name in configured_models if model_name in self.model_names]
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
        config.client.base_url = ("https://api.openai.com/v1" if profile.client_name == "openai"
                                  else profile.base_url)
        config.client.generate_kwargs.temperature = float(profile.temperature)
        config.client.generate_kwargs.max_tokens = int(profile.max_output_tokens)
        config.client.max_retries = int(self.study_config.execution.client_max_retries)
        config.client.delay = float(self.study_config.execution.retry_delay_seconds)
        config.client.timeout = float(self.study_config.execution.request_timeout_seconds)
        config.client.max_input_tokens = int(self.study_config.budget.max_input_tokens_per_call)
        config.eval.num_workers = 1
        config.eval.max_steps_per_episode = self._step_limit(spec["benchmark"])
        config.envs.names = spec["benchmark"]
        config.envs.env_kwargs.seed = spec["seed"]
        if spec["benchmark"] == "crafter":
            # Crafter seeds its world at construction and has no seed() API.
            config.envs.crafter_kwargs.seed = spec["seed"]
        return config

    def _episode_key(self, spec, config):
        identity = {
            "phase": self.phase,
            "spec": spec,
            "model": self.study_config.models[spec["model"]],
            "agent": OmegaConf.to_container(config.agent, resolve=True),
            "client": OmegaConf.to_container(config.client, resolve=True),
            "environment": OmegaConf.to_container(config.envs, resolve=True),
            "evaluation": OmegaConf.to_container(config.eval, resolve=True),
            "source_commit": self.source_commit,
            "source_files": self.source_files,
            "runtime_versions": self.runtime_versions,
            "local_model_snapshot": (os.environ.get("ASSIGNMENT2_QWEN_SNAPSHOT")
                                     if spec["model"] == "qwen_7b_local" else None),
        }
        return fingerprint(identity)[:20], fingerprint(identity)

    def _matching_completed_record(self, expected):
        for path in sorted(self.episodes_directory.glob(f"{expected['episode_key']}_attempt-*.json")):
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                raise IncompletePhaseError(f"Unreadable saved record: {path}") from error
            if record.get("status") == "completed":
                problems = completed_record_problems(record, expected, self.phase)
                if problems:
                    raise IncompletePhaseError(f"Saved completed record is invalid: {path}: {problems}")
                return path
        return None

    def _next_attempt_path(self, episode_key):
        return next_attempt_path(self.episodes_directory, episode_key)

    def _api_ledger_path(self):
        configured_path = os.environ.get("ASSIGNMENT2_BUDGET_LEDGER_PATH")
        if configured_path:
            return Path(configured_path)
        output_root = Path(self.study_config.output_root)
        if not output_root.is_absolute():
            output_root = self.repository_root / output_root
        return output_root / "api_spending_ledger.json"

    def _build_api_ledger(self):
        """Reuse cumulative charges; mismatched historical ledgers require migration."""
        return BudgetLedger.load_or_create(
            ceiling_usd=self.study_config.budget.total_ceiling_usd,
            max_input_tokens=self.study_config.budget.max_input_tokens_per_call,
            max_output_tokens=self.study_config.budget.max_output_tokens_per_call,
            input_price_per_million=self.study_config.budget.conservative_input_usd_per_million,
            output_price_per_million=self.study_config.budget.conservative_output_usd_per_million,
            state_path=self.api_ledger_path,
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
            "checkout_commit": self.checkout_commit,
            "deployment_receipt_hash": self.deployment_receipt_hash,
            "source_files": self.source_files,
            "runtime_versions": self.runtime_versions,
            "local_model_snapshot": (os.environ.get("ASSIGNMENT2_QWEN_SNAPSHOT")
                                     if spec["model"] == "qwen_7b_local" else None),
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

    def _run_episode(self, spec, config, record, record_path=None):
        profile = self.study_config.models[spec["model"]]
        ledger = self.run_ledger if profile.client_name == "openai" else self.local_usage_ledger
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
            if profile.client_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is required for the OpenAI profile."
                )
            agent = AgentFactory(config).create_agent(budget_ledger=ledger)
            environment = make_env(spec["benchmark"], spec["task"], config)
            agent.reset()
            if spec["benchmark"] == "crafter":
                observation, _ = environment.reset()
            else:
                observation, _ = environment.reset(seed=spec["seed"])
            if not isinstance(observation, dict) or not isinstance(observation.get("text"), dict):
                raise ValueError("Environment reset did not return a text observation.")
            record["reset"] = {"seed": spec["seed"], "mission": observation.get("mission"),
                               "seed_method": "constructor" if spec["benchmark"] == "crafter" else "reset"}
            if spec["benchmark"] == "babyai":
                base_environment = environment.unwrapped
                actual_kind = base_environment.action_kinds[0].replace(" ", "_")
                if actual_kind != spec["task"].split("/", 1)[1]:
                    raise ValueError("BabyAI reset changed the requested task kind.")
                record["reset"]["task_kind"] = actual_kind
                record["reset"]["task_selection_attempts"] = base_environment.task_selection_attempts
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
                record["usage"] = {"calls": ledger.calls[first_call_index:]}
                record["model_call_traces"] = agent.call_trace
                if record_path is not None:
                    atomic_json_write(record_path, record)
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
                try:
                    environment.close()
                except Exception as exc:
                    status = "incomplete"
                    termination_reason = "cleanup_error"
                    error = f"{type(exc).__name__}: environment.close failed"

        record["status"] = status
        record["completed_at"] = utc_now()
        record["elapsed_seconds"] = time.monotonic() - start
        record["termination_reason"] = termination_reason
        record["error"] = error
        record["model_call_traces"] = agent.call_trace if agent is not None else []
        episode_calls = ledger.calls[first_call_index:]
        record["usage"] = {
            "episode_estimated_cost_usd": sum(call["estimated_cost_usd"] for call in episode_calls),
            "provider_actual_cost_usd": None,
            "calls": episode_calls,
            "run_budget_after_episode": ledger.summary(),
        }
        return record

    def run(self):
        paid = any(self.study_config.models[name].client_name == "openai" for name in self.model_names)
        lock = paid_run_lock(self.api_ledger_path) if paid else nullcontext()
        with lock:
            with paid_run_lock(self.run_directory / "execution.json"):
                self.run_ledger = self._build_api_ledger() if paid else self.local_usage_ledger
                return self._run_phase()

    def _run_phase(self):
        planned = self._episode_specs()
        expected = []
        for spec in planned:
            config = self._effective_config(spec)
            key, config_hash = self._episode_key(spec, config)
            header = self._record_header(spec, config, key, config_hash)
            expected.append({field: header[field] for field in
                             ["episode_key", "config_fingerprint", "spec", "effective_config", "model_profile",
                              "source_commit", "source_files", "runtime_versions", "local_model_snapshot"]})
        manifest = {
            "run_id": self.run_id,
            "phase": self.phase,
            "created_at": utc_now(),
            "source_commit": self.source_commit,
            "checkout_commit": self.checkout_commit,
            "deployment_receipt_hash": self.deployment_receipt_hash,
            "source_files": self.source_files,
            "runtime_versions": self.runtime_versions,
            "selected_models": self.model_names,
            "expected_episodes": expected,
            "study_config": OmegaConf.to_container(self.study_config, resolve=True),
            "execution_note": "One worker only. Existing complete, compatible episodes are skipped on resume.",
            "api_budget_ledger_path": str(self.api_ledger_path),
        }
        manifest_path = self.run_directory / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("expected_episodes") != expected or previous.get("phase") != self.phase:
                raise IncompletePhaseError("Resume configuration/source differs from the saved expected identities.")
            manifest = previous
            manifest.setdefault("resumed_at", []).append(utc_now())
        atomic_json_write(self.run_directory / "manifest.json", manifest)
        completed = 0
        skipped = 0
        for index, spec in enumerate(planned, start=1):
            config = self._effective_config(spec)
            episode_key, config_fingerprint = self._episode_key(spec, config)
            existing = self._matching_completed_record(expected[index - 1])
            if existing:
                skipped += 1
                print(f"[{index}/{len(planned)}] skip compatible {spec['benchmark']} {spec['agent']} -> {existing}")
                continue
            record_path = self._next_attempt_path(episode_key)
            record = self._record_header(spec, config, episode_key, config_fingerprint)
            atomic_json_write(record_path, record)
            record = self._run_episode(spec, config, record, record_path)
            if record["status"] == "completed":
                problems = completed_record_problems(record, expected[index - 1], self.phase)
                if problems:
                    record["status"] = "incomplete"
                    record["termination_reason"] = "infrastructure_validation"
                    record["error"] = "; ".join(problems)
            atomic_json_write(record_path, record)
            completed += record["status"] == "completed"
            progression = record.get("outcome", {}).get("progression", "unavailable")
            print(
                f"[{index}/{len(planned)}] completed={completed} {record['status']} "
                f"{spec['benchmark']} {spec['agent']} "
                f"progression={progression} elapsed={record['elapsed_seconds']:.1f}s -> {record_path}"
            )
            if record["status"] != "completed":
                # Budget, authentication, configuration and exhausted retries
                # cannot improve by blindly proceeding to more paid episodes.
                break
        print(
            f"Run finished: {completed} completed, {skipped} compatible episodes skipped, "
            f"output={self.run_directory}"
        )
        manifest["last_attempt_finished_at"] = utc_now()
        manifest["api_budget"] = self.run_ledger.summary()
        atomic_json_write(self.run_directory / "manifest.json", manifest)
        validation = validate_run_directory(self.run_directory)
        manifest["validation"] = validation
        manifest["completed_at"] = utc_now()
        atomic_json_write(self.run_directory / "manifest.json", manifest)
        print(f"Validated {validation['completed']}/{validation['expected']} episode identities; "
              f"invalid actions={validation['invalid_actions']}, parser failures={validation['parser_failures']}")
        return self.run_directory
