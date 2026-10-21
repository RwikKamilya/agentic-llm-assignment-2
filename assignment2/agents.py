"""Planning and reflection agents for Assignment 2.

The agents deliberately share one action parser and one bounded state model so
that their comparison isolates the critic call rather than parsing behaviour.
"""

import copy
import re
from collections import namedtuple

from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message

from assignment2.budget import BudgetExceeded
from assignment2.prompts import trim_oldest_history


# BALROG consumes response fields rather than a concrete response class.  A
# local tuple keeps parser-only tests independent of API client dependencies.
LLMResponse = namedtuple(
    "LLMResponse", ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"]
)


class ModelCallError(RuntimeError):
    """A model call failed or returned a response marked as an error."""


def _compact_text(text, max_words):
    """Keep state small while preserving ordinary, readable text."""
    words = str(text or "").strip().split()
    return " ".join(words[:max_words])


def parse_plan_response(response_text, previous_plan=None):
    """Parse the strict actor format without guessing a missing action."""
    text = str(response_text or "")
    headings = ("PLAN", "SUBGOAL", "EXPECTED_OUTCOME", "ACTION")
    values = {}
    for index, heading in enumerate(headings):
        following = "|".join(headings[index + 1 :])
        boundary = rf"(?=^\s*(?:{following})\s*:|\Z)" if following else r"\Z"
        match = re.search(rf"^\s*{heading}\s*:\s*(.*?){boundary}", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        values[heading] = match.group(1).strip() if match else None

    errors = []
    action = values["ACTION"]
    if not action:
        errors.append("missing ACTION section")
    if values["PLAN"] is None:
        errors.append("missing PLAN section")
    if values["SUBGOAL"] is None:
        errors.append("missing SUBGOAL section")
    if values["EXPECTED_OUTCOME"] is None:
        errors.append("missing EXPECTED_OUTCOME section")

    plan = values["PLAN"]
    if plan and plan.lower() in {"unchanged", "no change", "no changes"}:
        plan = previous_plan
    return {
        "plan": plan,
        "subgoal": values["SUBGOAL"],
        "expected_outcome": values["EXPECTED_OUTCOME"],
        "action": action,
        "valid": not errors,
        "errors": errors,
    }


def parse_reflection_response(response_text, max_lessons):
    """Extract up to ``max_lessons`` corrective statements from a critic reply."""
    text = str(response_text or "")
    match = re.search(r"^\s*LESSONS\s*:\s*(.*)\Z", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not match:
        return {"lessons": [], "valid": False, "errors": ["missing LESSONS section"]}

    candidates = []
    for line in match.group(1).splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            candidates.append(_compact_text(cleaned, 16))
    if not candidates:
        return {"lessons": [], "valid": False, "errors": ["no lesson items"]}
    return {"lessons": candidates[:max_lessons], "valid": True, "errors": []}


class PlanningAgent(BaseAgent):
    """A stateful BALROG agent that makes a short plan before each action."""

    def __init__(self, client_factory, prompt_builder, config, budget_ledger=None):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        self.budget_ledger = budget_ledger
        self.max_plan_words = int(config.agent.get("max_plan_words", 80))
        self.max_subgoal_words = int(config.agent.get("max_subgoal_words", 32))
        self.max_recent_observations = int(config.agent.get("max_recent_observations", 3))
        self.reset()

    def reset(self):
        super().reset()
        self.plan = None
        self.subgoal = None
        self.expected_outcome = None
        self.lessons = []
        self.step_number = 0
        self.last_trace = []
        self.call_trace = []
        self._recent_observations = []
        self._recent_actions = []
        self.current_observation_text = ""
        self.request_metadata = {}

    def _observe(self, observation):
        text = observation.get("text", {})
        current = "\n".join(str(text.get(key) or "") for key in ["short_term_context", "long_term_context"]).strip()
        self.current_observation_text = current
        self._recent_observations.append(_compact_text(current, 160))
        self._recent_observations = self._recent_observations[-self.max_recent_observations :]

    def _remember_previous_action(self, previous_action):
        if previous_action is not None:
            self._recent_actions.append(_compact_text(previous_action, 32))
            self._recent_actions = self._recent_actions[-self.max_recent_observations :]

    def _actor_instruction(self):
        plan_text = self.plan or "No plan yet."
        subgoal_text = self.subgoal or "No subgoal yet."
        outcome_text = self.expected_outcome or "No expected outcome yet."
        lessons_text = "\n".join(f"- {lesson}" for lesson in self.lessons) or "- None"
        return f"""
Maintain a concise, evidence-based plan for the current episode. Use only the
game rules, observations, prior actions, and lessons below. Do not claim that
an unobserved event happened. Choose exactly one action from the listed actions.

Current plan: {plan_text}
Current subgoal: {subgoal_text}
Expected observable outcome: {outcome_text}
Corrective lessons:
{lessons_text}

Reply using all four headings exactly once:
PLAN: <at most 24 words, or UNCHANGED>
SUBGOAL: <at most 12 words>
EXPECTED_OUTCOME: <at most 12 words; observable evidence>
ACTION: <one valid game action>
Keep the entire reply below 120 words. Include ACTION even if the plan is unchanged.
""".strip()

    def _call_model(self, messages, role):
        reservation = None
        max_attempts = max(1, int(getattr(self.client, "max_retries", 1)))
        maximum = int(getattr(self.client, "max_input_tokens", None) or 4096)
        messages, self.request_metadata = trim_oldest_history(messages, maximum)
        if hasattr(self.client, "validate_request"):
            self.client.validate_request(messages)
        if self.budget_ledger is not None:
            reservation = self.budget_ledger.reserve(role, max_attempts=max_attempts)
        try:
            response = self.client.generate(messages)
        except Exception as error:
            call_log = None
            if reservation is not None:
                attempts = int(getattr(self.client, "last_call_attempts", 1))
                call_log = self._record_call(reservation, None, None, "exception", attempts)
            self.call_trace.append({"role": role, "raw_response": None, "call": call_log,
                                    "error": type(error).__name__})
            raise ModelCallError(f"{role} call raised {type(error).__name__}") from error

        status = str(getattr(response, "stop_reason", "unknown"))
        input_tokens = getattr(response, "input_tokens", None)
        output_tokens = getattr(response, "output_tokens", None)
        if input_tokens == 0 and output_tokens == 0:
            # A non-empty completion with two zero counts indicates a client
            # that did not expose usage, rather than a free model call.
            input_tokens = None
            output_tokens = None
        attempts = int(getattr(self.client, "last_call_attempts", 1))
        call_log = None
        if reservation is not None:
            call_log = self._record_call(reservation, input_tokens, output_tokens, status, attempts)
        # Keep a critic response even if the following actor request fails.
        self.call_trace.append({"role": role, "raw_response": response.completion,
                                "stop_reason": status, "call": call_log})
        if status.startswith("error"):
            raise ModelCallError(f"{role} call stopped with {status}")
        return response, call_log

    def _record_call(self, reservation, input_tokens, output_tokens, status, attempts):
        return self.budget_ledger.record(
            reservation, input_tokens, output_tokens, status, attempts=attempts,
            retry_errors=getattr(self.client, "last_retry_errors", []),
            attempt_usage=getattr(self.client, "last_attempt_usage", []),
            provider=getattr(self.client, "client_name", None),
            model_id=getattr(self.client, "model_id", None),
            prompt_metadata=self.request_metadata,
        )

    def _maybe_reflect(self):
        """Hook for the reflective variant; plain planning performs no critic call."""
        return None

    def act(self, obs, prev_action=None):
        if prev_action is not None:
            self.prompt_builder.update_action(prev_action)
        self._remember_previous_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._observe(obs)
        reflection = self._maybe_reflect()

        messages = copy.deepcopy(self.prompt_builder.get_prompt())
        if not messages or messages[-1].role != "user":
            raise RuntimeError("BALROG prompt builder did not provide a current user observation.")
        messages[-1].content += "\n\n" + self._actor_instruction()

        response, call_log = self._call_model(messages, "actor")
        parsed = parse_plan_response(response.completion, previous_plan=self.plan)
        if parsed["plan"]:
            self.plan = _compact_text(parsed["plan"], self.max_plan_words)
        if parsed["subgoal"]:
            self.subgoal = _compact_text(parsed["subgoal"], self.max_subgoal_words)
        if parsed["expected_outcome"]:
            self.expected_outcome = _compact_text(parsed["expected_outcome"], self.max_subgoal_words)

        action = parsed["action"] if parsed["valid"] else ""
        trace = {
            "step": self.step_number,
            "actor": {
                "raw_response": response.completion,
                "stop_reason": response.stop_reason,
                "parse": parsed,
                "call": call_log,
            },
            "state_after_decision": {
                "plan": self.plan,
                "subgoal": self.subgoal,
                "expected_outcome": self.expected_outcome,
                "lessons": list(self.lessons),
            },
            "reflection": reflection,
        }
        self.last_trace.append(trace)
        self.step_number += 1
        return LLMResponse(
            model_id=response.model_id,
            completion=action,
            stop_reason=response.stop_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning=self.plan,
        )

    def record_environment_feedback(self, applied_action, invalid_action, reward, terminated, truncated):
        """Attach observable evaluator feedback to the latest action trace."""
        if not self.last_trace:
            return
        self.last_trace[-1]["environment"] = {
            "applied_action": applied_action,
            "invalid_action": bool(invalid_action),
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }


class LoggedNaiveAgent(PlanningAgent):
    """The existing naive decision rule with Assignment 2 logging and budget guards.

    This class deliberately retains BALROG's naive instruction and character
    filter. It only wraps the model call and stores the raw/parsed output.
    """

    def act(self, obs, prev_action=None):
        if prev_action is not None:
            self.prompt_builder.update_action(prev_action)
        self._remember_previous_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._observe(obs)
        messages = copy.deepcopy(self.prompt_builder.get_prompt())
        naive_instruction = (
            "You always have to output one of the above actions at a time and no other text. "
            "You always have to output an action until the episode terminates."
        )
        if not messages or messages[-1].role != "user":
            raise RuntimeError("BALROG prompt builder did not provide a current user observation.")
        messages[-1].content += "\n\n" + naive_instruction
        response, call_log = self._call_model(messages, "actor")
        parsed_action = re.sub(r"[^a-zA-Z\s:]", "", str(response.completion)).strip()
        trace = {
            "step": self.step_number,
            "actor": {
                "raw_response": response.completion,
                "stop_reason": response.stop_reason,
                "parse": {"action": parsed_action, "valid": bool(parsed_action), "errors": []},
                "call": call_log,
            },
            "state_after_decision": {"plan": None, "subgoal": None, "expected_outcome": None, "lessons": []},
            "reflection": None,
        }
        self.last_trace.append(trace)
        self.step_number += 1
        return LLMResponse(
            model_id=response.model_id,
            completion=parsed_action,
            stop_reason=response.stop_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning=None,
        )


class ReflectivePlanningAgent(PlanningAgent):
    """Planning agent with bounded, within-episode, evidence-only critique."""

    def __init__(self, client_factory, prompt_builder, config, budget_ledger=None):
        super().__init__(client_factory, prompt_builder, config, budget_ledger=budget_ledger)
        self.reflection_interval = int(config.agent.get("reflection_interval", 4))
        self.max_lessons = int(config.agent.get("max_reflection_lessons", 3))
        if self.reflection_interval < 1:
            raise ValueError("agent.reflection_interval must be at least 1")

    def _maybe_reflect(self):
        if self.step_number == 0 or self.step_number % self.reflection_interval != 0:
            return None
        instruction = getattr(self.prompt_builder, "system_prompt", "") or ""
        messages = [Message(role="user", content=instruction)]
        for observation, action in zip(self._recent_observations[:-1], self._recent_actions):
            messages.append(Message(role="user", content=f"Earlier observation: {observation}\nPrior action: {action}"))
        last_action = self._recent_actions[-1] if self._recent_actions else "None yet."
        critic_prompt = f"""
You are a critic for a game-playing agent. Use only this game instruction and
the observations and prior submitted actions below. Do not infer hidden state
or invent rewards. Identify at most {self.max_lessons} short, concrete
corrections for the next actions, at most 16 words per lesson and 70 words total.

Current observation:
{self.current_observation_text}

Most recent submitted action: {last_action}

Reply only in this format:
LESSONS:
- <one corrective lesson>
""".strip()
        messages.append(Message(role="user", content=critic_prompt))
        response, call_log = self._call_model(messages, "critic")
        parsed = parse_reflection_response(response.completion, self.max_lessons)
        if parsed["valid"]:
            self.lessons = parsed["lessons"]
        return {
            "raw_response": response.completion,
            "stop_reason": response.stop_reason,
            "parse": parsed,
            "call": call_log,
            "lessons_after_update": list(self.lessons),
        }
