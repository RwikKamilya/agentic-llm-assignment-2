"""Planning and reflection agents for Assignment 2.

All agents share game evidence and action normalization. Planning variants also
share a structured actor format; reflection adds a scheduled critic call.
"""

import re
from collections import namedtuple

from balrog.agents.base import BaseAgent

from assignment2.prompts import input_token_upper_bound, observation_text, shared_evidence


# BALROG consumes response fields rather than a concrete response class.  A
# local tuple keeps parser-only tests independent of API client dependencies.
LLMResponse = namedtuple(
    "LLMResponse", ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"]
)


class ModelCallError(RuntimeError):
    """A model call failed or returned a response marked as an error."""


def _compact_text(text, max_words, max_bytes=160):
    """Keep state small while preserving ordinary, readable text."""
    words = str(text or "").strip().split()
    return " ".join(words[:max_words]).encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _sections(text, headings):
    """Require every named section once, in order, with no leading prose."""
    matches = list(re.finditer(r"^\s*([A-Z_]+)\s*:", str(text or ""), re.MULTILINE | re.IGNORECASE))
    names = [match.group(1).upper() for match in matches]
    errors = []
    for heading in headings:
        if heading not in names:
            errors.append(f"missing {heading} section")
    if names != list(headings):
        errors.append("headings must occur exactly once in the specified order")
    if matches and str(text)[:matches[0].start()].strip():
        errors.append("unexpected text before headings")
    values = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(str(text))
        values[names[index]] = str(text)[match.end():end].strip()
    for heading in headings:
        if heading in values and not values[heading]:
            errors.append(f"empty {heading} section")
    return values, errors


def parse_plan_response(response_text, previous_plan=None):
    values, errors = _sections(response_text, ("OUTCOME_CHECK", "PLAN", "SUBGOAL", "EXPECTED_OUTCOME", "ACTION"))
    check = values.get("OUTCOME_CHECK", "")
    if check.split(";", 1)[0].strip().upper() not in {"FIRST", "SUPPORTED", "CONTRADICTED", "UNCERTAIN"}:
        errors.append("invalid OUTCOME_CHECK status")
    if ";" not in check or not check.split(";", 1)[1].strip():
        errors.append("OUTCOME_CHECK needs a status and evidence separated by a semicolon")
    plan = values.get("PLAN")
    if plan and plan.upper() == "UNCHANGED":
        if not previous_plan:
            errors.append("UNCHANGED requires an existing plan")
        plan = previous_plan
    action = values.get("ACTION")
    if action and "\n" in action:
        errors.append("ACTION must contain one line")
    return {"plan": plan, "subgoal": values.get("SUBGOAL"),
            "expected_outcome": values.get("EXPECTED_OUTCOME"), "outcome_check": check,
            "action": action, "valid": not errors, "errors": errors}


def parse_reflection_response(response_text, max_lessons):
    values, errors = _sections(response_text, ("STATUS", "EVIDENCE", "LESSONS"))
    status = values.get("STATUS", "").upper()
    if status not in {"CHANGE", "NO_CHANGE", "UNCERTAIN"}:
        errors.append("invalid reflection status")
    lessons = []
    for line in values.get("LESSONS", "").splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned and cleaned.upper() != "NONE":
            lessons.append(_compact_text(cleaned, 16, 100))
    if status == "CHANGE" and not lessons:
        errors.append("CHANGE requires a lesson")
    if status in {"NO_CHANGE", "UNCERTAIN"} and lessons:
        errors.append("NO_CHANGE/UNCERTAIN require LESSONS: NONE")
    if len(lessons) > max_lessons:
        errors.append("too many lessons")
    return {"status": status, "evidence": values.get("EVIDENCE"), "lessons": lessons[:max_lessons],
            "valid": not errors, "errors": errors}


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
        self.transitions = []
        self.current_observation_text = ""
        self.request_metadata = {}

    def _observe(self, observation):
        self.current_observation_text = observation_text(observation)

    def _state_text(self):
        return (f"Previous plan: {self.plan or 'None'}\n"
                f"Previous subgoal: {self.subgoal or 'None'}\n"
                f"Prediction for the last action: {self.expected_outcome or 'None'}\n"
                "Retained lessons (hypotheses, check against evidence): " + "; ".join(self.lessons))

    def _actor_instruction(self):
        return ("Choose one allowed action. Check the last prediction against the visible transition; "
                "revise a contradicted plan. Avoid repeating a failed move or turn cycle without new evidence. "
                "Do not invent hidden state. Then predict the observable result of your next action.\n"
                "Use FIRST when there is no previous prediction.\n"
                + self._state_text() + "\nReply with these five headings, once each, in order:\n"
                "OUTCOME_CHECK: <FIRST, SUPPORTED, CONTRADICTED or UNCERTAIN>; <visible evidence, up to 12 words>\n"
                "PLAN: <up to 24 words, or UNCHANGED only if a plan exists>\n"
                "SUBGOAL: <up to 12 words>\n"
                "EXPECTED_OUTCOME: <up to 12 words>\n"
                "ACTION: <one allowed action>\nKeep the reply under 85 words; always include ACTION.")

    def _prepare_messages(self, instruction):
        maximum = int(getattr(self.client, "max_input_tokens", None) or 4096)
        reserve = int(self.config.agent.get("state_reserve_bytes", 1536))
        messages, metadata = shared_evidence(self.prompt_builder.system_prompt or "", self.current_observation_text,
                                             self.transitions, maximum, reserve)
        if len(("\n\n" + instruction).encode("utf-8")) > reserve:
            raise ValueError("Agent state/format exceeds the common state reserve.")
        messages[-1].content += "\n\n" + instruction
        metadata["input_token_upper_bound"] = input_token_upper_bound(messages)
        self.request_metadata = metadata
        return messages

    def _call_model(self, messages, role):
        reservation = None
        max_attempts = max(1, int(getattr(self.client, "max_retries", 1)))
        maximum = int(getattr(self.client, "max_input_tokens", None) or 4096)
        if input_token_upper_bound(messages) > maximum:
            raise ValueError("Prepared request exceeds the input bound.")
        saved_messages = [{"role": message.role, "content": message.content} for message in messages]
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
                                    "error": type(error).__name__, "messages": saved_messages})
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
                                "stop_reason": status, "call": call_log, "messages": saved_messages,
                                "response_metadata": dict(getattr(self.client, "last_response_metadata", {})),
                                "prompt_metadata": dict(self.request_metadata)})
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
        self.prompt_builder.update_observation(obs)
        self._observe(obs)
        reflection = self._maybe_reflect()

        messages = self._prepare_messages(self._actor_instruction())

        response, call_log = self._call_model(messages, "actor")
        parsed = parse_plan_response(response.completion, previous_plan=self.plan)
        if parsed["valid"] and parsed["plan"]:
            self.plan = _compact_text(parsed["plan"], self.max_plan_words, 160)
        if parsed["valid"] and parsed["subgoal"]:
            self.subgoal = _compact_text(parsed["subgoal"], self.max_subgoal_words, 90)
        if parsed["valid"] and parsed["expected_outcome"]:
            self.expected_outcome = _compact_text(parsed["expected_outcome"], self.max_subgoal_words, 90)

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

    def record_environment_feedback(self, applied_action, invalid_action, reward, terminated, truncated,
                                    observation_after=None, requested_action=None):
        """Feed the action actually executed and literal observation changes to all agents."""
        if not self.last_trace:
            raise RuntimeError("Cannot record feedback before a decision.")
        after = observation_text(observation_after) if observation_after is not None else self.current_observation_text
        before_lines = self.current_observation_text.splitlines()
        after_lines = after.splitlines()
        feedback = {"step": self.step_number - 1, "requested_action": str(requested_action or "")[:120],
                    "applied_action": applied_action, "invalid_action": bool(invalid_action),
                    "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                    "observation_changed": after != self.current_observation_text,
                    "no_longer_observed": [line for line in before_lines if line not in after_lines],
                    "newly_observed": [line for line in after_lines if line not in before_lines]}
        self.last_trace[-1]["environment"] = feedback
        if invalid_action:
            # The prediction concerned the requested action, not the fallback.
            self.expected_outcome = None
        self.transitions.append(feedback)
        self.transitions = self.transitions[-self.max_recent_observations:]


class LoggedNaiveAgent(PlanningAgent):
    """Action-only baseline with the same evidence, feedback and normalization."""

    def act(self, obs, prev_action=None):
        if prev_action is not None:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        self._observe(obs)
        messages = self._prepare_messages("Output exactly one allowed action and no other text.")
        response, call_log = self._call_model(messages, "actor")
        parsed_action = str(response.completion or "").strip()
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
        instruction = (
            "You are a critic for the game-playing agent. Review the previous plan and prediction against "
            "the observed transitions, applied actions, invalid-action flags, reward and current observation. "
            "Reward zero alone does not prove no progress. Do not infer hidden state. Correct only with evidence; "
            "otherwise choose NO_CHANGE or UNCERTAIN and retain prior lessons.\n"
            + self._state_text() + "\nReply with these headings once, in order:\n"
            "STATUS: <CHANGE, NO_CHANGE or UNCERTAIN>\n"
            "EVIDENCE: <up to 24 words citing visible evidence>\n"
            f"LESSONS: <NONE, or up to {self.max_lessons} bullet lessons of at most 16 words each>\n"
            "Keep the reply under 85 words."
        )
        messages = self._prepare_messages(instruction)
        response, call_log = self._call_model(messages, "critic")
        parsed = parse_reflection_response(response.completion, self.max_lessons)
        if parsed["valid"] and parsed["status"] == "CHANGE":
            self.lessons = parsed["lessons"]
        return {
            "raw_response": response.completion,
            "stop_reason": response.stop_reason,
            "parse": parsed,
            "call": call_log,
            "lessons_after_update": list(self.lessons),
        }
