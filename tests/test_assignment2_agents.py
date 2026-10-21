import copy
import tempfile
import unittest
from pathlib import Path

from assignment2.agents import (
    LLMResponse,
    LoggedNaiveAgent,
    ModelCallError,
    PlanningAgent,
    ReflectivePlanningAgent,
    parse_plan_response,
)
from assignment2.budget import BudgetLedger
from balrog.prompt_builder.history import HistoryPromptBuilder


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class FakeConfig:
    def __init__(self, **agent_values):
        self.agent = AttributeDict(agent_values)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def generate(self, messages):
        self.messages.append(copy.deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse("mock", response, "stop", 10, 5, None)


def observation(text="You see a key."):
    return {"text": {"short_term_context": text, "long_term_context": text}, "image": None}


def actor_reply(plan="look around", action="go forward"):
    return f"PLAN: {plan}\nSUBGOAL: inspect the room\nEXPECTED_OUTCOME: a new visible object\nACTION: {action}"


class AssignmentAgentTests(unittest.TestCase):
    def make_config(self):
        return FakeConfig(
            max_plan_words=5,
            max_subgoal_words=4,
            max_recent_observations=2,
            reflection_interval=2,
            max_reflection_lessons=3,
        )

    def make_builder(self):
        builder = HistoryPromptBuilder(max_text_history=4, max_image_history=0, max_cot_history=0)
        builder.update_instruction_prompt("Choose from the available actions.")
        return builder

    def test_malformed_action_is_not_guessed(self):
        parsed = parse_plan_response("PLAN: explore\nSUBGOAL: inspect\nEXPECTED_OUTCOME: object")
        self.assertFalse(parsed["valid"])
        self.assertIsNone(parsed["action"])
        self.assertIn("missing ACTION section", parsed["errors"])

    def test_planning_state_is_bounded_and_resets_between_episodes(self):
        long_plan = "one two three four five six seven"
        client = FakeClient([actor_reply(plan=long_plan)])
        agent = PlanningAgent(lambda: client, self.make_builder(), self.make_config())
        response = agent.act(observation())
        self.assertEqual(response.completion, "go forward")
        self.assertEqual(agent.plan, "one two three four five")
        self.assertEqual(agent.subgoal, "inspect the room")
        self.assertEqual(len(agent.last_trace), 1)

        agent.reset()
        self.assertIsNone(agent.plan)
        self.assertEqual(agent.lessons, [])
        self.assertEqual(agent.last_trace, [])

    def test_prompt_history_is_not_mutated_by_agent_instruction(self):
        client = FakeClient([actor_reply()])
        builder = self.make_builder()
        agent = PlanningAgent(lambda: client, builder, self.make_config())
        agent.act(observation("Observed text stays unchanged."))
        stored_observation = next(event for event in builder._events if event["type"] == "observation")
        self.assertEqual(stored_observation["text"], "Observed text stays unchanged.")
        self.assertIn("PLAN:", client.messages[0][-1].content)

    def test_reflection_runs_at_fixed_interval_and_keeps_three_lessons(self):
        responses = [
            actor_reply(action="a1"),
            actor_reply(action="a2"),
            "LESSONS:\n- Observe before moving.\n- Avoid repeating blocked actions.\n"
            "- Use the listed commands.\n- This must be dropped.",
            actor_reply(action="a3"),
        ]
        client = FakeClient(responses)
        agent = ReflectivePlanningAgent(lambda: client, self.make_builder(), self.make_config())
        agent.act(observation("first"))
        agent.act(observation("second"), prev_action="a1")
        agent.act(observation("third"), prev_action="a2")
        self.assertEqual(len(agent.lessons), 3)
        self.assertIsNone(agent.last_trace[1]["reflection"])
        self.assertTrue(agent.last_trace[2]["reflection"]["parse"]["valid"])
        self.assertIn("Prior action: a1", "\n".join(message.content for message in client.messages[2]))

    def test_model_error_stays_explicit(self):
        client = FakeClient([RuntimeError("network unavailable")])
        agent = PlanningAgent(lambda: client, self.make_builder(), self.make_config())
        with self.assertRaises(ModelCallError):
            agent.act(observation())
        self.assertEqual(agent.call_trace[-1]["error"], "RuntimeError")

    def test_critic_trace_survives_a_failed_following_actor_call(self):
        config = self.make_config()
        config.agent["reflection_interval"] = 1
        client = FakeClient([actor_reply(), "LESSONS:\n- Check the door.", RuntimeError("offline failure")])
        agent = ReflectivePlanningAgent(lambda: client, self.make_builder(), config)
        agent.act(observation())
        with self.assertRaises(ModelCallError):
            agent.act(observation(), prev_action="go forward")
        self.assertEqual(agent.call_trace[1]["role"], "critic")
        self.assertIn("Check the door", agent.call_trace[1]["raw_response"])

    def test_logged_naive_keeps_the_existing_naive_action_filter(self):
        client = FakeClient(["ACTION: go forward!!!"])
        agent = LoggedNaiveAgent(lambda: client, self.make_builder(), self.make_config())
        response = agent.act(observation())
        self.assertEqual(response.completion, "ACTION: go forward")
        self.assertEqual(agent.last_trace[0]["actor"]["raw_response"], "ACTION: go forward!!!")

    def test_budget_stops_before_the_next_request(self):
        ledger = BudgetLedger(
            ceiling_usd=0.00001,
            max_input_tokens=100,
            max_output_tokens=100,
            input_price_per_million=1,
            output_price_per_million=1,
        )
        client = FakeClient([actor_reply()])
        agent = PlanningAgent(lambda: client, self.make_builder(), self.make_config(), budget_ledger=ledger)
        with self.assertRaises(Exception) as error:
            agent.act(observation())
        self.assertIn("Stopping before", str(error.exception))
        self.assertEqual(client.messages, [])

    def test_retry_attempts_and_errors_are_logged_and_costed(self):
        ledger = BudgetLedger(
            ceiling_usd=1.0,
            max_input_tokens=100,
            max_output_tokens=100,
            input_price_per_million=1,
            output_price_per_million=1,
        )
        client = FakeClient([actor_reply()])
        client.max_retries = 2
        client.last_call_attempts = 2
        client.last_retry_errors = ["TimeoutError: simulated timeout"]
        agent = PlanningAgent(lambda: client, self.make_builder(), self.make_config(), budget_ledger=ledger)
        agent.act(observation())
        call = agent.last_trace[0]["actor"]["call"]
        self.assertEqual(call["retries"], 1)
        self.assertEqual(call["retry_errors"], ["TimeoutError: simulated timeout"])
        self.assertGreater(call["estimated_cost_usd"], 0.0)

    def test_persistent_budget_is_not_reset_when_a_job_is_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "api_spending_ledger.json"
            ledger = BudgetLedger.load_or_create(
                ceiling_usd=1.0,
                max_input_tokens=500_000,
                max_output_tokens=0,
                input_price_per_million=1.0,
                output_price_per_million=0.0,
                state_path=state_path,
            )
            reservation = ledger.reserve("actor")
            ledger.record(reservation, 500_000, 0, "stop")

            resumed_ledger = BudgetLedger.load_or_create(
                ceiling_usd=1.0,
                max_input_tokens=500_000,
                max_output_tokens=0,
                input_price_per_million=1.0,
                output_price_per_million=0.0,
                state_path=state_path,
            )
            self.assertAlmostEqual(resumed_ledger.estimated_spent_usd, 0.5)
            with self.assertRaises(Exception):
                resumed_ledger.reserve("actor", max_attempts=2)

    def test_inflight_persistent_reservation_remains_charged_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "api_spending_ledger.json"
            ledger = BudgetLedger.load_or_create(
                ceiling_usd=1.0,
                max_input_tokens=500_000,
                max_output_tokens=0,
                input_price_per_million=1.0,
                output_price_per_million=0.0,
                state_path=state_path,
            )
            ledger.reserve("actor")

            resumed_ledger = BudgetLedger.load_or_create(
                ceiling_usd=1.0,
                max_input_tokens=500_000,
                max_output_tokens=0,
                input_price_per_million=1.0,
                output_price_per_million=0.0,
                state_path=state_path,
            )
            self.assertAlmostEqual(resumed_ledger.reserved_usd, 0.5)
            with self.assertRaises(Exception):
                resumed_ledger.reserve("actor", max_attempts=2)
