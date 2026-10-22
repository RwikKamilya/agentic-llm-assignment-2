"""Focused regression tests for failures that can invalidate the comparison."""

import copy
import unittest

from assignment2.agents import LoggedNaiveAgent, PlanningAgent, ReflectivePlanningAgent, parse_plan_response, parse_reflection_response
from assignment2.analysis import repeated_run_variability, validate_paired_resets
from assignment2.prompts import input_token_upper_bound
from assignment2.protocol import normalize_action
from balrog.prompt_builder.history import HistoryPromptBuilder
from test_assignment2_agents import FakeClient, FakeConfig, actor_reply, observation
from test_assignment2_records_and_analysis import record


class ProtocolTests(unittest.TestCase):
    def make_agent(self, kind, replies):
        config = FakeConfig(max_plan_words=24, max_subgoal_words=12, max_recent_observations=3,
                            reflection_interval=1, max_reflection_lessons=3, state_reserve_bytes=1536)
        builder = HistoryPromptBuilder(max_image_history=0)
        builder.update_instruction_prompt('Allowed actions: go forward; turn left; turn right.')
        client = FakeClient(replies)
        agent = kind(lambda: client, builder, config)
        return agent, client

    def test_normalization_is_identical_and_never_guesses_an_action(self):
        for candidate in ['Move South.', ' move   south ', '`Move South`', '"Move South."']:
            self.assertEqual(normalize_action(candidate, 'crafter'), 'Move South')
        for candidate in ['Move South-East', 'MoveSouth', 'Collect Wood', 'Do then Move East', 'ACTION: Do']:
            self.assertEqual(normalize_action(candidate, 'crafter'), candidate)
        self.assertEqual(normalize_action('PICK UP!', 'babyai'), 'pick up')

    def test_duplicate_empty_and_initial_unchanged_fields_are_rejected(self):
        for reply in [actor_reply() + '\nACTION: turn left', actor_reply().replace('SUBGOAL: inspect the room', 'SUBGOAL:'),
                      actor_reply(plan='UNCHANGED'), actor_reply().replace('OUTCOME_CHECK: UNCERTAIN', 'OUTCOME_CHECK: MAGIC')]:
            self.assertFalse(parse_plan_response(reply)['valid'])
        self.assertTrue(parse_plan_response(actor_reply(plan='UNCHANGED'), 'existing plan')['valid'])

    def test_malformed_actor_does_not_overwrite_valid_plan(self):
        agent, _ = self.make_agent(PlanningAgent, [actor_reply(plan='valid plan'), 'PLAN: invented state\nACTION: go forward'])
        agent.act(observation())
        result = agent.act(observation())
        self.assertEqual(result.completion, '')
        self.assertEqual(agent.plan, 'valid plan')

    def test_critic_can_retain_lessons_without_inventing_a_correction(self):
        for status in ['NO_CHANGE', 'UNCERTAIN']:
            critic = f'STATUS: {status}\nEVIDENCE: no conclusive transition\nLESSONS: NONE'
            self.assertTrue(parse_reflection_response(critic, 3)['valid'])
            agent, client = self.make_agent(ReflectivePlanningAgent, [actor_reply(), critic, actor_reply()])
            agent.lessons = ['Existing supported lesson.']
            agent.act(observation('key at front'))
            agent.record_environment_feedback('go forward', True, 0, False, False,
                                              observation('key remains at front'), 'invalid command')
            agent.act(observation('key remains at front'))
            prompt = '\n'.join(message.content for message in client.messages[1])
            self.assertIn('Previous plan: look around', prompt)
            self.assertIn('"applied_action": "go forward"', prompt)
            self.assertIn('"invalid_action": true', prompt)
            self.assertIn('key remains at front', prompt)
            self.assertEqual(agent.lessons, ['Existing supported lesson.'])

    def test_invalid_critic_preserves_lessons_and_actor_still_runs(self):
        agent, client = self.make_agent(ReflectivePlanningAgent, [actor_reply(), 'garbage', actor_reply()])
        agent.lessons = ['Keep this lesson.']
        agent.act(observation())
        agent.act(observation())
        self.assertEqual(len(client.messages), 3)
        self.assertFalse(agent.last_trace[1]['reflection']['parse']['valid'])
        self.assertEqual(agent.lessons, ['Keep this lesson.'])

    def test_same_transition_window_despite_different_reasoning_state(self):
        agents = []
        for kind in [LoggedNaiveAgent, PlanningAgent, ReflectivePlanningAgent]:
            agent, _ = self.make_agent(kind, [])
            agent._observe(observation('CURRENT visible key'))
            agent.transitions = [{'step': i, 'requested_action': 'turn left', 'applied_action': 'turn left',
                                 'invalid_action': False, 'reward': 0, 'observation_changed': True,
                                 'newly_observed': ['tile ' * 160], 'no_longer_observed': ['wall ' * 160]}
                                for i in range(3)]
            if kind != LoggedNaiveAgent:
                agent.plan = 'p' * 160
                agent.subgoal = 's' * 90
                agent.expected_outcome = 'e' * 90
                agent.lessons = ['l' * 100] * 3
            instruction = 'Output one action.' if kind == LoggedNaiveAgent else agent._actor_instruction()
            messages = agent._prepare_messages(instruction)
            self.assertLessEqual(input_token_upper_bound(messages), 4096)
            agents.append((agent, messages))
        self.assertEqual(len({agent.request_metadata['retained_transitions'] for agent, _ in agents}), 1)
        self.assertEqual(agents[0][1][-1].content.split('\n\n')[0], agents[1][1][-1].content.split('\n\n')[0])
        self.assertGreater(agents[0][0].request_metadata['trimmed_history_messages'], 0)

    def test_repeats_include_original_without_changing_primary_sample_size(self):
        records = [record('m', 'planning', 0), record('m', 'planning', 1, repeat_index=1, variability=True),
                   record('m', 'planning', 1, repeat_index=2, variability=True)]
        row = repeated_run_variability(records)[0]
        self.assertEqual(row['n_repeats'], 3)
        self.assertAlmostEqual(row['mean_progression'], 2/3)

    def test_pairing_rejects_changed_initial_observation(self):
        first = record('m', 'naive', 0)
        first.update(schema_version=2, reset={'mission': 'get key'}, steps=[{'observation': {'x': 'key'}}])
        second = copy.deepcopy(first)
        second['spec']['agent'] = 'planning'
        second['steps'][0]['observation'] = {'x': 'door'}
        with self.assertRaises(ValueError):
            validate_paired_resets([first, second])


if __name__ == '__main__':
    unittest.main()
