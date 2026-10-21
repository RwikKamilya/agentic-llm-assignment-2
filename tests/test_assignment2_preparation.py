"""Offline boundary checks. All model responses and game environments are synthetic."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from assignment2.agents import LLMResponse, PlanningAgent, ReflectivePlanningAgent
from assignment2.budget import BudgetLedger, paid_run_lock
from assignment2.analysis import main as analyze_main
from assignment2.migrate_budget import migrate_ledger
from assignment2.runner import ExperimentRunner, execution_source_hashes, source_provenance
from assignment2.experiment_plan import planned_budget, planned_episodes
from assignment2.prompts import input_token_upper_bound, trim_oldest_history
from assignment2.storage import atomic_json_write
from assignment2.validation import IncompletePhaseError, validate_run_directory
from balrog.client import FatalClientError, OpenAIWrapper
from balrog.prompt_builder.history import HistoryPromptBuilder, Message

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'assignment2/config/experiment.yaml'


class FifteenDollarPlanTests(unittest.TestCase):
    def test_actual_manifest_cost_includes_variability_critics_and_both_attempts(self):
        config = OmegaConf.load(CONFIG)
        report = planned_budget(config, .203296, 0.)
        self.assertEqual(report['logical_calls'], {'smoke': 24, 'final': 6841})
        self.assertAlmostEqual(report['max_new_cost_usd'], 10.544640)
        self.assertAlmostEqual(report['remaining_after_plan_usd'], 4.252064)
        self.assertTrue(report['fits_with_recovery_reserve'])
        self.assertFalse(planned_budget(config, 4., 0.)['fits_with_recovery_reserve'])
        self.assertEqual(len(planned_episodes(config, 'final', list(config.models))), 138)

    def test_old_history_is_trimmed_but_instructions_current_state_and_format_survive(self):
        original = [Message('user', 'Game rules and allowed actions.'),
                    Message('user', 'oldest history ' * 800), Message('assistant', 'old action'),
                    Message('user', 'recent history ' * 30), Message('assistant', 'recent action'),
                    Message('user', 'Current observation: key. State: inspect. Reply: ACTION: command')]
        prepared, metadata = trim_oldest_history(original, 4096)
        self.assertLessEqual(input_token_upper_bound(prepared), 4096)
        self.assertEqual(metadata['trimmed_history_messages'], 2)
        self.assertEqual(prepared[0].content, original[0].content)
        self.assertEqual(prepared[-1].content, original[-1].content)
        self.assertEqual(len(original), 6)

    def test_actor_and_critic_fit_with_full_history_and_compact_valid_replies(self):
        config = OmegaConf.load(ROOT / 'balrog/config/config.yaml')
        config.agent.reflection_interval = 4
        class Client:
            max_input_tokens = 4096
            seen = []
            def generate(self, messages):
                self.seen.append(messages)
                self_test.assertLessEqual(input_token_upper_bound(messages), 4096)
                self_test.assertIn('GAME RULES', messages[0].content)
                self_test.assertIn('CURRENT', messages[-1].content)
                reply = ('LESSONS:\n- Inspect visible objects before moving.'
                         if 'You are a critic' in messages[-1].content else
                         'PLAN: inspect\nSUBGOAL: approach key\nEXPECTED_OUTCOME: key nearby\nACTION: go forward')
                self_test.assertLessEqual(len(reply.encode('utf-8')), 256)
                return LLMResponse('offline', reply, 'stop', 100, 30, None)
        self_test = self
        for model in ['openai_mini', 'qwen_7b_local']:
            client = Client()
            client.seen = []
            builder = HistoryPromptBuilder(max_text_history=6, max_image_history=0, max_cot_history=0)
            builder.update_instruction_prompt('GAME RULES: ' + 'allowed action. ' * 90)
            agent = ReflectivePlanningAgent(lambda: client, builder, config)
            for index in range(6):
                obs = {'text': {'long_term_context': 'CURRENT ' + 'visible tile ' * 80,
                                'short_term_context': ''}, 'image': None}
                agent.act(obs, prev_action='go forward' if index else None)
            self.assertEqual(len(client.seen), 7)
            self.assertTrue(agent.last_trace[4]['reflection']['parse']['valid'])
            self.assertTrue(all(trace['actor']['parse']['valid'] for trace in agent.last_trace))


class DeploymentProvenanceTests(unittest.TestCase):
    def test_verified_source_commit_is_separate_from_checkout_head(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'assignment2').mkdir()
            source = root / 'assignment2/runner.py'
            source.write_text('frozen source')
            receipt = {'source_commit': 'a' * 40,
                       'files': {'assignment2/runner.py': hashlib.sha256(source.read_bytes()).hexdigest()}}
            (root / 'assignment2/deployed_version.json').write_text(json.dumps(receipt))
            with patch('assignment2.runner.git_commit', return_value='b' * 40):
                source_commit, checkout_commit, receipt_hash = source_provenance(root)
                self.assertEqual(source_commit, 'a' * 40)
                self.assertEqual(checkout_commit, 'b' * 40)
                self.assertEqual(len(receipt_hash), 64)
                source.write_text('changed after deployment')
                with self.assertRaises(ValueError):
                    source_provenance(root)

    def test_historical_backups_are_not_executed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'assignment2/backup').mkdir(parents=True)
            (root / 'assignment2/runner.py').write_text('source')
            (root / 'assignment2/backup/runner.py').write_text('historical source')
            self.assertEqual(list(execution_source_hashes(root)), ['assignment2/runner.py'])


def response(content='go forward', input_tokens=100, output_tokens=20):
    usage = None if input_tokens is None else SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason='stop')], usage=usage)


def client_config():
    config = OmegaConf.load(ROOT / 'balrog/config/config.yaml').client
    config.model_id = 'gpt-4o-mini-2024-07-18'
    config.max_retries = 2
    config.delay = 0
    config.max_input_tokens = 4096
    config.generate_kwargs.max_tokens = 256
    return config


class OpenAIAccountingTests(unittest.TestCase):
    def make_client(self, results):
        client = OpenAIWrapper(client_config())
        client._initialized = True
        client.client = Mock()
        client.client.chat.completions.create.side_effect = results
        return client

    def test_official_endpoint_timeout_and_no_sdk_retries(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-only', 'OPENAI_BASE_URL': 'https://invalid.example/v1'}):
            with patch('balrog.client.OpenAI') as sdk:
                client = OpenAIWrapper(client_config())
                client.validate_request([Message('user', 'hello')])
                self.assertEqual(sdk.call_args.kwargs['base_url'], 'https://api.openai.com/v1')
                self.assertEqual(sdk.call_args.kwargs['timeout'], 60)
                self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FatalClientError):
                OpenAIWrapper(client_config()).validate_request([Message('user', 'hello')])

    def test_oversized_unicode_and_framing_rejected_before_request(self):
        client = self.make_client([response()])
        with self.assertRaises(FatalClientError):
            client.generate([Message('user', '界' * 2800)])
        with self.assertRaises(FatalClientError):
            client.generate([Message('user', 'x' * 8190)])
        self.assertEqual(client.client.chat.completions.create.call_count, 0)
        self.assertEqual(client.last_call_attempts, 0)

    def test_two_total_attempts_and_fatal_auth_does_not_retry(self):
        client = self.make_client([TimeoutError(), response()])
        self.assertEqual(client.generate([Message('user', 'hello')]).input_tokens, 100)
        self.assertEqual(client.last_call_attempts, 2)
        self.assertEqual(len(client.last_attempt_usage), 2)
        client = self.make_client([TimeoutError(), TimeoutError(), response()])
        with self.assertRaises(RuntimeError):
            client.generate([Message('user', 'hello')])
        self.assertEqual(client.client.chat.completions.create.call_count, 2)
        error = RuntimeError('do not log request bodies')
        error.status_code = 401
        client = self.make_client([error, response()])
        with self.assertRaises(FatalClientError):
            client.generate([Message('user', 'hello')])
        self.assertEqual(client.last_call_attempts, 1)
        self.assertNotIn('do not log', str(client.last_retry_errors))

    def test_empty_paid_response_keeps_usage_without_retry(self):
        client = self.make_client([response(content='')])
        result = client.generate([Message('user', 'hello')])
        self.assertEqual(result.stop_reason, 'error_empty_response')
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(client.last_call_attempts, 1)
        client = self.make_client([response(input_tokens=None)])
        self.assertIsNone(client.generate([Message('user', 'hello')]).input_tokens)

    def test_actor_critic_retry_costs_persist_and_unknown_usage_is_charged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'ledger.json'
            ledger = BudgetLedger.load_or_create(15, 4096, 256, .15, .60, path)
            config = OmegaConf.load(ROOT / 'balrog/config/config.yaml')
            config.agent.reflection_interval = 1
            results = [response('PLAN: explore\nSUBGOAL: move\nEXPECTED_OUTCOME: room\nACTION: go forward'),
                       TimeoutError(), response('LESSONS:\n- Look first.'),
                       response('PLAN: explore\nSUBGOAL: move\nEXPECTED_OUTCOME: room\nACTION: go forward')]
            client = self.make_client(results)
            builder = HistoryPromptBuilder(max_text_history=6, max_image_history=0, max_cot_history=0)
            agent = ReflectivePlanningAgent(lambda: client, builder, config, ledger)
            builder.update_instruction_prompt('Use a listed action.')
            obs = {'text': {'long_term_context': 'Room', 'short_term_context': ''}, 'image': None}
            agent.act(obs)
            agent.act(obs, prev_action='go forward')
            self.assertEqual([call['role'] for call in ledger.calls], ['actor', 'critic', 'actor'])
            self.assertEqual(ledger.calls[1]['attempts'], 2)
            expected = 3 * (100 * .15 + 20 * .60) / 1e6 + ledger.upper_bound_for_call()
            self.assertAlmostEqual(ledger.estimated_spent_usd, expected)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['calls'][1]['retry_errors'], ['TimeoutError (HTTP None)'])
            reservation = ledger.reserve('actor', 2)
            ledger.record(reservation, None, None, 'exception', attempts=2)
            self.assertAlmostEqual(ledger.calls[-1]['estimated_cost_usd'], 2 * ledger.upper_bound_for_call())

    def test_lock_and_backed_up_migration_preserve_every_charge_and_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'api_spending_ledger.json'
            state = {'schema_version': 1, 'ceiling_usd': 10., 'estimated_spent_usd': 2.3,
                     'reserved_usd': .8, 'calls': [{'provider': 'gemini', 'estimated_cost_usd': 2.3}]}
            atomic_json_write(path, state)
            original = path.read_bytes()
            with paid_run_lock(path):
                with self.assertRaises(RuntimeError):
                    migrate_ledger(path, OmegaConf.load(CONFIG).budget)
                self.assertTrue(path.with_suffix('.lock').exists())
            backup = migrate_ledger(path, OmegaConf.load(CONFIG).budget)
            self.assertEqual(backup.read_bytes(), original)
            migrated = json.loads(path.read_text())
            for field in ['estimated_spent_usd', 'reserved_usd', 'calls']:
                self.assertEqual(migrated[field], state[field])
            resumed = BudgetLedger.load_or_create(15, 4096, 256, .15, .60, path)
            self.assertAlmostEqual(resumed.reserved_usd, .8)
            self.assertIsNone(migrate_ledger(path, OmegaConf.load(CONFIG).budget))


class FakeEnvironment:
    max_steps = 32

    def __init__(self):
        self.failed_candidates = []
        self.steps = 0
        self.reset_kwargs = None
        self.unwrapped = SimpleNamespace(action_kinds=['goto'], task_selection_attempts=1)

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        return {'mission': 'go to the key', 'text': {'long_term_context': 'Room', 'short_term_context': ''}, 'image': None}, {}

    def get_instruction_prompt(self, instructions=None):
        return 'Choose go forward.'

    def check_action_validity(self, action):
        return action

    def step(self, action):
        self.steps += 1
        return self.reset()[0], 0., self.steps >= 5, False, {}

    def get_stats(self):
        return {'progression': 0.}

    def close(self):
        pass


class RunnerPreparationTests(unittest.TestCase):
    def setUp(self):
        # A Slurm diagnostic may inherit the production ledger location. All
        # synthetic runner tests must use their own temporary output root.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('ASSIGNMENT2_BUDGET_LEDGER_PATH', None)
        os.environ.pop('ASSIGNMENT2_OUTPUT_ROOT', None)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = OmegaConf.load(CONFIG)
        self.config.output_root = str(Path(self.temporary.name) / 'results')

    def runner(self, phase='smoke', models=None, resume=None, name='run'):
        return ExperimentRunner(ROOT, self.config, phase, resume_directory=resume,
                                run_directory=None if resume else Path(self.temporary.name) / name,
                                model_names=models or ['qwen_7b_local'])

    def test_phase_matrices_disjoint_seeds_and_effective_fingerprints(self):
        for phase, models, count in [('smoke',['openai_mini'],6), ('smoke',['qwen_7b_local'],6),
                                     ('final',['openai_mini'],69), ('final',['qwen_7b_local'],69)]:
            runner = self.runner(phase, models, name=f'{phase}-{count}-{models}')
            self.assertEqual(len(runner._episode_specs()), count)
            for spec in runner._episode_specs():
                config = runner._effective_config(spec)
                self.assertEqual(config.eval.max_steps_per_episode,
                                 (5 if spec['agent'] == 'reflective_planning' else 3) if phase == 'smoke' else (64 if spec['benchmark'] == 'babyai' else 128))
            if phase == 'final':
                self.assertEqual(sum('variability_subset' in s for s in runner._episode_specs()), 9)
        runner = self.runner(name='fingerprint')
        spec = runner._episode_specs()[0]
        config = runner._effective_config(spec)
        original = runner._episode_key(spec, config)
        config.eval.max_steps_per_episode += 1
        self.assertNotEqual(original, runner._episode_key(spec, config))
        with self.assertRaises(ValueError):
            self.runner(phase='pilot', name='disabled-pilot')
        with self.assertRaises(ValueError):
            self.runner(models=['openai_mini', 'qwen_7b_local'], name='combined')
        seeds = [set(int(e.seed) for e in self.config.benchmarks[benchmark].episodes[phase])
                 for benchmark in ['babyai','crafter'] for phase in ['smoke','development','pilot','final']]
        for i, left in enumerate(seeds):
            for right in seeds[i+1:]:
                self.assertTrue(left.isdisjoint(right))

    def fake_agent(self, config, budget_ledger=None):
        class Client:
            def generate(self, messages):
                return LLMResponse('synthetic-test-client',
                                   'PLAN: explore\nSUBGOAL: move\nEXPECTED_OUTCOME: room\nACTION: go forward'
                                   if not any('You are a critic' in message.content for message in messages) else 'LESSONS:\n- Look first.',
                                   'stop', 100, 20, None)
        from assignment2.agents import LoggedNaiveAgent
        kind = {'naive': LoggedNaiveAgent, 'planning': PlanningAgent, 'reflective_planning': ReflectivePlanningAgent}
        builder = HistoryPromptBuilder(max_text_history=6, max_image_history=0, max_cot_history=0)
        return kind[config.agent.type](lambda: Client(), builder, config, budget_ledger)

    def test_offline_run_trace_validation_resume_and_no_paid_ledger_for_qwen(self):
        runner = self.runner(models=['qwen_7b_local'])
        with patch('assignment2.runner.make_env', side_effect=lambda *args: FakeEnvironment()):
            with patch('assignment2.runner.AgentFactory') as factory:
                factory.side_effect = lambda config: SimpleNamespace(create_agent=lambda **kw: self.fake_agent(config, **kw))
                directory = runner.run()
                self.assertEqual(validate_run_directory(directory)['completed'], 6)
                self.assertFalse(runner.api_ledger_path.exists())
                self.assertTrue(all(call['estimated_cost_usd'] == 0 for call in runner.local_usage_ledger.calls))
                resumed = self.runner(models=['qwen_7b_local'], resume=directory)
                resumed.run()
                self.assertEqual(factory.call_count, 6)
        paths = sorted((directory / 'episodes').glob('*.json'))
        changed = json.loads(paths[0].read_text())
        changed['spec']['seed'] += 1
        atomic_json_write(paths[0], changed)
        with self.assertRaises(IncompletePhaseError):
            validate_run_directory(directory)  # Count is still six; identity is wrong.

    def test_incomplete_phase_stops_and_raises_without_fake_completion(self):
        runner = self.runner(models=['qwen_7b_local'])
        with patch('assignment2.runner.AgentFactory', side_effect=RuntimeError('offline injected failure')):
            with self.assertRaises(IncompletePhaseError):
                runner.run()
        paths = list((runner.run_directory / 'episodes').glob('*.json'))
        self.assertEqual(len(paths), 1)
        self.assertEqual(json.loads(paths[0].read_text())['status'], 'incomplete')

    def test_manual_resume_cannot_repeat_whole_episodes_without_a_bound(self):
        runner = self.runner(models=['qwen_7b_local'])
        key = 'bounded-recovery'
        atomic_json_write(runner._next_attempt_path(key), {'status': 'incomplete'})
        atomic_json_write(runner._next_attempt_path(key), {'status': 'incomplete'})
        with self.assertRaises(IncompletePhaseError):
            runner._next_attempt_path(key)

    def test_openai_runner_charges_actor_and_critic_and_accepts_zero_game_success(self):
        runner = self.runner(models=['openai_mini'])
        def reply(**kwargs):
            text = '\n'.join(part['text'] for message in kwargs['messages'] for part in message['content'])
            if 'You are a critic' in text:
                return response('LESSONS:\n- Observe before moving.')
            if 'Reply using all four headings' in text:
                return response('PLAN: explore\nSUBGOAL: move\nEXPECTED_OUTCOME: room\nACTION: go forward')
            return response('go forward')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-only'}):
            with patch('balrog.client.OpenAI') as sdk:
                sdk.return_value.chat.completions.create.side_effect = reply
                with patch('assignment2.runner.make_env', side_effect=lambda *args: FakeEnvironment()):
                    directory = runner.run()
        calls = json.loads(runner.api_ledger_path.read_text())['calls']
        self.assertEqual(sum(c['role'] == 'actor' for c in calls), 22)
        self.assertEqual(sum(c['role'] == 'critic' for c in calls), 2)
        self.assertTrue(all(c['provider'] == 'openai' for c in calls))
        self.assertEqual(validate_run_directory(directory)['completed'], 6)
        reflective_path = next(path for path in (directory/'episodes').glob('*.json')
                               if json.loads(path.read_text())['spec']['agent'] == 'reflective_planning')
        reflective = json.loads(reflective_path.read_text())
        broken = copy.deepcopy(reflective)
        broken['steps'][4]['agent_trace']['reflection'] = None
        atomic_json_write(reflective_path, broken)
        with self.assertRaises(IncompletePhaseError):
            validate_run_directory(directory)
        atomic_json_write(reflective_path, reflective)
        record_path = next((directory / 'episodes').glob('*.json'))
        saved = json.loads(record_path.read_text())
        saved['usage']['calls'][0]['usage_available'] = False
        atomic_json_write(record_path, saved)
        with self.assertRaises(IncompletePhaseError):
            validate_run_directory(directory)

    def test_budget_exhaustion_stops_before_any_model_request(self):
        self.config.budget.total_ceiling_usd = .000001
        runner = self.runner(models=['openai_mini'])
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-only'}):
            with patch('balrog.client.OpenAI') as sdk:
                with patch('assignment2.runner.make_env', side_effect=lambda *args: FakeEnvironment()):
                    with self.assertRaises(IncompletePhaseError):
                        runner.run()
                sdk.return_value.chat.completions.create.assert_not_called()
        paths = list((runner.run_directory / 'episodes').glob('*.json'))
        self.assertEqual(len(paths), 0)

    def test_cli_failure_is_nonzero_and_launcher_checks_qwen_conditionally(self):
        with patch.dict(os.environ, {'ASSIGNMENT2_OUTPUT_ROOT': self.config.output_root}, clear=True):
            result = subprocess.run([sys.executable, '-m', 'assignment2.run_experiment', '--phase', 'smoke',
                                     '--models', 'openai_mini', '--run-directory', str(Path(self.temporary.name)/'cli')],
                                    cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Phase incomplete', result.stdout)
        script = (ROOT / 'assignment2/scripts/run_pipeline.sbatch').read_text()
        self.assertNotIn('== 10.0', script)
        self.assertNotIn('gemini_flash_lite', script)
        self.assertIn('-m assignment2.run_experiment', script)

    def test_analysis_does_not_read_or_plot_an_incomplete_phase(self):
        with patch.object(sys, 'argv', ['analysis', 'incomplete', '--phase', 'final', '--output', 'unused']):
            with patch('assignment2.analysis.validate_run_directory', side_effect=IncompletePhaseError('missing identity')):
                with patch('assignment2.analysis.load_records') as load:
                    with self.assertRaises(IncompletePhaseError):
                        analyze_main()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
