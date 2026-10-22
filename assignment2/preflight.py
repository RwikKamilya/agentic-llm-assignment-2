"""No-API integration checks in the allocated evaluation environment.

Synthetic responses exercise real game wrappers. This is harness validation,
not model evaluation, and its output never enters the performance matrix.
"""

import json
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.agents import LLMResponse, ReflectivePlanningAgent
from assignment2.prompts import input_token_upper_bound
from assignment2.protocol import ACTIONS, game_instructions, normalize_action
from assignment2.runner import ExperimentRunner, make_env, source_provenance
from balrog.prompt_builder.history import HistoryPromptBuilder


class OfflineClient:
    max_input_tokens = 4096

    def __init__(self, action):
        self.action = action

    def generate(self, messages):
        if input_token_upper_bound(messages) > self.max_input_tokens:
            raise ValueError('Oversized integration-check prompt')
        if 'You are a critic' in messages[-1].content:
            reply = 'STATUS: UNCERTAIN\nEVIDENCE: Synthetic harness check, no model inference.\nLESSONS: NONE'
        else:
            reply = ('OUTCOME_CHECK: UNCERTAIN; synthetic harness check\nPLAN: inspect visible state\n'
                     'SUBGOAL: inspect\nEXPECTED_OUTCOME: uncertain\nACTION: ' + self.action)
        return LLMResponse('SYNTHETIC_PREFLIGHT', reply, 'stop', 0, 0, None)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run game integration checks inside a Slurm allocation.')
    root = Path(__file__).resolve().parents[1]
    source_provenance(root)
    study = OmegaConf.load(root / 'assignment2/config/experiment.yaml')
    checked = []
    with tempfile.TemporaryDirectory() as directory:
        runner = ExperimentRunner(root, study, 'final', run_directory=Path(directory) / 'check', model_names=['qwen_7b_local'])
        specs = [spec for spec in runner._episode_specs() if spec['agent'] == 'reflective_planning' and 'variability_subset' not in spec]
        for spec in specs:
            first = None
            config = runner._effective_config(spec)
            for repetition in range(2):
                env = make_env(spec['benchmark'], spec['task'], config)
                try:
                    obs, _ = env.reset() if spec['benchmark'] == 'crafter' else env.reset(seed=spec['seed'])
                    initial = (obs.get('mission'), dict(obs['text']))
                    if repetition and initial != first:
                        raise ValueError(f'Nonrepeatable initial observation: {spec}')
                    first = initial
                    actual_actions = list(env.env.language_action_space)
                    if actual_actions != ACTIONS[spec['benchmark']]:
                        raise ValueError('Protocol action list differs from the real game wrapper')
                    if spec['benchmark'] == 'babyai' and env.unwrapped.action_kinds[0].replace(' ', '_') != spec['task'].split('/')[1]:
                        raise ValueError('BabyAI task kind changed at reset')
                    action = 'turn left' if spec['benchmark'] == 'babyai' else 'Noop'
                    client = OfflineClient(action)
                    agent = ReflectivePlanningAgent(lambda: client, HistoryPromptBuilder(max_image_history=0), config)
                    agent.prompt_builder.update_instruction_prompt(game_instructions(spec['benchmark'], obs.get('mission')))
                    # Exercise the largest retained state as well as the critic schedule.
                    agent.plan = 'p' * 160
                    agent.subgoal = 's' * 90
                    agent.expected_outcome = 'e' * 90
                    agent.lessons = ['l' * 100] * 3
                    for _ in range(5):
                        reply = agent.act(obs)
                        applied = env.check_action_validity(normalize_action(reply.completion, spec['benchmark']))
                        obs, reward, terminated, truncated, _ = env.step(applied)
                        agent.record_environment_feedback(applied, False, reward, terminated, truncated,
                                                          observation_after=obs, requested_action=reply.completion)
                        if terminated or truncated:
                            break
                finally:
                    env.close()
            checked.append({key: spec[key] for key in ['benchmark', 'task', 'seed']})
    print(json.dumps({'status': 'PREFLIGHT_OK', 'model_evaluation': False, 'reset_settings_checked_twice': checked}, indent=2))


if __name__ == '__main__':
    main()
