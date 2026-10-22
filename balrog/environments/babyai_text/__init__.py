from .clean_lang_wrapper import BabyAITextCleanLangWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up a portable object directly in front when your hands are empty",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}


def get_instruction_prompt(env, mission="BabyAI-MixedTrainLocal-v0"):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.

Tips:
- Use 'pick up' to take a portable object directly in front; use 'toggle' to open or close a front door.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    return instruction_prompt
