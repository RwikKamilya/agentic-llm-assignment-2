"""Conservative text bounds shared by both experimental models."""

import copy
import json

from balrog.prompt_builder.history import Message


def observation_text(observation):
    text = observation.get("text", {})
    return "\n".join(str(text.get(key) or "") for key in
                     ["short_term_context", "long_term_context"]).strip()


def shared_evidence(instructions, current, transitions, maximum, state_reserve):
    """Select evidence before adding agent state, using the same reserve for all.

    Observed line changes are literal text, not a model summary or hidden game
    state. The latest execution feedback and full current observation always fit
    or the run fails before making a request. Older complete transitions may go.
    """
    latest = {key: transitions[-1][key] for key in
              ["step", "requested_action", "applied_action", "invalid_action", "reward", "observation_changed"]} if transitions else None
    current_text = "Current observation:\n" + current + "\nLatest execution feedback: " + json.dumps(latest)
    messages = [Message("user", instructions)]
    for transition in transitions:
        messages.append(Message("user", "Observed transition: " + json.dumps(transition, ensure_ascii=False)))
    messages.append(Message("user", current_text))
    prepared, metadata = trim_oldest_history(messages, maximum - state_reserve)
    metadata.update({"prompt_policy": "shared_transition_window_fixed_state_reserve_v2",
                     "state_reserve_bytes": state_reserve,
                     "retained_transitions": len(prepared) - 2,
                     "evidence_token_upper_bound": input_token_upper_bound(prepared)})
    return prepared, metadata


def input_token_upper_bound(messages):
    # UTF-8 bytes bound text BPE tokens; padding includes roles/chat framing.
    bound = 32
    for message in messages:
        if message.attachment is not None:
            raise ValueError("Assignment 2 requires text-only prompts.")
        bound += len(message.content.encode("utf-8")) + len(message.role.encode("utf-8")) + 32
    return bound


def trim_oldest_history(messages, maximum):
    """Keep game instructions and the current observation/response format intact.

    Agents put required state/format instructions in the final message. Only
    intervening historical observations/actions may be removed, oldest first.
    """
    prepared = copy.deepcopy(messages)
    removed = 0
    while input_token_upper_bound(prepared) > maximum and len(prepared) > 2:
        del prepared[1]
        removed += 1
        # Remove the action paired with the discarded old observation too.
        if len(prepared) > 2 and prepared[1].role == "assistant":
            del prepared[1]
            removed += 1
    bound = input_token_upper_bound(prepared)
    if bound > maximum:
        raise ValueError("Required game instructions/current observation exceed the input bound after history trimming.")
    return prepared, {"input_token_upper_bound": bound, "trimmed_history_messages": removed,
                      "prompt_policy": "trim_oldest_history_preserve_instructions_and_current_v1"}
