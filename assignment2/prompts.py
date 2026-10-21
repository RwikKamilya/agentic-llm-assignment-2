"""Conservative text bounds shared by both experimental models."""

import copy


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
