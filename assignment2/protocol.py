"""Shared public game rules and conservative action normalization for protocol v2."""

import re


PROTOCOL_ID = "outcome_feedback_v2"
ACTIONS = {
    "babyai": ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"],
    "crafter": ["Noop", "Move West", "Move East", "Move North", "Move South", "Do", "Sleep",
                "Place Stone", "Place Table", "Place Furnace", "Place Plant", "Make Wood Pickaxe",
                "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", "Make Stone Sword", "Make Iron Sword"],
}


def normalize_action(candidate, benchmark):
    """Accept case, whitespace, a wrapper quote and terminal punctuation only.

    Never convert a direction, invent spaces, extract an action from prose, or
    turn an achievement name into a command. Invalid actions use BALROG's fallback.
    """
    raw = str(candidate or "").strip()
    cleaned = raw.strip("`\"'").strip().rstrip(".!;").strip().strip("`\"'").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for action in ACTIONS[benchmark]:
        if cleaned.casefold() == action.casefold():
            return action
    return raw


def game_instructions(benchmark, mission=None):
    actions = "; ".join(ACTIONS[benchmark])
    if benchmark == "babyai":
        rules = (
            f"BabyAI mission: {mission}. Positions are relative to your current facing. "
            "Turning changes your facing, not your tile. go forward moves one tile. "
            "pick up takes a portable object in the tile directly in front when your hands are empty. "
            "drop places your carried object in an empty front tile. toggle opens/closes the front door "
            "or interacts with a front object; it does not pick up an object. Locked doors need the matching key."
        )
    elif benchmark == "crafter":
        rules = (
            "Crafter goal: survive and earn distinct achievements. Directions are compass directions. "
            "Move actions face that direction and attempt one step. Only cardinal moves exist. "
            "Do interacts with the tile in front: collect resources, drink water, or attack. "
            "Walking on grass does not collect it. Sleep restores energy. Placement needs materials and space. "
            "Wood tools need wood and a nearby table; stone tools also need stone; iron tools need "
            "wood, coal, iron and a nearby table and furnace. Achievements are outcomes, not commands: "
            "collect wood/sapling/drink/stone/coal/iron/diamond; eat cow/plant; defeat zombie/skeleton; "
            "wake up; place table/plant/stone/furnace; make wood/stone/iron pickaxes and swords."
        )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return rules + "\nAllowed actions (copy exactly one): " + actions
