"""Explicit $10 -> $60 migration; run in an allocation before cloud evaluation."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.budget import BudgetLedger, paid_run_lock
from assignment2.storage import atomic_json_write


def migrate_ledger(path, budget_config):
    path = Path(path)
    with paid_run_lock(path):
        if not path.exists():
            raise ValueError("No existing ledger to migrate; a new run creates one without migration.")
        original = path.read_bytes()
        state = json.loads(original)
        target = BudgetLedger(
            budget_config.total_ceiling_usd, budget_config.max_input_tokens_per_call,
            budget_config.max_output_tokens_per_call, budget_config.conservative_input_usd_per_million,
            budget_config.conservative_output_usd_per_million,
        )
        if state.get("ceiling_usd") == 60.0 and state.get("pricing_policy") == target.pricing_policy():
            print("Ledger already has the requested policy; no changes made.")
            return None
        if state.get("schema_version") != 1 or state.get("ceiling_usd") != 10.0 or target.ceiling_usd != 60.0:
            raise ValueError("Only the reviewed schema-1 $10 -> $60 migration is supported.")
        for key in ["estimated_spent_usd", "reserved_usd"]:
            if key not in state or not math.isfinite(state[key]) or state[key] < 0:
                raise ValueError(f"Invalid historical ledger balance: {key}")
        if not isinstance(state.get("calls"), list):
            raise ValueError("Historical ledger has no call list.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"{path.name}.before-60usd-{timestamp}.bak")
        with backup.open("xb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        state.setdefault("migrations", []).append({
            "at": timestamp, "backup": str(backup), "from_ceiling_usd": 10.0,
            "to_ceiling_usd": 60.0, "previous_pricing_policy": state.get("pricing_policy"),
            "note": "Historical calls, estimated spend and outstanding reservations preserved verbatim; "
                    "OpenAI prices apply only to new calls. Historical provider costs are not repriced.",
        })
        state["ceiling_usd"] = 60.0
        state["pricing_policy"] = target.pricing_policy()
        atomic_json_write(path, state)
        print(f"Migrated ledger; original bytes preserved at {backup}")
        return backup


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run ledger migration inside an allocation, not on calcium.")
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "assignment2/config/experiment.yaml")
    path = os.environ.get("ASSIGNMENT2_BUDGET_LEDGER_PATH")
    if not path:
        raise SystemExit("Set ASSIGNMENT2_BUDGET_LEDGER_PATH to the existing cumulative ledger.")
    migrate_ledger(path, config.budget)


if __name__ == "__main__":
    main()
