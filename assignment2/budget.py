"""Conservative, provider-independent accounting for planned model calls."""

import json
import math
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from assignment2.storage import atomic_json_write


class BudgetExceeded(RuntimeError):
    """Raised before a request that would exceed the configured guardrail."""


@contextmanager
def paid_run_lock(ledger_path):
    """Serialize cloud runs and migrations, including direct Python launches.

    An abandoned lock requires human inspection of its owner and Slurm state.
    Never remove or take over an existing lock automatically.
    """
    lock = Path(ledger_path).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Paid-run lock already exists: {lock}; inspect its owner before recovery.") from error
    owner = {"token": uuid.uuid4().hex, "host": socket.gethostname(), "pid": os.getpid(),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        atomic_json_write(lock / "owner", owner)
        yield
    finally:
        # Release only this process's lock, never a replacement owner's lock.
        if (lock / "owner").exists():
            current = json.loads((lock / "owner").read_text())
            if current.get("token") == owner["token"]:
                (lock / "owner").unlink()
                lock.rmdir()


@dataclass
class CallReservation:
    role: str
    upper_bound_usd: float
    max_attempts: int


class BudgetLedger:
    """Reserve a worst-case amount before each model request.

    Provider APIs do not always report usage.  In that case the reservation is
    retained as the estimate instead of recording a misleading zero cost.
    """

    def __init__(
        self,
        ceiling_usd,
        max_input_tokens,
        max_output_tokens,
        input_price_per_million,
        output_price_per_million,
        state_path=None,
    ):
        self.ceiling_usd = float(ceiling_usd)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.input_price_per_million = float(input_price_per_million)
        self.output_price_per_million = float(output_price_per_million)
        self.reserved_usd = 0.0
        self.estimated_spent_usd = 0.0
        self.calls = []
        self.state_path = Path(state_path) if state_path else None
        self.migrations = []
        if not all(math.isfinite(value) and value >= 0 for value in
                   [self.ceiling_usd, self.input_price_per_million, self.output_price_per_million]):
            raise ValueError("Budget ceiling and prices must be finite and nonnegative.")
        if self.max_input_tokens < 1 or self.max_output_tokens < 0:
            raise ValueError("Invalid input/output token bounds.")

    def pricing_policy(self):
        return {"input_usd_per_million": self.input_price_per_million,
                "output_usd_per_million": self.output_price_per_million,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens}

    @classmethod
    def load_or_create(
        cls,
        ceiling_usd,
        max_input_tokens,
        max_output_tokens,
        input_price_per_million,
        output_price_per_million,
        state_path,
    ):
        """Load one durable budget ledger or create it before the first call.

        Reservations are persisted too. If a job ends while a provider call is
        in flight, its worst-case reservation remains charged rather than being
        silently made available again on resubmission.
        """
        state_path = Path(state_path)
        ledger = cls(
            ceiling_usd=ceiling_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            state_path=state_path,
        )
        if state_path.exists():
            try:
                snapshot = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Unreadable persistent budget ledger: {state_path}") from error
            ledger._restore(snapshot)
        else:
            ledger._persist()
        return ledger

    def _restore(self, snapshot):
        if snapshot.get("schema_version") != 1:
            raise RuntimeError("Persistent budget ledger has an unsupported schema version.")
        if float(snapshot.get("ceiling_usd", -1)) != self.ceiling_usd:
            raise RuntimeError(
                "Persistent budget ledger ceiling differs from the configured total ceiling. "
                "Refusing to reset or reinterpret prior spending."
            )
        if snapshot.get("pricing_policy") != self.pricing_policy():
            raise RuntimeError("Ledger pricing policy differs or is historical; explicit backed-up migration required.")
        self.estimated_spent_usd = float(snapshot.get("estimated_spent_usd", 0.0))
        self.reserved_usd = float(snapshot.get("reserved_usd", 0.0))
        if not all(math.isfinite(value) and value >= 0 for value in
                   [self.estimated_spent_usd, self.reserved_usd]):
            raise RuntimeError("Invalid persistent ledger balances.")
        self.calls = list(snapshot.get("calls", []))
        self.migrations = list(snapshot.get("migrations", []))

    def _persist(self):
        if self.state_path is not None:
            atomic_json_write(self.state_path, self.snapshot())

    def upper_bound_for_call(self):
        input_cost = self.max_input_tokens * self.input_price_per_million / 1_000_000
        output_cost = self.max_output_tokens * self.output_price_per_million / 1_000_000
        return input_cost + output_cost

    def reserve(self, role, max_attempts=1):
        max_attempts = max(1, int(max_attempts))
        upper_bound = self.upper_bound_for_call() * max_attempts
        projected_total = self.estimated_spent_usd + self.reserved_usd + upper_bound
        if projected_total > self.ceiling_usd:
            raise BudgetExceeded(
                f"Stopping before the ${self.ceiling_usd:.2f} ceiling: "
                f"the next {role} call could raise the estimate to ${projected_total:.4f}."
            )
        self.reserved_usd += upper_bound
        self._persist()
        return CallReservation(role=role, upper_bound_usd=upper_bound, max_attempts=max_attempts)

    def record(self, reservation, input_tokens, output_tokens, status, attempts=1, retry_errors=None,
               attempt_usage=None, provider=None, model_id=None, prompt_metadata=None):
        attempts = int(attempts)
        if not 0 <= attempts <= reservation.max_attempts:
            raise RuntimeError("Client exceeded its reserved attempt bound.")
        if self.reserved_usd + 1e-10 < reservation.upper_bound_usd:
            raise RuntimeError("Reservation was already recorded or is not held by this ledger.")
        self.reserved_usd = max(0.0, self.reserved_usd - reservation.upper_bound_usd)
        tokens_available = all(type(value) is int and value >= 0 for value in [input_tokens, output_tokens])
        if tokens_available:
            final_attempt_estimate = (
                input_tokens * self.input_price_per_million / 1_000_000
                + output_tokens * self.output_price_per_million / 1_000_000
            )
            # Earlier failed attempts can consume provider tokens without
            # returning usage metadata. Charge their configured upper bound.
            estimate = final_attempt_estimate + max(0, attempts - 1) * self.upper_bound_for_call()
        else:
            estimate = attempts * self.upper_bound_for_call()

        if attempt_usage and len(attempt_usage) == attempts:
            estimate = 0.0
            for usage in attempt_usage:
                counts = [usage.get("input_tokens"), usage.get("output_tokens")]
                if all(type(value) is int and value >= 0 for value in counts):
                    estimate += (counts[0] * self.input_price_per_million
                                 + counts[1] * self.output_price_per_million) / 1_000_000
                else:
                    estimate += self.upper_bound_for_call()

        self.estimated_spent_usd += estimate
        call = {
            "role": reservation.role,
            "provider": provider,
            "model_id": model_id,
            "status": status,
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "retry_errors": list(retry_errors or []),
            "attempt_usage": list(attempt_usage or []),
            "pricing_policy": self.pricing_policy(),
            "input_tokens": input_tokens if tokens_available else None,
            "output_tokens": output_tokens if tokens_available else None,
            "usage_available": tokens_available,
            "estimated_cost_usd": estimate,
            "reserved_upper_bound_usd": reservation.upper_bound_usd,
            "provider_actual_cost_usd": None,
            "prompt": dict(prompt_metadata or {}),
        }
        self.calls.append(call)
        self._persist()
        return call

    def snapshot(self):
        return {
            "schema_version": 1,
            "ceiling_usd": self.ceiling_usd,
            "estimated_spent_usd": self.estimated_spent_usd,
            "reserved_usd": self.reserved_usd,
            "provider_actual_cost_usd": None,
            "calls": list(self.calls),
            "pricing_policy": self.pricing_policy(),
            "migrations": list(self.migrations),
        }

    def summary(self):
        """Episode records already contain their own calls; avoid copying all history."""
        state = self.snapshot()
        state["call_count"] = len(state.pop("calls"))
        return state
