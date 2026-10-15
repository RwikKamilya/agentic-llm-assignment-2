"""Conservative, provider-independent accounting for planned model calls."""

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised before a request that would exceed the configured guardrail."""


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
        self, ceiling_usd, max_input_tokens, max_output_tokens, input_price_per_million, output_price_per_million
    ):
        self.ceiling_usd = float(ceiling_usd)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.input_price_per_million = float(input_price_per_million)
        self.output_price_per_million = float(output_price_per_million)
        self.reserved_usd = 0.0
        self.estimated_spent_usd = 0.0
        self.calls = []

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
        return CallReservation(role=role, upper_bound_usd=upper_bound, max_attempts=max_attempts)

    def record(self, reservation, input_tokens, output_tokens, status, attempts=1):
        self.reserved_usd -= reservation.upper_bound_usd
        attempts = max(1, int(attempts))
        tokens_available = isinstance(input_tokens, int) and isinstance(output_tokens, int)
        if tokens_available:
            final_attempt_estimate = (
                input_tokens * self.input_price_per_million / 1_000_000
                + output_tokens * self.output_price_per_million / 1_000_000
            )
            # Earlier failed attempts can consume provider tokens without
            # returning usage metadata. Charge their configured upper bound.
            estimate = final_attempt_estimate + (attempts - 1) * self.upper_bound_for_call()
        else:
            estimate = attempts * self.upper_bound_for_call()

        self.estimated_spent_usd += estimate
        call = {
            "role": reservation.role,
            "status": status,
            "attempts": attempts,
            "retries": attempts - 1,
            "input_tokens": input_tokens if tokens_available else None,
            "output_tokens": output_tokens if tokens_available else None,
            "usage_available": tokens_available,
            "estimated_cost_usd": estimate,
            "reserved_upper_bound_usd": reservation.upper_bound_usd,
            "provider_actual_cost_usd": None,
        }
        self.calls.append(call)
        return call

    def snapshot(self):
        return {
            "ceiling_usd": self.ceiling_usd,
            "estimated_spent_usd": self.estimated_spent_usd,
            "reserved_usd": self.reserved_usd,
            "provider_actual_cost_usd": None,
            "calls": list(self.calls),
        }
