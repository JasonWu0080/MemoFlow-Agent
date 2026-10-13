"""Loop guards for one run.

The model decides what to do next; the budget decides whether it still may.
Every guard is checked by the runtime *before* the next model call or tool
call, so tripping one never interrupts a tool halfway.

Counters live on :class:`~memoflow_agent.state.RunState`, so a run recovered
on another worker carries on with what it had already spent rather than
getting a fresh allowance. Wall time is per *segment* (the stretch between
two user actions): time spent waiting for the user is not the run's.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .exceptions import BudgetExceeded
from .state import RunState

__all__ = ["AgentBudget", "BudgetExceeded", "BudgetLimits", "RunBudget"]


@dataclass(frozen=True)
class BudgetLimits:
    max_model_turns: int = 8
    max_tool_calls: int = 12
    max_run_tokens: int = 240_000
    max_run_seconds: int = 300
    max_consecutive_failures: int = 3
    # A tool called again with identical arguments gets the earlier result
    # pointed out instead of re-running; this many identical calls ends the run.
    max_identical_calls: int = 3
    # Optional separate ceilings on accumulated input / output tokens.
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        # Keep the historical shape when the optional limits are unused.
        for key in ("max_input_tokens", "max_output_tokens"):
            if data[key] is None:
                data.pop(key)
        return data


def AgentBudget(
    *,
    max_model_turns: int = 8,
    max_tool_calls: int = 12,
    max_tokens: int = 240_000,
    max_seconds: int = 300,
    max_consecutive_failures: int = 3,
    max_identical_calls: int = 3,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> BudgetLimits:
    """Friendly constructor for :class:`BudgetLimits` used by :class:`~memoflow_agent.Agent`."""
    return BudgetLimits(
        max_model_turns=max_model_turns,
        max_tool_calls=max_tool_calls,
        max_run_tokens=max_tokens,
        max_run_seconds=max_seconds,
        max_consecutive_failures=max_consecutive_failures,
        max_identical_calls=max_identical_calls,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    )


class RunBudget:
    """Guards evaluated against the run's persisted counters."""

    def __init__(self, limits: BudgetLimits, state: RunState, *, started_at: float | None = None):
        self.limits = limits
        self.state = state
        self.started_at = started_at if started_at is not None else time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def check_model_turn(self) -> None:
        if self.state.model_turns >= self.limits.max_model_turns:
            raise BudgetExceeded("max_model_turns")
        self._check_common()

    def check_tool_call(self) -> None:
        if self.state.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("max_tool_calls")
        self._check_common()

    def _check_common(self) -> None:
        if self.state.tokens_used >= self.limits.max_run_tokens:
            raise BudgetExceeded("max_run_tokens")
        if self.limits.max_input_tokens is not None and self.state.input_tokens_used >= self.limits.max_input_tokens:
            raise BudgetExceeded("max_input_tokens")
        if self.limits.max_output_tokens is not None and self.state.output_tokens_used >= self.limits.max_output_tokens:
            raise BudgetExceeded("max_output_tokens")
        if self.elapsed_seconds() >= self.limits.max_run_seconds:
            raise BudgetExceeded("max_run_seconds")
        if self.state.consecutive_failures >= self.limits.max_consecutive_failures:
            raise BudgetExceeded("max_consecutive_failures")

    def record_identical(self, signature: str) -> int:
        """Count a call signature; returns how many times it has now been seen."""
        count = self.state.call_signatures.get(signature, 0) + 1
        self.state.call_signatures[signature] = count
        if count >= self.limits.max_identical_calls:
            raise BudgetExceeded("repeated_tool_call")
        return count

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        self.state.model_turns += 1
        self.state.tokens_used += int(input_tokens or 0) + int(output_tokens or 0)
        self.state.input_tokens_used += int(input_tokens or 0)
        self.state.output_tokens_used += int(output_tokens or 0)

    def snapshot(self) -> dict:
        return {
            "model_turns": self.state.model_turns,
            "tool_calls": self.state.tool_calls,
            "tokens_used": self.state.tokens_used,
            "elapsed_ms": round(self.elapsed_seconds() * 1000),
            "consecutive_failures": self.state.consecutive_failures,
            "limits": self.limits.as_dict(),
        }
