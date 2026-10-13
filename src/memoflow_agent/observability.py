"""Usage accounting hooks.

The kernel counts model calls, tokens and tool calls and enforces the
budget. What usage *costs* — credits, wallets, invoices — belongs to the
application. A :class:`UsageObserver` receives one :class:`UsageEvent` per
model call and per tool call and may refuse further spending by raising
:class:`~memoflow_agent.exceptions.CapacityExhausted` from ``before_model_call``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class UsageEvent:
    kind: str  # "model_call" | "tool_call"
    run_id: Any
    provider: str | None = None
    model: str | None = None
    tool: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_microusd: int | None = None
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class UsageObserver(Protocol):
    def before_model_call(self, run_id: Any, *, estimated_input_tokens: int) -> None: ...

    def on_usage(self, event: UsageEvent) -> None: ...


class UsageTotals:
    """A trivial observer that sums usage (handy in tests and scripts)."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def before_model_call(self, run_id: Any, *, estimated_input_tokens: int) -> None:
        return None

    def on_usage(self, event: UsageEvent) -> None:
        self.events.append(event)

    @property
    def model_calls(self) -> int:
        return sum(1 for event in self.events if event.kind == "model_call")

    @property
    def input_tokens(self) -> int:
        return sum(event.input_tokens for event in self.events)

    @property
    def output_tokens(self) -> int:
        return sum(event.output_tokens for event in self.events)
