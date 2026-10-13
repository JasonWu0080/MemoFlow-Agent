"""The tool contract.

A tool is a contract plus a handler. Validation, authorization, idempotency,
auditing and output sanitization are *not* the handler's business — the
:class:`~memoflow_agent.tools.executor.ToolExecutor` runs every call through
the same pipeline, and a policy decides authorization before a handler is
ever entered.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..state import PendingAction, RunState


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SideEffect(StrEnum):
    """What running a tool does to the world.

    * ``READ_ONLY`` — nothing outside the process changes; safe to retry.
    * ``IDEMPOTENT`` — leaves the process (e.g. a search query to a third
      party) but repeating it changes nothing; safe to retry.
    * ``NON_IDEMPOTENT`` — repeating it repeats the effect. Executed at most
      once per ``(run, tool_call)``: guarded by an idempotency key, and never
      blindly retried after a crash.
    * ``HUMAN_APPROVED`` — produces something only a human can turn into a
      change (a proposal, a draft), or runs only after a human approved the
      exact arguments. Same at-most-once guard as ``NON_IDEMPOTENT``.
    """

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"
    HUMAN_APPROVED = "human_approved"

    @property
    def needs_idempotency_key(self) -> bool:
        return self in {SideEffect.NON_IDEMPOTENT, SideEffect.HUMAN_APPROVED}


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    # Error codes worth trying again; anything else fails the call at once.
    retry_on: frozenset[str] = frozenset()


class StrictInput(BaseModel):
    """Base for tool inputs. Unknown fields are rejected, never ignored.

    A forged ``user_id`` or ``scope_override`` argument is an error the model
    hears about — never a parameter that silently does nothing, or something.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    """What a handler receives besides its arguments.

    Identity comes from the run (``principal``), never from the model's
    arguments. Applications may pass their own context object instead; the
    executor only needs ``state`` and ``scope``.
    """

    run_id: Any
    state: RunState
    principal: Any = None
    session_id: Any = None
    scope: Any = None
    is_cancelled: Callable[[], bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    observation: dict
    # Safe for the client and the event log: counts, titles, sites — never
    # bodies of private documents or page text.
    summary: dict = field(default_factory=dict)
    # Third-party or user-material text the executor scans for injection.
    untrusted_texts: list[str] = field(default_factory=list)
    # Set when the user must confirm what this call prepared: the run
    # suspends and the observation is withheld until they decide.
    pending: PendingAction | None = None
    # Extra events to emit after `tool.completed`, as (type, payload).
    events: list[tuple[str, dict]] = field(default_factory=list)
    # Stored with the audit record so a replayed call returns the same result.
    result_ref: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolError(RuntimeError):
    """A call that failed in a way the model should be told about."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Any]
    risk: Risk = Risk.LOW
    side_effect: SideEffect = SideEffect.READ_ONLY
    timeout_seconds: float = 30.0
    retry: RetryPolicy = RetryPolicy()
    # The capability a principal needs to run it (see memoflow_agent.policy).
    permission: str | None = None
    # Policy hint: always ask for a human approval of the exact arguments.
    requires_approval: bool = False
    # Needs an interactive client (one that can show approval / authorization cards).
    interactive_only: bool = False
    # Capabilities the *model* must have for this tool to make sense.
    required_capabilities: frozenset[str] = frozenset()
    # Argument keys whose (short) values may be kept on the audit record.
    audit_fields: tuple[str, ...] = ()
    # Argument keys a UI may show while the call runs. Nothing else is shown.
    public_fields: tuple[str, ...] = ()
    output_schema: dict | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def declaration(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }

    def definition_hash(self) -> str:
        encoded = json.dumps(self.declaration(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.handler)

    def invoke(self, ctx: Any, arguments: BaseModel, *, runner: Any = None) -> ToolResult:
        """Run the handler; coroutine handlers run on ``runner`` (an ``asyncio.Runner``)."""
        if self.is_async:
            coroutine = asyncio.wait_for(self.handler(ctx, arguments), timeout=self.timeout_seconds)
            result = runner.run(coroutine) if runner is not None else asyncio.run(coroutine)
        else:
            result = self.handler(ctx, arguments)
        return coerce_result(result)

    def public_arguments(self, arguments: Mapping[str, Any]) -> dict:
        return {key: arguments.get(key) for key in self.public_fields if key in arguments}


def coerce_result(value: Any) -> ToolResult:
    """Handlers may return a ToolResult, a dict, a pydantic model or a scalar."""
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, BaseModel):
        return ToolResult(observation=value.model_dump(mode="json"))
    if isinstance(value, dict):
        return ToolResult(observation=value)
    return ToolResult(observation={"result": value})


class ToolRegistry(Mapping[str, ToolSpec]):
    """Name → spec, refusing duplicates and reserved names."""

    def __init__(self, specs: Iterable[ToolSpec] = (), *, reserved: Iterable[str] = ()):
        from ..decisions import CLARIFICATION_CHANNEL

        self._reserved = frozenset(reserved) | {CLARIFICATION_CHANNEL}
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._reserved:
            raise ValueError(f"tool name {spec.name!r} is reserved")
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def __getitem__(self, name: str) -> ToolSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def declarations(self, names: Iterable[str] | None = None) -> list[dict]:
        wanted = set(names) if names is not None else None
        return [spec.declaration() for name, spec in self._specs.items() if wanted is None or name in wanted]
