"""Tool Executor: one pipeline for every tool call.

    registry lookup → schema validation → policy (authorization /
    confirmation) → idempotency → execute (retry policy) → output cap →
    injection scan → audit → observation

The executor never trusts the model for identity or authority: the
principal comes from the run, authority from the policy, and the arguments
are validated against a closed schema before any handler sees them.
Custom tools, MCP-style tools and provider adapters all go through the same
pipeline — there is no other way for the loop to run a tool.

Where an application keeps audit rows, idempotency records or maps its own
exceptions, it plugs in :class:`ExecutionHooks`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..decisions import ToolCall
from ..policy import Allow, Deny, PolicyOutcome, RequireAuthorization, RequireConfirmation
from ..util import stable_hash
from .base import ToolError, ToolResult, ToolSpec

logger = logging.getLogger("memoflow_agent.executor")

MAX_OBSERVATION_BYTES = 60_000


@dataclass
class ExecOutcome:
    """What the loop does next with one call."""

    kind: str  # "observation" | "authorization" | "confirmation"
    call: ToolCall
    content: str = ""
    ok: bool = False
    result: ToolResult | None = None
    error_code: str | None = None
    authorization: RequireAuthorization | None = None
    confirmation: RequireConfirmation | None = None
    events: list[tuple[str, dict]] = field(default_factory=list)


@dataclass(frozen=True)
class MappedError:
    status: str  # "failed" | "denied"
    code: str
    message: str
    authorization: str = "allowed"


class ExecutionHooks:
    """Application seams of the pipeline. Every method has a safe default."""

    def begin(self, ctx: Any, spec: ToolSpec, call: ToolCall, arguments_hash: str) -> Any:
        return None

    def finish(
        self,
        ctx: Any,
        handle: Any,
        *,
        status: str,
        error_code: str | None = None,
        output: dict | None = None,
        authorization: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        return None

    def idempotency_key(self, ctx: Any, spec: ToolSpec, call: ToolCall) -> str:
        return f"run:{getattr(ctx, 'run_id', '')}:call:{call.id}"[:160]

    def replay(self, ctx: Any, spec: ToolSpec, call: ToolCall, key: str) -> ToolResult | MappedError | None:
        """The stored result of a side-effecting call that already completed;
        a :class:`MappedError` when it started but its outcome is unknown;
        ``None`` when it never ran."""
        return None

    def mark_idempotency(self, ctx: Any, handle: Any, key: str) -> None:
        return None

    def record_result_ref(self, ctx: Any, handle: Any, result_ref: dict) -> None:
        return None

    def on_retry(self, ctx: Any, handle: Any, attempts: int) -> None:
        return None

    def on_handler_error(self, ctx: Any) -> None:
        """Called before a failed call is finished (e.g. roll a transaction back)."""
        return None

    def map_exception(self, ctx: Any, spec: ToolSpec, error: BaseException) -> MappedError | None:
        return None


def _default_scanner(text: str) -> list[str]:
    from ..safety import scan_for_injection

    return scan_for_injection(text)


class ToolExecutor:
    def __init__(
        self,
        registry: Mapping[str, ToolSpec],
        policy: Any,
        *,
        hooks: ExecutionHooks | None = None,
        scanner: Callable[[str], list[str]] | None = None,
        observation_notice: str | None = None,
        max_observation_bytes: int = MAX_OBSERVATION_BYTES,
        truncation_marker: str = "… [truncated]",
        unknown_tool_message: str = "There is no tool named {name!r} available here.",
        internal_error_message: str = "The tool failed unexpectedly. Continue without it.",
    ):
        self.registry = registry
        self._evaluate = policy.evaluate if hasattr(policy, "evaluate") else policy
        self.hooks = hooks or ExecutionHooks()
        self.scanner = scanner or _default_scanner
        self.observation_notice = observation_notice
        self.max_observation_bytes = max_observation_bytes
        self.truncation_marker = truncation_marker
        self.unknown_tool_message = unknown_tool_message
        self.internal_error_message = internal_error_message

    # --- encoding -----------------------------------------------------------------

    def observation(self, payload: dict) -> str:
        body = {"notice": self.observation_notice, **payload} if self.observation_notice else payload
        return json.dumps(body, ensure_ascii=False, default=str)

    def error(self, call: ToolCall, code: str, message: str) -> ExecOutcome:
        return ExecOutcome(
            kind="observation",
            call=call,
            ok=False,
            error_code=code,
            content=self.observation({"ok": False, "error": {"code": code, "message": message}}),
        )

    def cap(self, payload: dict) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) <= self.max_observation_bytes:
            return payload
        # Shrink the largest string fields until the result fits; mark it.
        trimmed = json.loads(encoded)
        marker = self.truncation_marker

        def shrink(value: Any, budget: int) -> Any:
            if isinstance(value, str) and len(value) > budget:
                return value[:budget] + marker
            if isinstance(value, list):
                return [shrink(item, budget) for item in value]
            if isinstance(value, dict):
                return {key: shrink(item, budget) for key, item in value.items()}
            return value

        budget = 4000
        while budget >= 200:
            trimmed = shrink(trimmed, budget)
            if len(json.dumps(trimmed, ensure_ascii=False).encode("utf-8")) <= self.max_observation_bytes:
                break
            budget //= 2
        trimmed["truncated"] = True
        return trimmed

    # --- pipeline -------------------------------------------------------------------

    def execute(self, ctx: Any, call: ToolCall, *, offered: set[str] | frozenset[str], runner: Any = None) -> ExecOutcome:
        spec = self.registry.get(call.name)
        if spec is None or call.name not in offered:
            return self.error(call, "unknown_tool", self.unknown_tool_message.format(name=call.name))

        hooks = self.hooks
        arguments_hash = stable_hash(call.arguments)
        handle = hooks.begin(ctx, spec, call, arguments_hash)
        started = time.monotonic()

        def finish(status: str, *, error_code: str | None = None, output: dict | None = None, authorization: str | None = None) -> None:
            hooks.finish(
                ctx,
                handle,
                status=status,
                error_code=error_code,
                output=output,
                authorization=authorization,
                duration_ms=round((time.monotonic() - started) * 1000),
            )

        # 1. Schema.
        try:
            arguments = spec.input_model.model_validate(call.arguments)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}" for issue in error.errors()[:5]
            )
            finish("rejected", error_code="invalid_arguments", authorization="not_evaluated")
            return self.error(call, "invalid_arguments", problems)

        # 2. Policy (authorization, confirmation, scope).
        decision: PolicyOutcome = self._evaluate(spec, arguments, ctx)
        if isinstance(decision, Deny):
            finish("denied", error_code=decision.code, authorization="denied")
            return self.error(call, decision.code, decision.message)
        if isinstance(decision, RequireAuthorization):
            finish("waiting_authorization", authorization="authorization_required")
            return ExecOutcome(kind="authorization", call=call, authorization=decision)
        if isinstance(decision, RequireConfirmation):
            approved = ctx.state.approved_calls.get(call.id)
            if approved is None or approved != arguments_hash:
                finish("waiting_confirmation", authorization="confirmation_required")
                return ExecOutcome(kind="confirmation", call=call, confirmation=decision)
            # Consumed: one approval runs one call, with exactly these arguments.
            ctx.state.approved_calls.pop(call.id, None)
            decision = Allow(decision.scope)
        ctx.scope = decision.scope

        # 3. Idempotency for anything with a non-repeatable side effect.
        if spec.side_effect.needs_idempotency_key:
            key = hooks.idempotency_key(ctx, spec, call)
            replayed = hooks.replay(ctx, spec, call, key)
            if isinstance(replayed, MappedError):
                # Started before, outcome unknown: never re-run it blindly.
                finish(replayed.status, error_code=replayed.code, authorization=replayed.authorization)
                return self.error(call, replayed.code, replayed.message)
            if replayed is not None:
                finish("replayed", output=replayed.observation, authorization="allowed")
                return ExecOutcome(
                    kind="observation",
                    call=call,
                    ok=True,
                    result=replayed,
                    content=self.observation({"ok": True, "data": replayed.observation}),
                    events=replayed.events,
                )
            hooks.mark_idempotency(ctx, handle, key)

        # 4. Execute, with the tool's own retry policy.
        attempts = 0
        while True:
            attempts += 1
            try:
                result = spec.invoke(ctx, arguments, runner=runner)
                break
            except ToolError as error:
                if error.retryable and error.code in spec.retry.retry_on and attempts <= spec.retry.max_retries:
                    hooks.on_retry(ctx, handle, attempts)
                    continue
                hooks.on_handler_error(ctx)
                finish("failed", error_code=error.code, authorization="allowed")
                return self.error(call, error.code, error.message)
            except Exception as error:  # noqa: BLE001 — the model must hear about every failure
                hooks.on_handler_error(ctx)
                mapped = hooks.map_exception(ctx, spec, error)
                if mapped is not None:
                    finish(mapped.status, error_code=mapped.code, authorization=mapped.authorization)
                    return self.error(call, mapped.code, mapped.message)
                if spec.is_async and isinstance(error, TimeoutError):
                    finish("failed", error_code="timeout", authorization="allowed")
                    return self.error(call, "timeout", "The tool timed out. Continue without it.")
                logger.exception("tool %s failed", spec.name)
                finish("failed", error_code="internal_error", authorization="allowed")
                return self.error(call, "internal_error", self.internal_error_message)

        # 5. Output cap and injection scan.
        observation = self.cap(result.observation)
        signals: list[str] = []
        for text in result.untrusted_texts:
            signals.extend(self.scanner(text or ""))
        if signals:
            flags = ctx.state.security_flags
            ctx.state.security_flags = {
                "injection_detected": True,
                "signals": list(dict.fromkeys(flags.get("signals", []) + signals))[:20],
                "tool": spec.name,
            }
        if result.result_ref is not None:
            hooks.record_result_ref(ctx, handle, result.result_ref)

        # 6. Audit.
        finish("completed", output=observation, authorization="allowed")
        return ExecOutcome(
            kind="observation",
            call=call,
            ok=True,
            result=result,
            content=self.observation({"ok": True, "data": observation}),
            events=result.events,
        )
