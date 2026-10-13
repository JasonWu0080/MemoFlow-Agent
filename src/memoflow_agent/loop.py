"""The Agent Loop — the only thing that advances a run.

    Prepare → Reason → Decision
        FINAL         → complete
        CLARIFICATION → suspend for the user
        TOOL_CALL     → Policy → Execute → Observation → Reason …

The model decides what to do next. Everything about authority — which tools
are offered, whether a call may run, how much may be spent, what state the
run is in — is decided here and in the policy, deterministically.

``execute()`` is a generator of event envelopes. Every state change is
written to the store before the event describing it is yielded, so a client
that reconnects — or a worker that recovers the run after a crash — sees
exactly what happened. The loop holds the run's lease for its whole
execution and renews it on every step and periodically while text streams;
losing the lease stops it without another write.

What the loop does *not* know is the embedding application: how messages are
stored, what an event payload carries, how money is held and settled, which
context goes into a call. Those are the :class:`Application` hooks. The same
loop runs the public :class:`~memoflow_agent.Agent` (with the default
hooks) and MemoFlow's production chat (with MemoFlow's hooks).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .budget import BudgetLimits, RunBudget
from .context import BuiltContext, ContextBuilder
from .decisions import (
    CLARIFICATION_CHANNEL,
    DEFAULT_CLARIFICATION_DESCRIPTION,
    ClarificationDecision,
    FinalDecision,
    InvalidModelOutput,
    ToolCall,
    ToolCallDecision,
    clarification_declaration,
    decide,
)
from .events import EventType
from .exceptions import BudgetExceeded, CapacityExhausted, ContextTooLong, LeaseLost, ModelProviderError
from .models.base import ModelProvider, ModelRequest, ModelTurn, ProviderEvent, TextDelta, TurnCancelled
from .policy import RequireAuthorization, new_approval
from .state import EXECUTABLE_STATUSES, PendingAction, RunState, RunStatus, StepType, StopReason, WaitingReason
from .tools.base import SideEffect, ToolContext, ToolSpec
from .tools.executor import ExecOutcome, ToolExecutor
from .util import stable_hash

logger = logging.getLogger("memoflow_agent.loop")

# Budget reasons after which one more model call (without tools) is still
# worth making, to turn what was gathered into a labelled partial answer.
WRAP_UP_REASONS = frozenset({"max_model_turns", "max_tool_calls", "repeated_tool_call", "max_consecutive_failures"})


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    stop_reason: str = StopReason.PROVIDER_ERROR.value


class _Stop(Exception):
    """Unwinds the loop once the run has reached a resting state."""


class _Cancelled(Exception):
    def __init__(self, partial_text: str = ""):
        super().__init__("cancelled")
        self.partial_text = partial_text


class CancellationRegistry:
    """Instant stop for runs executing in this process; the store flag covers the rest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[Any, threading.Event] = {}

    def event_for(self, run_id: Any) -> threading.Event:
        with self._lock:
            return self._events.setdefault(run_id, threading.Event())

    def cancel(self, run_id: Any) -> None:
        with self._lock:
            event = self._events.get(run_id)
        if event is not None:
            event.set()

    def discard(self, run_id: Any) -> None:
        with self._lock:
            self._events.pop(run_id, None)


class _LazyRunner:
    """An ``asyncio.Runner`` created on first use: async tools of one run share a loop."""

    def __init__(self) -> None:
        self._runner: asyncio.Runner | None = None

    def run(self, coroutine: Any) -> Any:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(coroutine)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


def _step_id(step: Any) -> Any:
    if step is None:
        return None
    if isinstance(step, dict):
        return step.get("id", step.get("sequence"))
    return getattr(step, "id", None)


DEFAULT_INSTRUCTIONS = "You are a helpful assistant. Use tools when they help; answer directly when they do not."
DEFAULT_WRAP_UP = (
    "You have reached this request's step budget. Do not call tools. Answer now with what you already "
    "have, and say briefly what is missing."
)
DEFAULT_INVALID_OUTPUT_NUDGE = (
    "The previous turn produced neither a usable answer nor a valid tool call. Answer the user directly, "
    "or make one tool call with correct arguments."
)


class Application:
    """The embedding application's side of a run. Defaults suit a standalone agent.

    Hooks that may emit events are generators: they ``yield from loop.emit(...)``
    and ``return`` their value.
    """

    instructions: str = DEFAULT_INSTRUCTIONS
    wrap_up_instruction: str = DEFAULT_WRAP_UP
    invalid_output_nudge: str = DEFAULT_INVALID_OUTPUT_NUDGE
    clarification_description: str = DEFAULT_CLARIFICATION_DESCRIPTION
    budget_exhausted_text: str = "I had to stop before finishing because this request reached its budget."

    def __init__(
        self,
        *,
        instructions: str | None = None,
        context_builder: ContextBuilder | None = None,
        history: list[dict] | None = None,
        max_output_tokens: int = 4096,
        principal: Any = None,
        session_id: Any = None,
        clarification: bool = True,
        fallback_enabled: bool = False,
    ):
        if instructions is not None:
            self.instructions = instructions
        self.context_builder = context_builder or ContextBuilder()
        self.history = list(history or [])
        self.max_output_tokens = max_output_tokens
        self.principal = principal
        self.session_id = session_id
        self.clarification = clarification
        self.fallback_enabled = fallback_enabled
        self.text = ""

    # --- identity and tools ------------------------------------------------------

    def principal_id(self, loop: AgentLoop) -> str | None:
        return getattr(self.principal, "id", None)

    def tool_context(self, loop: AgentLoop) -> Any:
        return ToolContext(
            run_id=loop.run.id,
            state=loop.state,
            principal=self.principal,
            session_id=self.session_id,
            is_cancelled=loop.is_cancelled,
        )

    def offered_tools(self, loop: AgentLoop) -> set[str]:
        return {
            name
            for name, spec in loop.executor.registry.items()
            if loop.state.interactive or not spec.interactive_only
        }

    def declarations(self, loop: AgentLoop) -> list[dict]:
        offered = self.offered_tools(loop)
        declarations = [spec.declaration() for name, spec in loop.executor.registry.items() if name in offered]
        if declarations and self.clarification and loop.state.interactive:
            declarations.append(clarification_declaration(self.clarification_description))
        return declarations

    def public_arguments(self, loop: AgentLoop, call: ToolCall) -> dict:
        spec = loop.executor.registry.get(call.name)
        return spec.public_arguments(call.arguments) if spec is not None else {}

    # --- segment lifecycle ----------------------------------------------------------

    def prepare_segment(self, loop: AgentLoop, *, recovering: bool) -> None:
        if recovering:
            self.text = ""

    def started_payload(self, loop: AgentLoop) -> dict:
        return {"model": getattr(loop.provider, "model", None), "tools_enabled": bool(self.offered_tools(loop))}

    def resumed_payload(self, loop: AgentLoop, *, recovering: bool) -> dict:
        return {"segment": getattr(loop.run, "segment", 0), "recovered": recovering}

    def message_ref(self, loop: AgentLoop) -> dict:
        return {}

    def reset_payload(self, loop: AgentLoop, reason: str) -> dict | None:
        return {**self.message_ref(loop), "reason": reason}

    def begin_segment(self, loop: AgentLoop) -> None:
        return None

    # --- model turns ------------------------------------------------------------------

    def prepare_turn(self, loop: AgentLoop, *, wrap_up: bool) -> BuiltContext:
        """Build the next call's context (raise CapacityExhausted to stop spending)."""
        state = loop.state
        return self.context_builder.build(
            system=self.instructions,
            request=state.user_texts[0] if state.user_texts else "",
            transcript=state.transcript,
            history=self.history,
            declarations=[] if wrap_up else self.declarations(loop),
            extra_system=[self.wrap_up_instruction] if wrap_up else None,
        )

    def model_request(self, loop: AgentLoop, built: BuiltContext, *, wrap_up: bool) -> ModelRequest:
        declarations = [] if wrap_up else self.declarations(loop)
        return ModelRequest(
            messages=built.messages,
            tools=declarations or None,
            max_output_tokens=self.max_output_tokens,
            allow_fallback=self.fallback_enabled and not loop.state.side_effects_started,
        )

    def model_step_name(self, loop: AgentLoop) -> str | None:
        return getattr(loop.provider, "model", None)

    def flush_text(self, loop: AgentLoop, text: str) -> bool:
        """Persist the answer text so far (about once a second). True if anything was persisted."""
        self.text = text
        return True

    def on_provider_event(self, loop: AgentLoop, event: ProviderEvent) -> Iterator[dict]:
        return
        yield  # pragma: no cover

    def after_model_turn(self, loop: AgentLoop, turn: ModelTurn, step: Any, request: ModelRequest) -> Generator[dict, None, Any]:
        """May emit events; the return value is handed to the step-usage and content hooks."""
        return None
        yield  # pragma: no cover

    def model_step_usage(self, loop: AgentLoop, turn: ModelTurn, built: BuiltContext, outcome: Any) -> dict:
        manifest = built.manifest.as_dict() if hasattr(built.manifest, "as_dict") else None
        return {
            **turn.usage_dict(),
            "model": turn.model,
            "attempts": turn.attempts,
            "ttft_ms": turn.ttft_ms,
            "finish_reason": turn.finish_reason,
            "response_id": turn.response_id,
            "context": manifest,
        }

    def manifest(self, loop: AgentLoop, built: BuiltContext) -> dict | None:
        return built.manifest.as_dict() if hasattr(built.manifest, "as_dict") else None

    def final_content(self, loop: AgentLoop, content: str, outcome: Any) -> str:
        return content

    def tool_turn_content(self, loop: AgentLoop, text: str, outcome: Any) -> str:
        return text

    def discard_streamed_text(self, loop: AgentLoop) -> None:
        self.text = ""

    # --- tools --------------------------------------------------------------------------

    def on_tool_outcome(self, loop: AgentLoop, call: ToolCall, spec: ToolSpec | None, outcome: ExecOutcome) -> None:
        return None

    def on_pending(self, loop: AgentLoop, pending: PendingAction) -> None:
        return None

    def authorization_request(self, loop: AgentLoop, requirement: RequireAuthorization) -> tuple[dict, str]:
        pending = loop.state.pending
        payload = {
            "resource": requirement.resource,
            "reason": requirement.reason,
            "tool_call_id": pending.tool_call_id if pending else None,
            "tool": pending.tool_name if pending else None,
            "durations": ["once", "session", "persistent"],
        }
        return payload, "Waiting for the user to authorize access."

    def approval_payload(self, loop: AgentLoop, pending: PendingAction) -> dict:
        approval = dict(pending.approval or {})
        spec = loop.executor.registry.get(pending.tool_name or "")
        return {
            "approval_id": approval.get("approval_id"),
            "tool_call_id": pending.tool_call_id,
            "tool": pending.tool_name,
            "arguments": spec.public_arguments(pending.arguments) if spec else {},
            "arguments_hash": approval.get("arguments_hash"),
            "expires_at": approval.get("expires_at"),
        }

    def confirmation_text(self, loop: AgentLoop, pending: PendingAction) -> str:
        return "Waiting for the user to confirm."

    # --- resting states -------------------------------------------------------------------

    def complete(self, loop: AgentLoop, text: str, *, partial: bool, stop_reason: str) -> dict:
        self.text = text
        state = loop.state
        return {
            "content": text,
            "partial": partial,
            "stop_reason": stop_reason,
            "budget_reason": state.partial_reason,
            "usage": {
                "model_calls": state.model_turns,
                "input_tokens": state.input_tokens_used,
                "output_tokens": state.output_tokens_used,
                "tool_calls": state.tool_calls,
            },
        }

    def after_complete(self, loop: AgentLoop) -> None:
        return None

    def pause(self, loop: AgentLoop, reason: WaitingReason, text: str, *, finalize_message: bool) -> None:
        return None

    def cancel(self, loop: AgentLoop, partial_text: str) -> None:
        self.text = partial_text or self.text

    def fail(self, loop: AgentLoop, failure: Failure) -> None:
        return None

    # --- failures --------------------------------------------------------------------------

    def classify_error(self, loop: AgentLoop, error: BaseException) -> Failure | None:
        if isinstance(error, ModelProviderError):
            return Failure(f"provider.{error.code}", str(error))
        if isinstance(error, ContextTooLong):
            return Failure("context_too_long", str(error))
        return None

    def internal_failure(self, loop: AgentLoop) -> Failure:
        return Failure("internal_error", "The run failed unexpectedly.", StopReason.INTERNAL_ERROR.value)

    def invalid_output_failure(self, loop: AgentLoop) -> Failure:
        return Failure("invalid_model_output", "The model did not produce a usable answer.", StopReason.INVALID_MODEL_OUTPUT.value)


class AgentLoop:
    def __init__(
        self,
        *,
        run: Any,
        state: RunState,
        store: Any,
        events: Any,
        app: Application,
        executor: ToolExecutor,
        provider: ModelProvider,
        limits: BudgetLimits,
        worker_id: str,
        cancellations: CancellationRegistry | None = None,
        flush_interval: float = 1.0,
        lease_renew_interval: float = 5.0,
        cancel_poll_interval: float = 1.0,
    ):
        self.run = run
        self.state = state
        self.store = store
        self.events = events
        self.app = app
        self.executor = executor
        self.provider = provider
        self.worker_id = worker_id
        self.budget = RunBudget(limits, state)
        self.cancellations = cancellations
        self.cancel_event = cancellations.event_for(run.id) if cancellations is not None else threading.Event()
        self.flush_interval = flush_interval
        self.lease_renew_interval = lease_renew_interval
        self.cancel_poll_interval = cancel_poll_interval
        self._last_cancel_poll = 0.0
        self._last_lease_renew = time.monotonic()
        self._runner = _LazyRunner()

    # --- plumbing -----------------------------------------------------------------------

    def emit(self, event_type: str, payload: dict, *, commit: bool = True) -> Iterator[dict]:
        event = self.events.emit(str(event_type), payload, commit=commit)
        if event is not None:
            yield event

    def emit_all(self, events: Iterable[tuple[str, dict]]) -> Iterator[dict]:
        for event_type, payload in events:
            yield from self.emit(event_type, payload)

    def checkpoint(self, *, step: Any = None, manifest: dict | None = None) -> None:
        self.store.save_checkpoint(self.run, self.state, step_id=_step_id(step), manifest=manifest)

    def renew_lease(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_lease_renew >= self.lease_renew_interval:
            self.store.renew_lease(self.run, self.worker_id)
            self._last_lease_renew = now

    def is_cancelled(self) -> bool:
        if self.cancel_event.is_set():
            return True
        now = time.monotonic()
        if now - self._last_cancel_poll >= self.cancel_poll_interval:
            self._last_cancel_poll = now
            if self.store.cancel_requested(self.run.id):
                self.cancel_event.set()
                return True
        return False

    def _tool_result(self, call_id: str, payload: dict) -> None:
        self.state.transcript.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload, ensure_ascii=False)})

    # --- entry --------------------------------------------------------------------------

    def execute(self) -> Iterator[dict]:
        try:
            yield from self._start()
            while True:
                if self.is_cancelled():
                    raise _Cancelled()
                if self.state.queued_calls:
                    calls = [ToolCall.model_validate(call) for call in self.state.queued_calls]
                    self.state.queued_calls = []
                    yield from self._execute_calls(calls)
                    continue
                yield from self._model_turn()
        except _Stop:
            return
        except _Cancelled as cancelled:
            yield from self._finish_cancelled(cancelled.partial_text)
        except LeaseLost:
            logger.warning("run %s: lease lost; another worker continues it", self.run.id)
            return
        except Exception as error:  # noqa: BLE001 — every failure ends the run cleanly
            failure = self.app.classify_error(self, error)
            if failure is None:
                logger.exception("run %s crashed", self.run.id)
                self.store.rollback()
                failure = self.app.internal_failure(self)
            yield from self._fail(failure)
        finally:
            self._runner.close()
            if self.cancellations is not None and self.store.get_status(self.run) not in EXECUTABLE_STATUSES:
                self.cancellations.discard(self.run.id)

    def _start(self) -> Iterator[dict]:
        # `running` here means a previous executor died mid-segment: this is a
        # recovery, and whatever it streamed is not part of the answer.
        recovering = self.store.get_status(self.run) == RunStatus.RUNNING
        self.store.set_status(self.run, RunStatus.RUNNING)
        self.app.prepare_segment(self, recovering=recovering)
        if getattr(self.run, "segment", 0) == 0 and getattr(self.events, "seq", 0) == 0 and not recovering:
            yield from self.emit(EventType.RUN_STARTED, self.app.started_payload(self))
        else:
            yield from self.emit(EventType.RUN_RESUMED, self.app.resumed_payload(self, recovering=recovering))
            if recovering:
                reset = self.app.reset_payload(self, "recovered")
                if reset is not None:
                    yield from self.emit(EventType.MODEL_RESET, reset)
        self.app.begin_segment(self)

    # --- one model turn -------------------------------------------------------------------

    def _model_turn(self, *, wrap_up: bool = False) -> Iterator[dict]:
        if not wrap_up:
            try:
                self.budget.check_model_turn()
            except BudgetExceeded as exceeded:
                yield from self._budget_exceeded(exceeded.reason)
                raise _Stop() from None
        try:
            built = self.app.prepare_turn(self, wrap_up=wrap_up)
        except CapacityExhausted as exhausted:
            self.state.partial_reason = exhausted.reason
            yield from self._complete_deterministic(StopReason.BUDGET_EXCEEDED.value, exhausted.reason)
            raise _Stop() from None
        self.renew_lease(force=True)
        step = self.store.begin_step(
            self.run, StepType.MODEL.value, name=self.app.model_step_name(self), budget_snapshot=self.budget.snapshot()
        )
        request = self.app.model_request(self, built, wrap_up=wrap_up)
        message_ref = self.app.message_ref(self)
        text = ""
        turn: ModelTurn | None = None
        last_flush = time.monotonic()
        try:
            for frame in self.provider.stream_turn(request, is_cancelled=self.is_cancelled):
                if isinstance(frame, TextDelta):
                    offset = len(text)
                    text += frame.text
                    yield from self.emit(EventType.MODEL_DELTA, {**message_ref, "text": frame.text, "offset": offset})
                    now = time.monotonic()
                    if now - last_flush >= self.flush_interval and self.app.flush_text(self, text):
                        last_flush = now
                        self.renew_lease()
                elif isinstance(frame, ProviderEvent):
                    yield from self.app.on_provider_event(self, frame)
                elif isinstance(frame, ModelTurn):
                    turn = frame
        except TurnCancelled as cancelled:
            partial = cancelled.partial
            self._account(partial)
            self.store.finish_step(step, status="cancelled", duration_ms=partial.duration_ms, token_usage=partial.usage_dict())
            raise _Cancelled(partial.text) from None
        except ModelProviderError as error:
            self.state.consecutive_failures += 1
            self.store.finish_step(step, status="failed", error_code=error.code)
            self.checkpoint(step=step)
            raise
        assert turn is not None
        self._account(turn)
        outcome = yield from self.app.after_model_turn(self, turn, step, request)
        self.store.finish_step(
            step,
            status="completed",
            duration_ms=turn.duration_ms,
            token_usage=self.app.model_step_usage(self, turn, built, outcome),
            commit=False,
        )
        self.checkpoint(step=step, manifest=self.app.manifest(self, built))

        if turn.assembly_error is not None:
            yield from self._invalid_output(turn, turn.assembly_error)
            return
        try:
            decision = decide(turn.text, turn.tool_calls)
        except InvalidModelOutput as error:
            yield from self._invalid_output(turn, error)
            return

        if isinstance(decision, FinalDecision):
            self.state.consecutive_failures = 0
            content = self.app.final_content(self, decision.content, outcome)
            yield from self._complete(content, partial=wrap_up)
            raise _Stop()
        if wrap_up:
            # A wrap-up turn that still tries to act gets no second chance.
            yield from self._complete_deterministic(StopReason.BUDGET_EXCEEDED.value, self.state.partial_reason or "budget")
            raise _Stop()
        if text:
            reset = self.app.reset_payload(self, "tool_call")
            if reset is not None:
                yield from self.emit(EventType.MODEL_RESET, reset, commit=False)
                self.app.discard_streamed_text(self)
        entry: dict = {
            "role": "assistant",
            "content": self.app.tool_turn_content(self, turn.text, outcome) or None,
            "tool_calls": turn.raw_tool_calls,
        }
        if turn.reasoning_text:
            entry["reasoning_content"] = turn.reasoning_text
        self.state.transcript.append(entry)
        self.state.consecutive_failures = 0
        if isinstance(decision, ClarificationDecision):
            yield from self._clarify(decision, turn.tool_calls)
            raise _Stop()
        assert isinstance(decision, ToolCallDecision)
        yield from self._execute_calls(decision.tool_calls)

    def _account(self, turn: ModelTurn) -> None:
        self.budget.record_usage(input_tokens=turn.prompt_tokens, output_tokens=turn.completion_tokens)

    def _invalid_output(self, turn: ModelTurn, error: Exception) -> Iterator[dict]:
        code = getattr(error, "code", "invalid_model_output")
        logger.info("run %s: invalid model output (%s)", self.run.id, code)
        self.state.consecutive_failures += 1
        if turn.text:
            reset = self.app.reset_payload(self, "invalid_output")
            if reset is not None:
                yield from self.emit(EventType.MODEL_RESET, reset, commit=False)
                self.app.discard_streamed_text(self)
        if turn.raw_tool_calls:
            self.state.transcript.append({"role": "assistant", "content": turn.text or None, "tool_calls": turn.raw_tool_calls})
            for call in turn.raw_tool_calls:
                self.state.transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {"ok": False, "error": {"code": code, "message": "Arguments were not valid JSON objects. Retry with valid arguments."}},
                            ensure_ascii=False,
                        ),
                    }
                )
        else:
            self.state.transcript.append({"role": "system", "content": self.app.invalid_output_nudge})
        if self.state.consecutive_failures >= self.budget.limits.max_consecutive_failures:
            yield from self._fail(self.app.invalid_output_failure(self))
            raise _Stop()
        self.checkpoint()

    # --- tools ----------------------------------------------------------------------------

    def _execute_calls(self, calls: list[ToolCall]) -> Iterator[dict]:
        for index, call in enumerate(calls):
            # Durable intent: the decided calls not yet answered. A run
            # recovered after a crash executes exactly these (same call ids,
            # so side-effecting tools hit their idempotency records) instead
            # of asking the model again.
            self.state.queued_calls = [pending.model_dump() for pending in calls[index:]]
            self.checkpoint()
            if self.is_cancelled():
                # Every call needs an answer in the transcript, even unrun ones.
                for skipped in calls[index:]:
                    self._tool_result(skipped.id, {"ok": False, "error": {"code": "cancelled"}})
                self.state.queued_calls = []
                raise _Cancelled()
            signature = f"{call.name}:{stable_hash(call.arguments)}"
            try:
                self.budget.check_tool_call()
                self.budget.record_identical(signature)
            except BudgetExceeded as exceeded:
                for skipped in calls[index:]:
                    self._tool_result(skipped.id, {"ok": False, "error": {"code": "budget_exceeded"}})
                self.state.queued_calls = []
                yield from self._budget_exceeded(exceeded.reason)
                raise _Stop() from None
            if self.state.call_signatures.get(signature, 0) > 1:
                self._tool_result(
                    call.id,
                    {"ok": False, "error": {"code": "duplicate_call", "message": "You already made this exact call; use its earlier result."}},
                )
                continue
            self.renew_lease(force=True)
            self.state.tool_calls += 1
            yield from self.emit(
                EventType.TOOL_STARTED,
                {"tool_call_id": call.id, "tool": call.name, "arguments": self.app.public_arguments(self, call)},
            )
            outcome = self.executor.execute(
                self.app.tool_context(self), call, offered=self.app.offered_tools(self), runner=self._runner
            )
            remaining = [later.model_dump() for later in calls[index + 1 :]]
            # From here the call has an outcome; what is still queued is the rest
            # (or, if the run pauses, `pending.remaining_calls`).
            self.state.queued_calls = []
            if outcome.kind in ("authorization", "confirmation"):
                # Not executed: it runs again once the user answers, and must
                # not count as a repeat of itself or spend the budget twice.
                self.state.call_signatures[signature] = max(0, self.state.call_signatures.get(signature, 1) - 1)
                self.state.tool_calls = max(0, self.state.tool_calls - 1)
            if outcome.kind == "authorization":
                assert outcome.authorization is not None
                self.state.pending = self.state.make_pending(
                    reason=WaitingReason.AUTHORIZATION,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    remaining_calls=remaining,
                    resource=outcome.authorization.resource,
                )
                yield from self.emit(
                    EventType.TOOL_COMPLETED, {"tool_call_id": call.id, "tool": call.name, "status": "waiting_authorization"}
                )
                yield from self._wait_authorization(outcome.authorization)
                raise _Stop()
            if outcome.kind == "confirmation":
                self.state.pending = self.state.make_pending(
                    reason=WaitingReason.CONFIRMATION,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    remaining_calls=remaining,
                    approval=new_approval(
                        run_id=self.run.id,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        principal_id=self.app.principal_id(self),
                    ),
                )
                yield from self.emit(
                    EventType.TOOL_COMPLETED, {"tool_call_id": call.id, "tool": call.name, "status": "waiting_approval"}
                )
                yield from self._wait_approval(self.state.pending)
                raise _Stop()
            spec = self.executor.registry.get(call.name)
            result = outcome.result
            self.app.on_tool_outcome(self, call, spec, outcome)
            if outcome.ok and spec is not None and spec.side_effect is not SideEffect.READ_ONLY:
                self.state.side_effects_started = True
            if not outcome.ok:
                self.state.consecutive_failures += 1
            yield from self.emit(
                EventType.TOOL_COMPLETED if outcome.ok else EventType.TOOL_FAILED,
                {
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "status": "completed" if outcome.ok else "failed",
                    "error_code": outcome.error_code,
                    "summary": result.summary if result is not None else {},
                },
            )
            yield from self.emit_all(outcome.events)
            if result is not None and result.pending is not None:
                # The observation is withheld until the user decides; the
                # model will hear the outcome, not "awaiting confirmation".
                pending = result.pending
                pending.tool_call_id = call.id
                pending.tool_name = call.name
                pending.arguments = call.arguments
                pending.remaining_calls = remaining
                self.state.pending = pending
                self.app.on_pending(self, pending)
                yield from self._wait_confirmation(pending)
                raise _Stop()
            self.state.transcript.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})
            self.state.queued_calls = remaining
            self.checkpoint()
        self.state.queued_calls = []

    # --- resting states ---------------------------------------------------------------------

    def _complete(self, text: str, *, partial: bool, stop_reason: str = StopReason.FINAL.value) -> Iterator[dict]:
        partial = partial or bool(self.state.partial_reason)
        payload = self.app.complete(self, text, partial=partial, stop_reason=stop_reason)
        self.store.set_status(self.run, RunStatus.COMPLETED, stop_reason=stop_reason, commit=False)
        self.store.clear_working_state(self.run, self.state)
        self.store.commit()
        self.app.after_complete(self)
        yield from self.emit(EventType.RUN_COMPLETED, payload)

    def _complete_deterministic(self, stop_reason: str, reason: str) -> Iterator[dict]:
        self.state.partial_reason = reason
        yield from self._complete(self.app.budget_exhausted_text, partial=True, stop_reason=stop_reason)

    def _budget_exceeded(self, reason: str) -> Iterator[dict]:
        self.state.partial_reason = reason
        logger.info("run %s: budget exceeded (%s)", self.run.id, reason)
        if reason in WRAP_UP_REASONS:
            try:
                yield from self._model_turn(wrap_up=True)
            except _Stop:
                return
            return
        yield from self._complete_deterministic(StopReason.BUDGET_EXCEEDED.value, reason)

    def _pause(self, reason: WaitingReason, text: str, *, finalize_message: bool) -> Iterator[dict]:
        self.app.pause(self, reason, text, finalize_message=finalize_message)
        self.store.set_status(self.run, RunStatus.SUSPENDED, waiting_reason=reason.value, commit=False)
        self.checkpoint()
        self.store.release_lease(self.run.id, self.worker_id)
        yield from self.emit(EventType.RUN_SUSPENDED, {"reason": reason.value, **self.app.message_ref(self)})

    def _clarify(self, decision: ClarificationDecision, calls: list[ToolCall]) -> Iterator[dict]:
        clarification_call = next(call for call in calls if call.name == CLARIFICATION_CHANNEL)
        for call in calls:
            if call.id != clarification_call.id:
                self._tool_result(call.id, {"ok": False, "error": {"code": "skipped", "message": "Not run: a clarification was asked first."}})
        payload = {
            "tool_call_id": clarification_call.id,
            "question": decision.question,
            "reason": decision.reason,
            "options": [option.model_dump() for option in decision.options or []],
            "allow_multiple": decision.allow_multiple,
            "allow_free_text": decision.allow_free_text,
        }
        self.state.pending = self.state.make_pending(
            reason=WaitingReason.CLARIFICATION,
            tool_call_id=clarification_call.id,
            tool_name=CLARIFICATION_CHANNEL,
            question=decision.question,
            clarification=payload,
        )
        yield from self.emit(EventType.CLARIFICATION_REQUIRED, payload)
        yield from self._pause(WaitingReason.CLARIFICATION, decision.question, finalize_message=True)

    def _wait_authorization(self, requirement: RequireAuthorization) -> Iterator[dict]:
        payload, text = self.app.authorization_request(self, requirement)
        yield from self.emit(EventType.AUTHORIZATION_REQUIRED, payload)
        yield from self._pause(WaitingReason.AUTHORIZATION, text, finalize_message=False)

    def _wait_confirmation(self, pending: PendingAction) -> Iterator[dict]:
        yield from self._pause(WaitingReason.CONFIRMATION, self.app.confirmation_text(self, pending), finalize_message=False)

    def _wait_approval(self, pending: PendingAction) -> Iterator[dict]:
        yield from self.emit(EventType.APPROVAL_REQUIRED, self.app.approval_payload(self, pending))
        yield from self._pause(WaitingReason.CONFIRMATION, self.app.confirmation_text(self, pending), finalize_message=False)

    def _finish_cancelled(self, partial_text: str) -> Iterator[dict]:
        self.app.cancel(self, partial_text)
        self.store.set_status(self.run, RunStatus.CANCELLED, stop_reason=StopReason.CANCELLED.value, commit=False)
        self.store.clear_working_state(self.run, self.state)
        self.store.commit()
        yield from self.emit(EventType.RUN_CANCELLED, self.app.message_ref(self))

    def _fail(self, failure: Failure) -> Iterator[dict]:
        self.store.rollback()
        self.app.fail(self, failure)
        self.store.set_status(
            self.run,
            RunStatus.FAILED,
            stop_reason=failure.stop_reason,
            error_code=failure.code,
            error_message=failure.message,
            commit=False,
        )
        self.store.clear_working_state(self.run, self.state)
        self.store.commit()
        yield from self.emit(EventType.RUN_FAILED, {**self.app.message_ref(self), "code": failure.code, "message": failure.message})
