"""The public, batteries-included API.

    agent = Agent(model="openai:<model-id>", tools=[calculate])
    result = await agent.run("What is 17 * 23?")
    print(result.output)

``Agent`` wires the kernel's single :class:`~memoflow_agent.loop.AgentLoop`
to a model provider, a tool registry, a policy and a run store (in-memory by
default). Runs that suspend — for a clarification, an authorization or an
approval — are continued with :meth:`Agent.resume`.

Concurrency model (ADR-K1): the loop is synchronous. ``run_sync`` /
``stream_sync`` drive it in the calling thread; ``run`` / ``stream`` drive it
in a worker thread (with the caller's context variables copied) and hand
events back to the event loop. Async tools run on one private event loop
per run.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetLimits
from .context import ContextBuilder
from .events import Event, EventType
from .exceptions import ConfigurationError, MemoFlowAgentError, RunNotFound
from .execution.worker import make_worker_id
from .loop import AgentLoop, Application, CancellationRegistry
from .models.base import ModelProvider, ModelTurn, RetryingProvider
from .models.registry import resolve_model
from .observability import UsageEvent, UsageObserver
from .persistence.memory import InMemoryStore
from .policy import CapabilityPolicy, GrantStore, InMemoryGrantStore, Principal
from .resume import Resume, resume_run
from .state import RunState, RunStatus
from .tools.base import ToolRegistry, ToolResult, ToolSpec
from .tools.executor import ExecOutcome, ExecutionHooks, MappedError, ToolExecutor

_PENDING_EVENTS = {
    EventType.CLARIFICATION_REQUIRED.value,
    EventType.AUTHORIZATION_REQUIRED.value,
    EventType.APPROVAL_REQUIRED.value,
}


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    output: str
    partial: bool = False
    stop_reason: str | None = None
    error_code: str | None = None
    # When suspended: what the run waits for ({"type": "approval.required", ...payload}).
    pending: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    @property
    def suspended(self) -> bool:
        return self.status is RunStatus.SUSPENDED

    @property
    def completed(self) -> bool:
        return self.status is RunStatus.COMPLETED


class _StoreHooks(ExecutionHooks):
    """Idempotency records for side-effecting tools, kept in the run store."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def begin(self, ctx, spec, call, arguments_hash):
        return {"key": None, "tool": spec.name, "arguments_hash": arguments_hash}

    def replay(self, ctx, spec, call, key):
        record = self.store.tool_record(key)
        if record is None:
            return None
        if record.get("status") == "completed":
            return ToolResult(observation=record.get("output") or {}, summary=record.get("summary") or {})
        # Started but never finished: the side effect may or may not have
        # happened. Never retry blindly — ask the tool, or stop.
        reconcile = spec.metadata.get("reconcile")
        if reconcile is not None:
            arguments = spec.input_model.model_validate(call.arguments)
            reconciled = reconcile(ctx, arguments)
            if reconciled is not None:
                return reconciled
        return MappedError(
            "failed",
            "requires_reconciliation",
            "This action was started earlier but its outcome is unknown. Do not retry it; tell the user it needs checking.",
        )

    def mark_idempotency(self, ctx, handle, key):
        handle["key"] = key
        self.store.save_tool_record(key, {"status": "running", "tool": handle["tool"], "arguments_hash": handle["arguments_hash"]})

    def finish(self, ctx, handle, *, status, error_code=None, output=None, authorization=None, duration_ms=0):
        if handle and handle.get("key") and status in ("completed", "failed", "denied"):
            self.store.save_tool_record(
                handle["key"],
                {"status": status, "tool": handle["tool"], "arguments_hash": handle["arguments_hash"], "output": output, "error_code": error_code},
            )


class _AgentApplication(Application):
    def __init__(self, agent: Agent, run: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.run = run

    def flush_text(self, loop, text):
        self.text = text
        self.agent.store.update_run(self.run, output=text)
        return True

    def discard_streamed_text(self, loop):
        self.text = ""
        self.agent.store.update_run(self.run, output="")

    def prepare_turn(self, loop, *, wrap_up):
        built = super().prepare_turn(loop, wrap_up=wrap_up)
        for observer in self.agent.usage_observers:
            observer.before_model_call(self.run.id, estimated_input_tokens=built.total_tokens)
        return built

    def after_model_turn(self, loop, turn: ModelTurn, step, request):
        event = UsageEvent(
            kind="model_call",
            run_id=self.run.id,
            provider=turn.provider,
            model=turn.model,
            input_tokens=turn.prompt_tokens,
            output_tokens=turn.completion_tokens,
            cached_input_tokens=turn.cached_input_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            cost_microusd=turn.cost_microusd,
        )
        for observer in self.agent.usage_observers:
            observer.on_usage(event)
        return None
        yield  # pragma: no cover

    def on_tool_outcome(self, loop, call, spec, outcome: ExecOutcome):
        event = UsageEvent(kind="tool_call", run_id=self.run.id, tool=call.name, ok=outcome.ok)
        for observer in self.agent.usage_observers:
            observer.on_usage(event)

    def complete(self, loop, text, *, partial, stop_reason):
        payload = super().complete(loop, text, partial=partial, stop_reason=stop_reason)
        self.agent.store.update_run(self.run, commit=False, output=text)
        return payload

    def cancel(self, loop, partial_text):
        super().cancel(loop, partial_text)
        self.agent.store.update_run(self.run, commit=False, output=self.text)


class Agent:
    def __init__(
        self,
        model: str | ModelProvider,
        *,
        tools: Iterable[ToolSpec] = (),
        instructions: str | None = None,
        policy: Any = None,
        store: Any = None,
        budget: BudgetLimits | None = None,
        principal: Principal | None = None,
        grants: GrantStore | None = None,
        max_output_tokens: int = 4096,
        input_budget: int | None = None,
        context_builder: ContextBuilder | None = None,
        retries: int = 2,
        usage_observers: Iterable[UsageObserver] = (),
        clarification: bool = True,
        executor_options: dict[str, Any] | None = None,
        model_options: dict[str, Any] | None = None,
    ) -> None:
        provider = resolve_model(model, **(model_options or {}))
        self.registry = ToolRegistry(tools)
        if len(self.registry) and not provider.capabilities.tool_calling:
            raise ConfigurationError("tools were given, but the model's capabilities do not include tool calling")
        self.provider: Any = RetryingProvider(provider, max_attempts=retries) if retries > 1 else provider
        self.instructions = instructions
        self.grants = grants or InMemoryGrantStore()
        self.policy = policy if policy is not None else CapabilityPolicy(grants=self.grants)
        self.store = store if store is not None else InMemoryStore()
        self.budget = budget or BudgetLimits()
        self.principal = principal
        self.max_output_tokens = min(max_output_tokens, provider.capabilities.max_output_tokens)
        window = provider.capabilities.context_window - self.max_output_tokens - 1024
        self.context_builder = context_builder or ContextBuilder(input_budget=max(8_000, input_budget or min(64_000, window)))
        self.usage_observers = list(usage_observers)
        self.clarification = clarification
        self.executor = ToolExecutor(self.registry, self.policy, hooks=_StoreHooks(self.store), **(executor_options or {}))
        self.cancellations = CancellationRegistry()
        self.worker_id = make_worker_id()

    # --- runs ----------------------------------------------------------------------------

    def start(
        self,
        prompt: str,
        *,
        history: list[dict] | None = None,
        session_id: str | None = None,
        interactive: bool = True,
        idempotency_key: str | None = None,
    ) -> str:
        """Create a run without executing it; returns its id."""
        principal_id = self.principal.id if self.principal else None
        if idempotency_key:
            existing = self.store.find_by_idempotency_key(principal_id, idempotency_key)
            if existing is not None:
                return existing.id
        state = RunState(user_texts=[prompt], interactive=interactive)
        # Without a session, a run is its own session: a "session" grant then
        # covers this run (and its resumes) but not the next one.
        session_id = session_id or f"run-session:{uuid.uuid4().hex}"
        run = self.store.create_run(
            state=state,
            principal_id=principal_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            metadata={"history": list(history or [])},
        )
        self.store.commit()
        return run.id

    def execute(self, run_id: str) -> Iterator[Event]:
        """Drive a created / resumed / recovered run until it rests, yielding events."""
        run = self.store.load_run(run_id)
        if run is None:
            raise RunNotFound(str(run_id))
        if not self.store.acquire_lease(run.id, self.worker_id):
            return
        run = self.store.load_run(run_id)  # as of the lease we now hold
        assert run is not None
        state = self.store.load_state(run, RunState)
        app = _AgentApplication(
            self,
            run,
            instructions=self.instructions,
            context_builder=self.context_builder,
            history=run.metadata.get("history"),
            max_output_tokens=self.max_output_tokens,
            principal=self.principal,
            session_id=run.session_id,
            clarification=self.clarification,
        )
        loop = AgentLoop(
            run=run,
            state=state,
            store=self.store,
            events=self.store.event_sink(run),
            app=app,
            executor=self.executor,
            provider=self.provider,
            limits=self.budget,
            worker_id=self.worker_id,
            cancellations=self.cancellations,
        )
        for envelope in loop.execute():
            yield Event.from_envelope(envelope)

    def stream_sync(self, prompt: str, **options: Any) -> Iterator[Event]:
        yield from self.execute(self.start(prompt, **options))

    def run_sync(self, prompt: str, **options: Any) -> RunResult:
        run_id = self.start(prompt, **options)
        return self._result(run_id, list(self.execute(run_id)))

    def resume_sync(self, run_id: str, resume: Resume) -> RunResult:
        return self._result(run_id, list(self.stream_resume_sync(run_id, resume)))

    def stream_resume_sync(self, run_id: str, resume: Resume) -> Iterator[Event]:
        resume_run(
            self.store,
            run_id,
            resume,
            principal_id=self.principal.id if self.principal else None,
            grants=self.grants,
        )
        yield from self.execute(run_id)

    def cancel(self, run_id: str) -> None:
        """Stop a run: a running one stops at its next check, a suspended one is closed."""
        from .state import StopReason

        run = self.store.load_run(run_id)
        if run is None:
            raise RunNotFound(str(run_id))
        status = self.store.get_status(run)
        if status is RunStatus.SUSPENDED:
            state = self.store.load_state(run, RunState)
            self.store.set_status(run, RunStatus.CANCELLED, stop_reason=StopReason.CANCELLED.value, commit=False)
            self.store.clear_working_state(run, state)
            self.store.event_sink(run).emit(EventType.RUN_CANCELLED.value, {}, commit=False)
            self.store.commit()
            return
        if status in (RunStatus.CREATED, RunStatus.RUNNING):
            self.store.request_cancel(run)
            self.cancellations.cancel(run.id)

    def get_run(self, run_id: str) -> Any:
        return self.store.load_run(run_id)

    def events(self, run_id: str, *, after_seq: int = 0) -> list[Event]:
        return [Event.from_envelope(item) for item in self.store.events_after(run_id, after_seq)]

    # --- async API --------------------------------------------------------------------------

    async def run(self, prompt: str, **options: Any) -> RunResult:
        run_id = self.start(prompt, **options)
        events = [event async for event in _in_thread(lambda: self.execute(run_id))]
        return self._result(run_id, events)

    async def stream(self, prompt: str, **options: Any) -> AsyncIterator[Event]:
        run_id = self.start(prompt, **options)
        async for event in _in_thread(lambda: self.execute(run_id)):
            yield event

    async def resume(self, run_id: str, resume: Resume) -> RunResult:
        events = [event async for event in _in_thread(lambda: self.stream_resume_sync(run_id, resume))]
        return self._result(run_id, events)

    async def stream_resume(self, run_id: str, resume: Resume) -> AsyncIterator[Event]:
        async for event in _in_thread(lambda: self.stream_resume_sync(run_id, resume)):
            yield event

    # --- results -------------------------------------------------------------------------------

    def _result(self, run_id: str, events: list[Event]) -> RunResult:
        run = self.store.load_run(run_id)
        if run is None:
            raise RunNotFound(str(run_id))
        pending = None
        completed: dict = {}
        for event in events:
            if event.type in _PENDING_EVENTS:
                pending = {"type": event.type, **event.payload}
            if event.type == EventType.RUN_COMPLETED.value:
                completed = event.payload
        status = self.store.get_status(run)
        return RunResult(
            run_id=run.id,
            status=status,
            output=completed.get("content", run.output or ""),
            partial=bool(completed.get("partial")),
            stop_reason=run.stop_reason,
            error_code=run.error_code,
            pending=pending if status is RunStatus.SUSPENDED else None,
            usage=completed.get("usage") or {},
            events=events,
        )


_DONE = object()


async def _in_thread(factory: Callable[[], Iterator[Event]]) -> AsyncIterator[Event]:
    """Drive a synchronous event iterator in a thread; yield its events asynchronously."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    context = contextvars.copy_context()

    def pump() -> None:
        try:
            for item in factory():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as error:  # noqa: BLE001 — re-raised in the caller
            loop.call_soon_threadsafe(queue.put_nowait, error)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    thread = threading.Thread(target=context.run, args=(pump,), name="memoflow-agent-run", daemon=True)
    thread.start()
    while True:
        item = await queue.get()
        if item is _DONE:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    await asyncio.to_thread(thread.join)


__all__ = ["Agent", "MemoFlowAgentError", "RunResult"]
