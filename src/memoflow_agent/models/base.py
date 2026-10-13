"""Model provider interface: one streamed turn, normalized.

A provider turns a :class:`ModelRequest` (OpenAI-chat-shaped messages plus
tool declarations) into a stream of frames:

* :class:`TextDelta` — user-visible text, as it arrives;
* :class:`ProviderEvent` — anything else worth surfacing (retries, fallback,
  provider-hosted tool activity);
* finally exactly one :class:`ModelTurn` with the assembled tool calls,
  usage and metadata.

Tool calls are read from the provider's native tool-calling protocol and
assembled with :class:`~memoflow_agent.decisions.ToolCallAssembler`; prose is
never parsed for them.

The interface is synchronous (a generator). The kernel loop is synchronous
so the same loop can run in a thread pool next to synchronous database code;
:class:`~memoflow_agent.Agent` offers the async API on top (ADR-K1).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..decisions import ToolCall, ToolCallAssembler
from ..exceptions import ModelProviderError

logger = logging.getLogger("memoflow_agent.models")

RETRYABLE_CODES = frozenset({"rate_limit", "upstream", "timeout", "network"})


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model can do — declared or verified, never guessed.

    Defaults are conservative: only ``streaming`` and ``tool_calling`` (the
    chat APIs these providers expose support both); everything else must be
    switched on explicitly by whoever verified it.
    """

    tool_calling: bool = True
    streaming: bool = True
    structured_output: bool = False
    native_web_search: bool = False
    vision: bool = False
    reasoning: bool = False
    parallel_tool_calls: bool = False
    context_window: int = 128_000
    max_output_tokens: int = 8_192

    def require(self, capability: str) -> None:
        from ..exceptions import ConfigurationError

        if not getattr(self, capability, False):
            raise ConfigurationError(f"model does not declare the {capability!r} capability")


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict]
    tools: list[dict] | None = None
    max_output_tokens: int = 4096
    allow_fallback: bool = False
    temperature: float | None = None
    # Provider-specific extras (e.g. hosted tools) and, for application
    # adapters, the application's own request object.
    options: dict[str, Any] = field(default_factory=dict)
    native: Any = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ProviderEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict] = field(default_factory=list)
    # Hidden reasoning, retained only when a transport needs it echoed back.
    reasoning_text: str = ""
    finish_reason: str | None = None
    provider: str = ""
    model: str = ""
    response_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_microusd: int | None = None
    usage_reported: bool = False
    ttft_ms: int | None = None
    duration_ms: int = 0
    attempts: int = 1
    fell_back: bool = False
    assembly_error: Exception | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def usage_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_microusd": self.cost_microusd,
            "provider_reported": self.usage_reported,
        }


Frame = TextDelta | ProviderEvent | ModelTurn


class TurnCancelled(RuntimeError):
    """The run was cancelled while this turn was streaming."""

    def __init__(self, partial: ModelTurn):
        super().__init__("cancelled")
        self.partial = partial


@runtime_checkable
class ModelProvider(Protocol):
    name: str
    model: str
    capabilities: ModelCapabilities

    def stream_turn(self, request: ModelRequest, *, is_cancelled: Callable[[], bool] = ...) -> Iterator[Frame]: ...


def generate(provider: ModelProvider, request: ModelRequest) -> ModelTurn:
    """Run one turn to completion and return it (no streaming)."""
    turn: ModelTurn | None = None
    for frame in provider.stream_turn(request, is_cancelled=lambda: False):
        if isinstance(frame, ModelTurn):
            turn = frame
    if turn is None:  # pragma: no cover - providers always end with a turn
        raise ModelProviderError("provider ended without a turn", code="upstream")
    return turn


class StreamingProvider:
    """Base for HTTP providers: subclasses yield raw frames, this assembles the turn.

    Raw frames are dicts with any of: ``delta`` (str), ``reasoning_delta``,
    ``tool_calls`` (OpenAI-shaped fragments), ``usage`` (``prompt_tokens``,
    ``completion_tokens``, ``cached_input_tokens``, ``reasoning_tokens``),
    ``finish_reason``, ``response_id``.
    """

    name = "provider"

    def __init__(self, model: str, *, capabilities: ModelCapabilities | None = None, keep_reasoning: bool = False):
        self.model = model
        self.capabilities = capabilities or ModelCapabilities()
        self.keep_reasoning = keep_reasoning

    def raw_frames(self, request: ModelRequest) -> Iterator[dict]:  # pragma: no cover - abstract
        raise NotImplementedError

    def stream_turn(self, request: ModelRequest, *, is_cancelled: Callable[[], bool] = lambda: False) -> Iterator[Frame]:
        if request.tools and not self.capabilities.tool_calling:
            raise ModelProviderError("this model does not support tool calling", code="tools_unsupported")
        started = time.monotonic()
        turn = ModelTurn(provider=self.name, model=self.model)
        assembler = ToolCallAssembler()
        reasoning: list[str] = []
        stream = self.raw_frames(request)
        try:
            for frame in stream:
                if is_cancelled():
                    turn.raw_tool_calls = assembler.raw_calls()
                    turn.duration_ms = round((time.monotonic() - started) * 1000)
                    raise TurnCancelled(turn)
                if frame.get("response_id"):
                    turn.response_id = str(frame["response_id"])
                usage = frame.get("usage")
                if usage:
                    turn.usage_reported = True
                    turn.prompt_tokens = int(usage.get("prompt_tokens") or turn.prompt_tokens or 0)
                    turn.completion_tokens = int(usage.get("completion_tokens") or turn.completion_tokens or 0)
                    turn.cached_input_tokens = int(usage.get("cached_input_tokens") or turn.cached_input_tokens or 0)
                    turn.reasoning_tokens = int(usage.get("reasoning_tokens") or turn.reasoning_tokens or 0)
                if frame.get("tool_calls"):
                    assembler.feed(frame["tool_calls"])
                if frame.get("reasoning_delta"):
                    reasoning.append(frame["reasoning_delta"])
                if frame.get("finish_reason"):
                    turn.finish_reason = str(frame["finish_reason"])
                if frame.get("metadata"):
                    turn.provider_metadata.update(frame["metadata"])
                piece = frame.get("delta")
                if piece:
                    if turn.ttft_ms is None:
                        turn.ttft_ms = round((time.monotonic() - started) * 1000)
                    turn.text += piece
                    yield TextDelta(piece)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        turn.raw_tool_calls = assembler.raw_calls()
        try:
            turn.tool_calls = assembler.assemble()
        except Exception as error:  # InvalidModelOutput, surfaced by the loop
            turn.assembly_error = error
        if self.keep_reasoning:
            turn.reasoning_text = "".join(reasoning)
        turn.duration_ms = round((time.monotonic() - started) * 1000)
        yield turn


class RetryingProvider:
    """Retries a turn on transient errors — only before any text reached the caller.

    Text that is already on the user's screen is never repeated: once a
    delta has been yielded, a failure is raised as is. Tool-call fragments are
    not user-visible, so a turn that fails half way through emitting a tool
    call is retried whole.
    """

    def __init__(self, inner: ModelProvider, *, max_attempts: int = 2, backoff_seconds: float = 0.5, sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.max_attempts = max(1, max_attempts)
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.inner.capabilities

    def stream_turn(self, request: ModelRequest, *, is_cancelled: Callable[[], bool] = lambda: False) -> Iterator[Frame]:
        attempt = 0
        while True:
            attempt += 1
            committed = False
            try:
                for frame in self.inner.stream_turn(request, is_cancelled=is_cancelled):
                    if isinstance(frame, TextDelta):
                        committed = True
                    if isinstance(frame, ModelTurn):
                        frame.attempts = attempt
                    yield frame
                return
            except ModelProviderError as error:
                if committed or error.code not in RETRYABLE_CODES or attempt >= self.max_attempts:
                    raise
                logger.warning("model turn attempt %d failed before output (code=%s)", attempt, error.code)
                yield ProviderEvent("retry", {"attempt": attempt, "reason": error.code, "model": self.model})
                if self.backoff_seconds:
                    self._sleep(min(self.backoff_seconds * (2 ** (attempt - 1)), 2.0))
