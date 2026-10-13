"""Context engine: assembling a model call's input within a token budget.

    system          stable instructions (cache-friendly: never per-turn data)
    history         earlier turns, newest kept first
    request         the user's message
    transcript      this run's own tool calls and observations

Compaction is layered, and never silent:

1. old tool observations are replaced by a stub the model can re-fetch with
   the same call (the assistant tool-call message and the tool message stay,
   so every call keeps its result — providers require it);
2. older conversation history is dropped (counted in the manifest);
3. if the system prompt, the request and the newest observation still do not
   fit, :class:`~memoflow_agent.exceptions.ContextTooLong` is raised.

Untrusted content (tool results, documents, web pages) only ever enters as
``user``/``tool`` data — never as system text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ContextTooLong

TokenEstimator = Callable[[str], int]

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")

DEFAULT_COMPACTED_STUB = (
    "(An earlier tool result was omitted to save space. Call the same tool with the same "
    "arguments again if you still need it.)"
)


def estimate_tokens(text: str) -> int:
    """Rough, dependency-free token estimate: ~1 per CJK character, ~1 per 4 other characters.

    Errs on the high side for English. Plug in a real tokenizer through
    ``ContextBuilder(estimator=...)`` when budgets must be tight.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 4)


def message_tokens(messages: list[dict], estimator: TokenEstimator = estimate_tokens) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimator(content)
        if message.get("tool_calls"):
            total += estimator(json.dumps(message["tool_calls"], ensure_ascii=False))
        total += 4
    return total


def declaration_tokens(declarations: list[dict], estimator: TokenEstimator = estimate_tokens) -> int:
    if not declarations:
        return 0
    return estimator(json.dumps(declarations, ensure_ascii=False))


def compact_transcript(
    transcript: list[dict],
    budget_tokens: int,
    *,
    count_tokens: Callable[[list[dict]], int] | None = None,
    stub: str = DEFAULT_COMPACTED_STUB,
) -> tuple[list[dict], int]:
    """Stub the oldest observations until the transcript fits.

    Returns ``(messages, compacted_count)``. Assistant tool-call messages are
    always kept and the newest observation is never compacted — it is what
    the model is about to reason over.
    """
    count = count_tokens or message_tokens
    messages = [dict(message) for message in transcript]
    compacted = 0
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    for index in tool_indexes[:-1]:
        if count(messages) <= budget_tokens:
            break
        if messages[index].get("content") != stub:
            messages[index]["content"] = stub
            compacted += 1
    return messages, compacted


@dataclass
class ContextManifest:
    input_budget: int
    system_tokens: int = 0
    tool_tokens: int = 0
    history_tokens: int = 0
    history_messages: int = 0
    history_dropped: int = 0
    request_tokens: int = 0
    transcript_tokens: int = 0
    compacted_observations: int = 0

    @property
    def total(self) -> int:
        return self.system_tokens + self.tool_tokens + self.history_tokens + self.request_tokens + self.transcript_tokens

    def as_dict(self) -> dict:
        return {
            "input_budget": self.input_budget,
            "total": self.total,
            "system": self.system_tokens,
            "tools": self.tool_tokens,
            "history": self.history_tokens,
            "history_messages": self.history_messages,
            "history_dropped": self.history_dropped,
            "request": self.request_tokens,
            "transcript": self.transcript_tokens,
            "compacted_observations": self.compacted_observations,
        }


@dataclass
class BuiltContext:
    messages: list[dict]
    manifest: Any
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(getattr(self.manifest, "total", 0))


@dataclass
class ContextBuilder:
    """Default context builder used by :class:`~memoflow_agent.Agent`."""

    input_budget: int = 64_000
    estimator: TokenEstimator = estimate_tokens
    compacted_stub: str = DEFAULT_COMPACTED_STUB

    def count(self, messages: list[dict]) -> int:
        return message_tokens(messages, self.estimator)

    def build(
        self,
        *,
        system: str,
        request: str,
        transcript: list[dict],
        history: list[dict] | None = None,
        declarations: list[dict] | None = None,
        extra_system: list[str] | None = None,
    ) -> BuiltContext:
        manifest = ContextManifest(input_budget=self.input_budget)
        system_message = {"role": "system", "content": system}
        request_message = {"role": "user", "content": request}
        tail = [{"role": "system", "content": note} for note in extra_system or []]
        manifest.system_tokens = self.count([system_message])
        manifest.tool_tokens = declaration_tokens(declarations or [], self.estimator)
        manifest.request_tokens = self.count([request_message])
        fixed = manifest.system_tokens + manifest.tool_tokens + manifest.request_tokens + self.count(tail)
        if fixed > self.input_budget:
            raise ContextTooLong(fixed, self.input_budget)
        remaining = self.input_budget - fixed

        # 1. Transcript (the run's working state), oldest observations stubbed first.
        compacted, manifest.compacted_observations = compact_transcript(
            transcript, max(remaining // 2, remaining - 6_000), count_tokens=self.count, stub=self.compacted_stub
        )
        manifest.transcript_tokens = self.count(compacted)
        if manifest.transcript_tokens > remaining:
            raise ContextTooLong(fixed + manifest.transcript_tokens, self.input_budget)
        remaining -= manifest.transcript_tokens

        # 2. History, newest first.
        kept: list[dict] = []
        history = list(history or [])
        cutoff = len(history)
        for index in range(len(history) - 1, -1, -1):
            item = {"role": history[index]["role"], "content": history[index].get("content") or ""}
            tokens = self.count([item])
            if manifest.history_tokens + tokens > remaining:
                break
            kept.insert(0, item)
            manifest.history_tokens += tokens
            cutoff = index
        manifest.history_messages = len(kept)
        manifest.history_dropped = cutoff

        messages = [system_message, *kept, request_message, *compacted, *tail]
        return BuiltContext(messages=messages, manifest=manifest)
