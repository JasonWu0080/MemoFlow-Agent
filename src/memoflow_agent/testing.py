"""Test doubles: a scripted model provider that needs no network or key.

    model = ScriptedModel([
        calls(("calculate", {"expression": "17*23"})),
        text("17 × 23 = 391"),
    ])
    agent = Agent(model, tools=[calculate])

Each script item is one model turn. Text is streamed in small deltas and
tool calls are streamed as fragmented OpenAI-style deltas, so tests exercise
the same assembly path as real providers. Every request is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ModelProviderError
from .models.base import ModelCapabilities, ModelRequest, StreamingProvider


@dataclass
class Text:
    text: str
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class Calls:
    calls: list[tuple[str, dict]]
    text: str = ""
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class Fail:
    code: str = "upstream"


@dataclass
class Raw:
    """Raw provider frames, for edge cases (malformed arguments, …)."""

    frames: list[dict]


def text(value: str, **usage: int) -> Text:
    return Text(value, **usage)


def calls(*items: tuple[str, dict], text: str = "") -> Calls:
    return Calls(list(items), text=text)


def fail(code: str = "upstream") -> Fail:
    return Fail(code)


@dataclass
class ScriptedModel(StreamingProvider):
    script: list[Any] = field(default_factory=list)
    model: str = "scripted"
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    requests: list[ModelRequest] = field(default_factory=list)
    on_frame: Callable[[dict], None] | None = None
    keep_reasoning: bool = False
    name: str = "scripted"

    def raw_frames(self, request: ModelRequest) -> Iterator[dict]:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("scripted model called more times than scripted")
        turn = self.script.pop(0)
        for frame in self._frames(turn):
            if self.on_frame is not None:
                self.on_frame(frame)
            yield frame

    def _frames(self, turn: Any) -> Iterator[dict]:
        if isinstance(turn, Fail):
            raise ModelProviderError("scripted failure", code=turn.code)
        if isinstance(turn, Raw):
            yield from turn.frames
            return
        if isinstance(turn, str):
            turn = Text(turn)
        if turn.text:
            for start in range(0, len(turn.text), 4):
                yield {"delta": turn.text[start : start + 4]}
        if isinstance(turn, Calls):
            for index, (name, arguments) in enumerate(turn.calls):
                encoded = json.dumps(arguments, ensure_ascii=False)
                yield {"tool_calls": [{"index": index, "id": f"call_{len(self.requests)}_{index}", "function": {"name": name, "arguments": ""}}]}
                for start in range(0, len(encoded), 7):
                    yield {"tool_calls": [{"index": index, "function": {"arguments": encoded[start : start + 7]}}]}
            yield {"finish_reason": "tool_calls"}
        else:
            yield {"finish_reason": "stop"}
        yield {"usage": {"prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens}}


__all__ = ["Calls", "Fail", "Raw", "ScriptedModel", "Text", "calls", "fail", "text"]
