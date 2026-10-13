"""Anthropic Messages API provider (streaming, native tool use).

The kernel speaks OpenAI-chat-shaped messages; this adapter converts them:

* leading ``system`` messages → the ``system`` parameter; a ``system``
  message later in the conversation becomes a user text block marked as a
  system note (the Messages API has no mid-conversation system role);
* assistant ``tool_calls`` → ``tool_use`` blocks (arguments parsed from JSON);
* ``tool`` messages → ``tool_result`` blocks in a user turn, adjacent results
  merged, so every ``tool_use`` is answered in the next user turn;
* consecutive same-role messages are merged (the API requires alternation).

Streamed ``tool_use`` blocks are normalized into the same fragment shape as
OpenAI's, so the same assembler builds the calls.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

from ..exceptions import ConfigurationError, ModelProviderError
from ._http import DEFAULT_TIMEOUT, sse_events
from .base import ModelCapabilities, ModelRequest, StreamingProvider

ANTHROPIC_VERSION = "2023-06-01"
_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    converted: list[dict] = []
    leading = True

    def append(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _text_of(message.get("content"))
            if leading:
                system_parts.append(text)
            elif text:
                append("user", [{"type": "text", "text": f"[System note] {text}"}])
            continue
        leading = False
        if role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(message.get("tool_call_id") or ""),
                        "content": _text_of(message.get("content")),
                    }
                ],
            )
        elif role == "assistant":
            blocks: list[dict] = []
            text = _text_of(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                blocks.append({"type": "tool_use", "id": call.get("id"), "name": function.get("name"), "input": arguments})
            append("assistant", blocks)
        else:
            text = _text_of(message.get("content"))
            if text:
                append("user", [{"type": "text", "text": text}])
    system = "\n\n".join(part for part in system_parts if part) or None
    return system, converted


def convert_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted = []
    for declaration in tools:
        function = declaration.get("function") or {}
        converted.append(
            {
                "name": function.get("name"),
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


class AnthropicProvider(StreamingProvider):
    name = "anthropic"
    default_base_url = "https://api.anthropic.com"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        capabilities: ModelCapabilities | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(model, capabilities=capabilities)
        self.api_key = api_key if api_key is not None else os.getenv(self.api_key_env)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)
        self.extra_headers = dict(extra_headers or {})

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ConfigurationError(f"anthropic: no API key (pass api_key= or set {self.api_key_env})")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            **self.extra_headers,
        }

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = convert_messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        tools = convert_tools(request.tools)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto"}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        payload.update(request.options.get("body") or {})
        return payload

    def raw_frames(self, request: ModelRequest) -> Iterator[dict]:
        # Anthropic block index → our tool-call fragment index.
        tool_blocks: dict[int, int] = {}
        for event_name, data in sse_events(
            self.client, f"{self.base_url}/v1/messages", headers=self._headers(), payload=self.build_payload(request)
        ):
            kind = data.get("type") or event_name
            if kind == "message_start":
                message = data.get("message") or {}
                usage = message.get("usage") or {}
                yield {
                    "response_id": message.get("id"),
                    "usage": {
                        "prompt_tokens": (usage.get("input_tokens") or 0)
                        + (usage.get("cache_read_input_tokens") or 0)
                        + (usage.get("cache_creation_input_tokens") or 0),
                        "cached_input_tokens": usage.get("cache_read_input_tokens"),
                        "completion_tokens": usage.get("output_tokens"),
                    },
                }
            elif kind == "content_block_start":
                block = data.get("content_block") or {}
                if block.get("type") == "tool_use":
                    index = len(tool_blocks)
                    tool_blocks[int(data.get("index", 0))] = index
                    yield {"tool_calls": [{"index": index, "id": block.get("id"), "function": {"name": block.get("name"), "arguments": ""}}]}
                elif block.get("type") == "text" and block.get("text"):
                    yield {"delta": block["text"]}
            elif kind == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield {"delta": delta["text"]}
                elif delta.get("type") == "input_json_delta":
                    call_index = tool_blocks.get(int(data.get("index", 0)))
                    if call_index is not None:
                        yield {"tool_calls": [{"index": call_index, "function": {"arguments": delta.get("partial_json") or ""}}]}
                elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                    yield {"reasoning_delta": delta["thinking"]}
            elif kind == "message_delta":
                frame: dict[str, Any] = {}
                stop = (data.get("delta") or {}).get("stop_reason")
                if stop:
                    frame["finish_reason"] = _STOP_REASONS.get(stop, stop)
                usage = data.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    frame["usage"] = {"completion_tokens": usage.get("output_tokens")}
                if frame:
                    yield frame
            elif kind == "error":
                error = data.get("error") or {}
                code = {"overloaded_error": "upstream", "rate_limit_error": "rate_limit", "api_error": "upstream"}.get(
                    str(error.get("type")), "upstream"
                )
                raise ModelProviderError("provider stream error", code=code)
