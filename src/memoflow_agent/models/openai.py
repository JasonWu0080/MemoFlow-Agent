"""OpenAI Chat Completions provider (streaming, native tool calling)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx

from ..exceptions import ConfigurationError
from ._http import DEFAULT_TIMEOUT, sse_events
from .base import ModelCapabilities, ModelRequest, StreamingProvider


class OpenAIProvider(StreamingProvider):
    """``POST {base_url}/chat/completions`` with ``stream: true``.

    Messages are already OpenAI-chat-shaped; tool declarations are passed as
    ``tools`` with ``tool_choice: "auto"``. Usage comes from the final chunk
    (``stream_options.include_usage``).
    """

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    max_tokens_field = "max_completion_tokens"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        capabilities: ModelCapabilities | None = None,
        extra_headers: dict[str, str] | None = None,
        keep_reasoning: bool = False,
    ):
        super().__init__(model, capabilities=capabilities, keep_reasoning=keep_reasoning)
        self.api_key = api_key if api_key is not None else os.getenv(self.api_key_env)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)
        self.extra_headers = dict(extra_headers or {})

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ConfigurationError(f"{self.name}: no API key (pass api_key= or set {self.api_key_env})")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", **self.extra_headers}

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            self.max_tokens_field: request.max_output_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        payload.update(request.options.get("body") or {})
        return payload

    def raw_frames(self, request: ModelRequest) -> Iterator[dict]:
        for _event, chunk in sse_events(
            self.client, f"{self.base_url}/chat/completions", headers=self._headers(), payload=self.build_payload(request)
        ):
            frame: dict[str, Any] = {}
            if chunk.get("id"):
                frame["response_id"] = chunk["id"]
            usage = chunk.get("usage")
            if usage:
                details = usage.get("prompt_tokens_details") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                frame["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_input_tokens": details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens"),
                    "reasoning_tokens": completion_details.get("reasoning_tokens"),
                }
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    frame["delta"] = delta["content"]
                if delta.get("reasoning_content"):
                    frame["reasoning_delta"] = delta["reasoning_content"]
                if delta.get("tool_calls"):
                    frame["tool_calls"] = delta["tool_calls"]
                if choice.get("finish_reason"):
                    frame["finish_reason"] = choice["finish_reason"]
            if frame:
                yield frame
