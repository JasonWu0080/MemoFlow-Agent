"""HTTP plumbing shared by the bundled providers (httpx, server-sent events)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from ..exceptions import ModelProviderError

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def error_for_status(status: int, body: str) -> ModelProviderError:
    lowered = body.lower()
    if status in (401, 403):
        code = "auth"
    elif status == 429:
        code = "rate_limit"
    elif status in (400, 413, 422):
        code = "context_length" if ("context" in lowered and ("length" in lowered or "too long" in lowered)) else "bad_request"
    elif status == 408:
        code = "timeout"
    elif status >= 500:
        code = "upstream"
    else:
        code = "bad_request"
    # Never echo the body: it may contain the request (and so private text).
    return ModelProviderError(f"provider returned HTTP {status}", code=code, status_code=status)


def sse_events(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> Iterator[tuple[str | None, dict]]:
    """POST and yield ``(event_name, data)`` for each JSON SSE event."""
    try:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", "replace")
                raise error_for_status(response.status_code, body)
            event_name: str | None = None
            for line in response.iter_lines():
                if not line:
                    event_name = None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield event_name, json.loads(data)
                except json.JSONDecodeError:
                    continue
    except httpx.TimeoutException as error:
        raise ModelProviderError("provider timed out", code="timeout") from error
    except httpx.TransportError as error:
        raise ModelProviderError("network error talking to the provider", code="network") from error
