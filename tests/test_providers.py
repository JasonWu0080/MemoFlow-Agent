"""Provider adapters over a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from memoflow_agent import Agent, tool
from memoflow_agent.exceptions import ConfigurationError, ModelProviderError
from memoflow_agent.models import (
    AnthropicProvider,
    DeepSeekProvider,
    ModelRequest,
    OpenAIProvider,
    TextDelta,
    generate,
    resolve_model,
)
from memoflow_agent.models.anthropic import convert_messages


def sse(*chunks, done=True, named=False):
    lines = []
    for chunk in chunks:
        if named:
            lines.append(f"event: {chunk['type']}")
        lines.append("data: " + json.dumps(chunk))
        lines.append("")
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@tool
def weather(city: str) -> str:
    "Weather."
    return f"sunny in {city}"


def test_openai_streams_text_tool_calls_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        body = sse(
            {"id": "resp_1", "choices": [{"delta": {"content": "Hi "}}]},
            {"id": "resp_1", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "weather", "arguments": '{"ci'}}]}}]},
            {"id": "resp_1", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Oslo"}'}}]}, "finish_reason": "tool_calls"}]},
            {"id": "resp_1", "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 10}}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider("gpt-test", api_key="sk-test", client=client(handler))
    request = ModelRequest(messages=[{"role": "user", "content": "weather?"}], tools=[weather.declaration()], max_output_tokens=64)
    frames = list(provider.stream_turn(request))
    turn = frames[-1]
    assert [f.text for f in frames if isinstance(f, TextDelta)] == ["Hi "]
    assert turn.response_id == "resp_1" and turn.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [("call_a", "weather", {"city": "Oslo"})]
    assert (turn.prompt_tokens, turn.completion_tokens, turn.cached_input_tokens) == (50, 9, 10)
    assert seen["body"]["stream"] is True and seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["max_completion_tokens"] == 64 and seen["auth"] == "Bearer sk-test"


def test_deepseek_uses_max_tokens_and_hides_reasoning():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "prompt_cache_hit_tokens": 2}},
        ))

    provider = DeepSeekProvider("deepseek-test", api_key="k", client=client(handler))
    turn = generate(provider, ModelRequest(messages=[{"role": "user", "content": "q"}], max_output_tokens=32))
    assert turn.text == "Answer" and turn.reasoning_text == "" and turn.cached_input_tokens == 2
    assert seen["url"] == "https://api.deepseek.com/chat/completions" and seen["body"]["max_tokens"] == 32


@pytest.mark.parametrize("status, code", [(401, "auth"), (429, "rate_limit"), (500, "upstream"), (400, "bad_request")])
def test_http_errors_map_to_stable_codes_without_echoing_the_body(status, code):
    provider = OpenAIProvider("m", api_key="k", client=client(lambda r: httpx.Response(status, text="private prompt text")))
    with pytest.raises(ModelProviderError) as error:
        generate(provider, ModelRequest(messages=[{"role": "user", "content": "x"}]))
    assert error.value.code == code and "private" not in str(error.value)


def test_timeouts_and_network_errors_are_retryable_codes():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("down", request=request)

    for handler, code in ((timeout, "timeout"), (down, "network")):
        provider = OpenAIProvider("m", api_key="k", client=client(handler))
        with pytest.raises(ModelProviderError) as error:
            generate(provider, ModelRequest(messages=[{"role": "user", "content": "x"}]))
        assert error.value.code == code


def test_missing_api_key_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider("m", client=client(lambda r: httpx.Response(200)))
    with pytest.raises(ConfigurationError):
        generate(provider, ModelRequest(messages=[{"role": "user", "content": "x"}]))


def test_anthropic_message_conversion_keeps_calls_and_results_paired():
    system, messages = convert_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "weather in Oslo and Rome"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Oslo"}'}},
                {"id": "t2", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Rome"}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "t2", "content": "rain"},
            {"role": "system", "content": "Wrap up now."},
        ]
    )
    assert system == "Be brief."
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["type"] for b in messages[1]["content"]] == ["tool_use", "tool_use"]
    assert messages[1]["content"][0]["input"] == {"city": "Oslo"}
    results = messages[2]["content"]
    assert [b.get("tool_use_id") for b in results[:2]] == ["t1", "t2"]
    assert results[2]["text"].startswith("[System note]")


def test_anthropic_streams_tool_use_into_native_calls():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 30, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking."}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "weather"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ' "Oslo"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"},
            done=False, named=True,
        ))

    provider = AnthropicProvider("claude-test", api_key="ak", client=client(handler))
    request = ModelRequest(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "weather"}], tools=[weather.declaration()])
    turn = generate(provider, request)
    assert turn.text == "Checking." and turn.finish_reason == "tool_calls" and turn.response_id == "msg_1"
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [("toolu_1", "weather", {"city": "Oslo"})]
    assert (turn.prompt_tokens, turn.completion_tokens) == (30, 12)
    assert seen["headers"]["x-api-key"] == "ak" and seen["body"]["system"] == "S"
    assert seen["body"]["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"


def test_anthropic_stream_errors_raise_provider_errors():
    def handler(request):
        return httpx.Response(200, content=sse({"type": "error", "error": {"type": "overloaded_error"}}, done=False, named=True))

    with pytest.raises(ModelProviderError) as error:
        generate(AnthropicProvider("c", api_key="k", client=client(handler)), ModelRequest(messages=[{"role": "user", "content": "x"}]))
    assert error.value.code == "upstream"


def test_model_strings_resolve_to_providers():
    assert isinstance(resolve_model("openai:any-model", api_key="k"), OpenAIProvider)
    assert isinstance(resolve_model("anthropic:any-model", api_key="k"), AnthropicProvider)
    assert isinstance(resolve_model("deepseek:any-model", api_key="k"), DeepSeekProvider)
    for bad in ("openai", "nope:model", "openai:"):
        with pytest.raises(ConfigurationError):
            resolve_model(bad)


def test_agent_over_a_mocked_openai_endpoint_runs_a_tool_round_trip():
    turns = [
        sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "weather", "arguments": '{"city": "Oslo"}'}}]}, "finish_reason": "tool_calls"}]}),
        sse({"choices": [{"delta": {"content": "Sunny in Oslo."}, "finish_reason": "stop"}]}),
    ]
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=turns.pop(0))

    provider = OpenAIProvider("gpt-test", api_key="k", client=client(handler))
    result = Agent(provider, tools=[weather]).run_sync("Weather in Oslo?")
    assert result.output == "Sunny in Oslo."
    assert bodies[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true, "data": {"result": "sunny in Oslo"}}'}


# --- live tests: opt-in, never in default CI ---------------------------------------------------

LIVE = os.getenv("MEMOFLOW_AGENT_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="set MEMOFLOW_AGENT_LIVE_TESTS=1 and provider keys")
@pytest.mark.parametrize(
    "env_model, env_key",
    [("MEMOFLOW_AGENT_LIVE_OPENAI_MODEL", "OPENAI_API_KEY"),
     ("MEMOFLOW_AGENT_LIVE_ANTHROPIC_MODEL", "ANTHROPIC_API_KEY"),
     ("MEMOFLOW_AGENT_LIVE_DEEPSEEK_MODEL", "DEEPSEEK_API_KEY")],
)
def test_live_tool_round_trip(env_model, env_key):  # pragma: no cover - live only
    model = os.getenv(env_model)
    if not model or not os.getenv(env_key):
        pytest.skip(f"{env_model}/{env_key} not set")
    result = Agent(model, tools=[weather]).run_sync("Use the weather tool for Oslo, then answer in one sentence.")
    assert result.completed and any(event.type == "tool.completed" for event in result.events)
