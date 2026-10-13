"""The single Agent Loop, exercised end to end through the public Agent."""

from __future__ import annotations

import asyncio
import threading

import pytest

from memoflow_agent import Agent, AgentBudget, CapabilityPolicy, Principal, Resume, RunStatus, tool
from memoflow_agent.exceptions import InvalidResume, RunNotFound, RunNotSuspended
from memoflow_agent.models import ModelCapabilities
from memoflow_agent.observability import UsageTotals
from memoflow_agent.testing import Raw, ScriptedModel, calls, fail, text


@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b


def types(events):
    return [event.type for event in events if event.type != "model.delta"]


def test_a_direct_answer_streams_and_completes():
    model = ScriptedModel([text("Hello there, friend.")])
    result = Agent(model).run_sync("hi")
    assert result.completed and result.output == "Hello there, friend."
    deltas = [event for event in result.events if event.type == "model.delta"]
    assert "".join(event.text for event in deltas) == "Hello there, friend."
    assert [event.payload["offset"] for event in deltas][:2] == [0, 4]
    assert types(result.events) == ["run.started", "run.completed"]
    # Without tools nothing is declared, not even the clarification channel.
    assert model.requests[0].tools is None


def test_tool_turn_text_is_reset_and_observations_go_back_to_the_model():
    model = ScriptedModel([calls(("add", {"a": 2, "b": 3}), text="Let me compute"), text("It is 5.")])
    result = Agent(model, tools=[add]).run_sync("2+3?")
    assert types(result.events) == ["run.started", "model.reset", "tool.started", "tool.completed", "run.completed"]
    assert result.output == "It is 5."
    second = model.requests[1].messages
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "add"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1_0", "content": '{"ok": true, "data": {"result": 5}}'}
    names = [d["function"]["name"] for d in model.requests[0].tools]
    assert names == ["add", "request_clarification"]


def test_invalid_arguments_get_one_correction_then_succeed():
    broken = Raw([{"tool_calls": [{"index": 0, "id": "x", "function": {"name": "add", "arguments": "{oops"}}]}, {"finish_reason": "tool_calls"}])
    model = ScriptedModel([broken, calls(("add", {"a": 1, "b": 1})), text("2")])
    result = Agent(model, tools=[add]).run_sync("1+1")
    assert result.completed and result.output == "2"
    assert "not valid JSON" in model.requests[1].messages[-1]["content"]


def test_repeated_invalid_output_fails_the_run():
    empty = Raw([{"finish_reason": "stop"}])
    model = ScriptedModel([empty, empty, empty])
    result = Agent(model, budget=AgentBudget(max_consecutive_failures=3)).run_sync("?")
    assert result.status is RunStatus.FAILED and result.error_code == "invalid_model_output"


def test_turn_budget_ends_with_a_labelled_wrap_up_answer():
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), calls(("add", {"a": 2, "b": 2})), text("Partial: 2 and 4.")])
    result = Agent(model, tools=[add], budget=AgentBudget(max_model_turns=2)).run_sync("go")
    assert result.completed and result.partial and result.output == "Partial: 2 and 4."
    assert model.requests[2].tools is None  # the wrap-up turn cannot call tools


def test_identical_calls_are_short_circuited_then_stopped():
    loop_calls = [calls(("add", {"a": 1, "b": 1})) for _ in range(3)]
    model = ScriptedModel([*loop_calls, text("Done.")])
    result = Agent(model, tools=[add]).run_sync("loop")
    assert result.partial and result.stop_reason == "final"
    duplicate = model.requests[2].messages[-1]["content"]
    assert "duplicate_call" in duplicate


def test_transient_provider_errors_are_retried_before_any_text():
    model = ScriptedModel([fail("rate_limit"), text("ok")])
    result = Agent(model, retries=2).run_sync("x")
    assert result.completed and result.output == "ok"


def test_provider_failure_fails_the_run_cleanly():
    model = ScriptedModel([fail("auth")])
    result = Agent(model).run_sync("x")
    assert result.status is RunStatus.FAILED and result.error_code == "provider.auth"
    assert types(result.events) == ["run.started", "run.failed"]


def test_cancellation_mid_stream_keeps_the_partial_text():
    agent_holder = {}

    def on_frame(frame):
        if frame.get("delta") == "efgh":
            agent_holder["agent"].cancel(agent_holder["run_id"])

    model = ScriptedModel([text("abcdefghijklmnop")], on_frame=on_frame)
    agent = Agent(model)
    run_id = agent.start("long answer")
    agent_holder.update(agent=agent, run_id=run_id)
    events = list(agent.execute(run_id))
    assert events[-1].type == "run.cancelled"
    run = agent.get_run(run_id)
    assert run.status is RunStatus.CANCELLED and run.output == "abcd"


def test_cancelling_a_suspended_run_closes_it():
    model = ScriptedModel([calls(("request_clarification", {"question": "Which?"}))])
    agent = Agent(model, tools=[add])
    result = agent.run_sync("do it")
    agent.cancel(result.run_id)
    assert agent.get_run(result.run_id).status is RunStatus.CANCELLED
    with pytest.raises(RunNotSuspended):
        agent.resume_sync(result.run_id, Resume.answer("the first"))


def test_clarification_suspends_and_resumes_the_same_run():
    model = ScriptedModel(
        [
            calls(("request_clarification", {"question": "Which file?", "options": [{"id": "a", "label": "a.txt"}, {"id": "b", "label": "b.txt"}]})),
            text("Using b.txt."),
        ]
    )
    agent = Agent(model, tools=[add])
    first = agent.run_sync("open the file")
    assert first.suspended and first.pending["type"] == "clarification.required"
    with pytest.raises(InvalidResume):
        agent.resume_sync(first.run_id, Resume.answer(option_ids=["zzz"]))
    second = agent.resume_sync(first.run_id, Resume.answer(option_ids=["b"]))
    assert second.completed and second.output == "Using b.txt."
    assert types(second.events)[0] == "run.resumed"
    answer = model.requests[1].messages[-1]
    assert answer["role"] == "tool" and "b.txt" in answer["content"]


def test_authorization_is_asked_once_and_grants_are_scoped():
    @tool(permission="files.read")
    def read_file(path: str) -> str:
        "Read a file."
        return f"contents of {path}"

    grants_policy = CapabilityPolicy(grantable=frozenset({"files.read"}))
    model = ScriptedModel([calls(("read_file", {"path": "a"})), text("It says hi."), calls(("read_file", {"path": "b"}))])
    agent = Agent(model, tools=[read_file], policy=grants_policy, principal=Principal("alice"))
    grants_policy.grants = agent.grants
    first = agent.run_sync("read a")
    assert first.suspended and first.pending["type"] == "authorization.required" and first.pending["resource"] == "files.read"
    done = agent.resume_sync(first.run_id, Resume.authorize(True, duration="once"))
    assert done.completed and "contents of a" in model.requests[1].messages[-1]["content"]
    # A `once` grant does not cover the next run.
    again = agent.run_sync("read b")
    assert again.suspended


def test_denied_authorization_tells_the_model():
    @tool(permission="files.read")
    def read_file(path: str) -> str:
        "Read a file."
        return "secret"

    policy = CapabilityPolicy(grantable=frozenset({"files.read"}))
    model = ScriptedModel([calls(("read_file", {"path": "a"})), text("I can't read it.")])
    agent = Agent(model, tools=[read_file], policy=policy, principal=Principal("alice"))
    first = agent.run_sync("read a")
    done = agent.resume_sync(first.run_id, Resume.authorize(False))
    assert done.completed and "user_denied_authorization" in model.requests[1].messages[-1]["content"]


def test_approval_runs_the_exact_call_once_and_cannot_be_replayed():
    sent = []

    @tool(side_effect="non_idempotent", requires_approval=True)
    def send_email(to: str, body: str) -> dict:
        "Send an email."
        sent.append((to, body))
        return {"sent": True}

    model = ScriptedModel([calls(("send_email", {"to": "a@example.com", "body": "hi"})), text("Sent.")])
    agent = Agent(model, tools=[send_email], principal=Principal("alice"))
    first = agent.run_sync("email a")
    assert first.suspended and first.pending["type"] == "approval.required" and sent == []
    approval_id = first.pending["approval_id"]
    with pytest.raises(InvalidResume):
        agent.resume_sync(first.run_id, Resume.approve("forged"))
    done = agent.resume_sync(first.run_id, Resume.approve(approval_id))
    assert done.completed and sent == [("a@example.com", "hi")]
    with pytest.raises(RunNotSuspended):
        agent.resume_sync(first.run_id, Resume.approve(approval_id))


def test_rejected_approval_does_not_run_the_tool():
    sent = []

    @tool(requires_approval=True, side_effect="non_idempotent")
    def send_email(to: str) -> str:
        "Send."
        sent.append(to)
        return "ok"

    model = ScriptedModel([calls(("send_email", {"to": "x@example.com"})), text("Not sent.")])
    agent = Agent(model, tools=[send_email], principal=Principal("alice"))
    first = agent.run_sync("email")
    done = agent.resume_sync(first.run_id, Resume.reject(first.pending["approval_id"]))
    assert done.completed and sent == [] and "user_rejected" in model.requests[1].messages[-1]["content"]


def test_only_the_runs_principal_can_resume_it():
    model = ScriptedModel([calls(("request_clarification", {"question": "?"}))])
    owner = Agent(model, tools=[add], principal=Principal("alice"))
    first = owner.run_sync("x")
    intruder = Agent(ScriptedModel([]), tools=[add], principal=Principal("mallory"), store=owner.store)
    with pytest.raises(RunNotFound):
        intruder.resume_sync(first.run_id, Resume.answer("hijack"))


def test_budget_counters_are_not_reset_by_a_resume():
    model = ScriptedModel(
        [calls(("add", {"a": 1, "b": 1})), calls(("request_clarification", {"question": "More?"})), calls(("add", {"a": 2, "b": 2})), text("wrap")]
    )
    agent = Agent(model, tools=[add], budget=AgentBudget(max_model_turns=3))
    first = agent.run_sync("x")
    assert first.suspended
    second = agent.resume_sync(first.run_id, Resume.answer("yes"))
    # Turn 3 was the last allowed one; the wrap-up turn made it 4 calls in total.
    assert second.partial and len(model.requests) == 4


def test_tools_need_a_tool_capable_model():
    from memoflow_agent.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Agent(ScriptedModel(capabilities=ModelCapabilities(tool_calling=False)), tools=[add])


def test_usage_observers_see_every_model_and_tool_call():
    totals = UsageTotals()
    model = ScriptedModel([calls(("add", {"a": 1, "b": 2})), text("3")])
    result = Agent(model, tools=[add], usage_observers=[totals]).run_sync("1+2")
    assert totals.model_calls == 2 and totals.input_tokens == 200
    assert result.usage == {"model_calls": 2, "input_tokens": 200, "output_tokens": 40, "tool_calls": 1}


def test_async_api_streams_events_and_runs_async_tools():
    @tool
    async def slow_add(a: int, b: int) -> int:
        "Add slowly."
        await asyncio.sleep(0)
        return a + b

    async def main():
        model = ScriptedModel([calls(("slow_add", {"a": 4, "b": 5})), text("9")])
        agent = Agent(model, tools=[slow_add])
        seen = [event.type async for event in agent.stream("4+5")]
        result = await Agent(ScriptedModel([text("hello")])).run("hi")
        return seen, result

    seen, result = asyncio.run(main())
    assert "tool.completed" in seen and seen[-1] == "run.completed"
    assert result.output == "hello"


def test_idempotent_start_returns_the_same_run():
    agent = Agent(ScriptedModel([text("a")]))
    assert agent.start("x", idempotency_key="k") == agent.start("x", idempotency_key="k")


def test_concurrent_runs_do_not_interfere():
    results = []

    def run(index):
        results.append(Agent(ScriptedModel([text(f"answer {index}")])).run_sync("q").output)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == sorted(f"answer {index}" for index in range(8))
