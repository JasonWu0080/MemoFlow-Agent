"""Unit tests: decisions, budget, tool schema/registry, context, state machine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from memoflow_agent.budget import AgentBudget, BudgetLimits, RunBudget
from memoflow_agent.context import ContextBuilder, compact_transcript, estimate_tokens
from memoflow_agent.decisions import (
    CLARIFICATION_CHANNEL,
    ClarificationDecision,
    FinalDecision,
    InvalidModelOutput,
    ToolCall,
    ToolCallAssembler,
    ToolCallDecision,
    decide,
    parse_decision,
)
from memoflow_agent.exceptions import BudgetExceeded, CheckpointIncompatible, ContextTooLong
from memoflow_agent.state import InvalidTransition, RunState, RunStatus, check_transition
from memoflow_agent.tools import ToolRegistry, ToolResult, coerce_result, tool

# --- decisions ------------------------------------------------------------------------


def test_assembler_handles_interleaved_fragments():
    assembler = ToolCallAssembler()
    assembler.feed([{"index": 0, "id": "a", "function": {"name": "search", "arguments": '{"q": '}}])
    assembler.feed([{"index": 1, "id": "b", "function": {"name": "read", "arguments": '{"id": 2}'}}])
    assembler.feed([{"index": 0, "function": {"arguments": '"x"}'}}])
    calls = assembler.assemble()
    assert [(c.id, c.name, c.arguments) for c in calls] == [("a", "search", {"q": "x"}), ("b", "read", {"id": 2})]
    assert assembler.raw_calls()[0]["function"]["arguments"] == '{"q": "x"}'


@pytest.mark.parametrize(
    "fragment, code",
    [
        ({"index": 0, "id": "a", "function": {"name": "t", "arguments": "{nope"}}, "tool_arguments_not_json"),
        ({"index": 0, "id": "a", "function": {"name": "t", "arguments": "[1]"}}, "tool_arguments_not_object"),
        ({"index": 0, "id": "a", "function": {"arguments": "{}"}}, "tool_call_without_name"),
    ],
)
def test_assembler_rejects_malformed_calls(fragment, code):
    assembler = ToolCallAssembler()
    assembler.feed([fragment])
    with pytest.raises(InvalidModelOutput) as error:
        assembler.assemble()
    assert error.value.code == code


def test_decisions_come_from_native_calls_never_from_prose():
    assert isinstance(decide("Call search_notes with q=x", []), FinalDecision)
    decision = decide("", [ToolCall(id="1", name="search", arguments={})])
    assert isinstance(decision, ToolCallDecision)
    with pytest.raises(InvalidModelOutput):
        decide("   ", [])


def test_clarification_wins_over_tool_calls_in_the_same_turn():
    decision = decide(
        "",
        [
            ToolCall(id="1", name="delete", arguments={}),
            ToolCall(id="2", name=CLARIFICATION_CHANNEL, arguments={"question": "Which one?"}),
        ],
    )
    assert isinstance(decision, ClarificationDecision) and decision.question == "Which one?"
    with pytest.raises(InvalidModelOutput):
        decide("", [ToolCall(id="3", name=CLARIFICATION_CHANNEL, arguments={"question": ""})])


def test_decisions_parse_from_data():
    assert parse_decision({"type": "final", "content": "x"}).content == "x"


# --- budget ---------------------------------------------------------------------------


def test_budget_guards_and_counters_survive_restore():
    limits = BudgetLimits(max_model_turns=2, max_tool_calls=1, max_run_tokens=1000, max_run_seconds=60, max_consecutive_failures=2)
    state = RunState()
    budget = RunBudget(limits, state)
    budget.record_usage(input_tokens=300, output_tokens=200)
    budget.record_usage(input_tokens=300, output_tokens=200)
    with pytest.raises(BudgetExceeded) as error:
        budget.check_model_turn()
    assert error.value.reason == "max_model_turns"
    restored = RunState.restore(state.model_dump(mode="json"))
    assert (restored.model_turns, restored.tokens_used, restored.input_tokens_used, restored.output_tokens_used) == (2, 1000, 600, 400)
    with pytest.raises(BudgetExceeded) as error:
        RunBudget(limits, restored).check_tool_call()
    assert error.value.reason == "max_run_tokens"


def test_budget_identical_calls_and_separate_token_limits():
    state = RunState()
    budget = RunBudget(BudgetLimits(max_identical_calls=3, max_output_tokens=10), state)
    assert budget.record_identical("a") == 1 and budget.record_identical("a") == 2
    with pytest.raises(BudgetExceeded):
        budget.record_identical("a")
    state.output_tokens_used = 10
    with pytest.raises(BudgetExceeded) as error:
        budget.check_model_turn()
    assert error.value.reason == "max_output_tokens"


def test_agent_budget_constructor():
    limits = AgentBudget(max_model_turns=4, max_tokens=1000)
    assert limits.max_model_turns == 4 and limits.max_run_tokens == 1000
    assert "max_input_tokens" not in limits.as_dict()


# --- state machine -------------------------------------------------------------------------


def test_state_machine_rejects_illegal_transitions():
    check_transition(RunStatus.CREATED, RunStatus.RUNNING)
    check_transition(RunStatus.RUNNING, RunStatus.SUSPENDED)
    check_transition(RunStatus.SUSPENDED, RunStatus.CREATED)
    for current, target in [
        (RunStatus.COMPLETED, RunStatus.RUNNING),
        (RunStatus.SUSPENDED, RunStatus.RUNNING),
        (RunStatus.CANCELLED, RunStatus.CREATED),
        (RunStatus.CREATED, RunStatus.COMPLETED),
    ]:
        with pytest.raises(InvalidTransition):
            check_transition(current, target)


def test_newer_checkpoints_are_refused_not_half_loaded():
    with pytest.raises(CheckpointIncompatible):
        RunState.restore({"version": 99})
    assert RunState.restore(None).model_turns == 0


# --- tools --------------------------------------------------------------------------------


def test_tool_decorator_builds_a_strict_schema():
    @tool(risk="medium")
    def lookup(city: str, days: int = 3, ctx=None) -> dict:
        """Weather for a city.

        Longer text that is not part of the description."""
        return {"city": city, "days": days}

    assert lookup.name == "lookup" and lookup.description == "Weather for a city."
    declaration = lookup.declaration()["function"]
    assert set(declaration["parameters"]["properties"]) == {"city", "days"}
    assert declaration["parameters"]["required"] == ["city"]
    with pytest.raises(ValidationError):
        lookup.input_model.model_validate({"city": "x", "user_id": 7})
    result = lookup.invoke(None, lookup.input_model.model_validate({"city": "Oslo"}))
    assert result.observation == {"city": "Oslo", "days": 3}


def test_async_tools_and_result_coercion():
    @tool
    async def double(n: int) -> int:
        "Double it."
        return n * 2

    assert double.is_async
    assert double.invoke(None, double.input_model.model_validate({"n": 4})).observation == {"result": 8}

    class Out(BaseModel):
        ok: bool

    assert coerce_result(Out(ok=True)).observation == {"ok": True}
    assert coerce_result(ToolResult(observation={"a": 1})).observation == {"a": 1}


def test_registry_refuses_duplicates_and_the_clarification_channel():
    @tool
    def a() -> str:
        "A."
        return "a"

    registry = ToolRegistry([a])
    with pytest.raises(ValueError):
        registry.register(a)

    @tool(name=CLARIFICATION_CHANNEL)
    def sneaky() -> str:
        "Pretend to be the decision channel."
        return ""

    with pytest.raises(ValueError):
        ToolRegistry([sneaky])


# --- context ------------------------------------------------------------------------------


def test_compaction_keeps_calls_and_the_newest_observation():
    transcript = []
    for index in range(4):
        transcript.append({"role": "assistant", "content": None, "tool_calls": [{"id": str(index), "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        transcript.append({"role": "tool", "tool_call_id": str(index), "content": "x" * 4000})
    messages, compacted = compact_transcript(transcript, 1500, stub="STUB")
    assert compacted == 3
    assert [m["content"] for m in messages if m["role"] == "tool"] == ["STUB", "STUB", "STUB", "x" * 4000]
    assert sum(1 for m in messages if m["role"] == "assistant") == 4


def test_context_builder_drops_history_then_refuses_explicitly():
    builder = ContextBuilder(input_budget=400)
    history = [{"role": "user", "content": "old " * 600}, {"role": "assistant", "content": "recent"}]
    built = builder.build(system="sys", request="question", transcript=[], history=history)
    assert built.manifest.history_dropped == 1 and built.messages[1]["content"] == "recent"
    assert built.messages[0]["role"] == "system" and built.messages[-1]["content"] == "question"
    with pytest.raises(ContextTooLong):
        builder.build(system="sys", request="q" * 4000, transcript=[])


def test_token_estimate_counts_cjk_per_character():
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcdefgh") == 2
