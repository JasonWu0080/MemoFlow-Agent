"""Every example runs, offline, against a scripted model."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from memoflow_agent import InMemoryStore
from memoflow_agent.testing import ScriptedModel, calls, text

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), EXAMPLES / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


def test_all_examples_are_covered():
    names = sorted(path.name for path in EXAMPLES.glob("0*.py"))
    assert names == [
        "01_basic_agent.py", "02_tool_calling.py", "03_streaming.py", "04_multi_provider.py",
        "05_permissions.py", "06_approval.py", "07_persistent_agent.py", "08_knowledge_agent.py",
    ]


def test_01_basic(capsys):
    assert asyncio.run(load("01_basic_agent.py").main(ScriptedModel([text("Agents loop: think, act, observe.")]))).startswith("Agents")


def test_02_tool_calling():
    module = load("02_tool_calling.py")
    model = ScriptedModel([calls(("calculate", {"expression": "17 * 23 + 4"})), text("395")])
    assert asyncio.run(module.main(model)) == "395"
    assert '"result": 395' in model.requests[1].messages[-1]["content"]
    bad = ScriptedModel([calls(("calculate", {"expression": "__import__('os')"})), text("I can't.")])
    asyncio.run(module.main(bad))
    assert "unsupported_expression" in bad.requests[1].messages[-1]["content"]


def test_03_streaming():
    seen = asyncio.run(load("03_streaming.py").main(ScriptedModel([text("Runs survive / restarts and retries / checkpoints hold")])))
    assert seen[0] == "run.started" and "model.delta" in seen and seen[-1] == "run.completed"


def test_04_multi_provider():
    models = [ScriptedModel([calls(("word_count", {"text": "small core reliable agents"})), text("4 words.")]) for _ in range(3)]
    answers = asyncio.run(load("04_multi_provider.py").main(models))
    assert list(answers.values()) == ["4 words."]  # same label; last one wins
    assert asyncio.run(load("04_multi_provider.py").main([])) == {}


@pytest.mark.parametrize("grant, expected", [(True, "Stand-up"), (False, "no access")])
def test_05_permissions(grant, expected):
    model = ScriptedModel([calls(("read_calendar", {"day": "2026-10-14"})), text(f"Answer: {expected}")])
    result = load("05_permissions.py").main(model, grant=grant)
    assert result.completed and expected in result.output
    if grant:
        assert "Stand-up" in model.requests[1].messages[-1]["content"]
    else:
        assert "user_denied_authorization" in model.requests[1].messages[-1]["content"]


@pytest.mark.parametrize("approve", [True, False])
def test_06_approval(approve):
    module = load("06_approval.py")
    module.OUTBOX.clear()
    model = ScriptedModel([calls(("send_email", {"to": "ana@example.com", "subject": "Report", "body": "Ready."})), text("Done.")])
    result = module.main(model, approve=approve)
    assert result.completed and len(module.OUTBOX) == (1 if approve else 0)


def test_07_persistent():
    first = ScriptedModel([calls(("request_clarification", {"question": "Where to?"}))])
    later = ScriptedModel([text("Two days in Kyoto: temples, then Arashiyama.")])
    result = load("07_persistent_agent.py").main(first, later_model=later, store=InMemoryStore())
    assert result.completed and "Kyoto" in result.output
    assert "Kyoto" in later.requests[0].messages[-1]["content"]


def test_08_knowledge():
    model = ScriptedModel([calls(("search_docs", {"query": "refund"})), calls(("read_doc", {"doc_id": "d1"})), text("Yes, within 30 days [d1].")])
    result = load("08_knowledge_agent.py").main(model)
    assert result.output.endswith("[d1].")
    assert "30 days" in model.requests[2].messages[-1]["content"]
