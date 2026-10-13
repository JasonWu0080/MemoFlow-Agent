"""The Agent Decision Protocol.

A model turn ends in exactly one of three decisions:

* ``FINAL`` — the answer;
* ``TOOL_CALL`` — one or more registered tools to run;
* ``CLARIFICATION`` — a question the model cannot responsibly guess past.

Everything else a run can end in — suspended for authorization or approval,
budget exceeded, failed, cancelled — is produced by the *runtime*, never by
the model (see :mod:`memoflow_agent.state`). Keeping the two apart is what
lets a policy overrule the model without the model having any way to express
"skip the check".

The decision is read from the provider's native tool-calling protocol, never
from the model's prose: no regular expression ever decides whether a tool
runs. Clarification travels over the same native channel as a reserved
function (:data:`CLARIFICATION_CHANNEL`) because that is the only structured
channel every tool-capable provider offers mid-stream — but it is a decision
channel, not a tool: it is not in any tool registry, the tool executor never
sees it, and the runtime alone decides what waiting for the user means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

CLARIFICATION_CHANNEL = "request_clarification"
MAX_CLARIFICATION_OPTIONS = 6

DEFAULT_CLARIFICATION_DESCRIPTION = (
    "Ask the user one question when the request has more than one reasonable "
    "reading and guessing would change the outcome. Do not use it for small "
    "details you can decide yourself, and never together with other tools in "
    "the same turn."
)


class DecisionType(StrEnum):
    FINAL = "final"
    TOOL_CALL = "tool_call"
    CLARIFICATION = "clarification"


class ToolCall(BaseModel):
    """One tool call the model decided on, as the provider identified it."""

    id: str
    name: str
    arguments: dict[str, Any]


class ClarificationOption(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=400)


class FinalDecision(BaseModel):
    type: Literal["final"] = "final"
    content: str


class ToolCallDecision(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_calls: list[ToolCall] = Field(min_length=1)


class ClarificationDecision(BaseModel):
    type: Literal["clarification"] = "clarification"
    reason: str = Field(default="", max_length=500)
    question: str = Field(min_length=1, max_length=1000)
    options: list[ClarificationOption] | None = Field(default=None, max_length=MAX_CLARIFICATION_OPTIONS)
    allow_multiple: bool = False
    allow_free_text: bool = True


Decision = Annotated[
    FinalDecision | ToolCallDecision | ClarificationDecision,
    Field(discriminator="type"),
]
_DECISION_ADAPTER = TypeAdapter(Decision)


def parse_decision(payload: dict) -> FinalDecision | ToolCallDecision | ClarificationDecision:
    """Validate a decision that arrived as data (checkpoints, tests)."""
    return _DECISION_ADAPTER.validate_python(payload)


class InvalidModelOutput(ValueError):
    """The model produced something the protocol cannot interpret."""

    def __init__(self, code: str, detail: str = "", *, tool_call_id: str | None = None):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
        self.tool_call_id = tool_call_id


def clarification_declaration(description: str = DEFAULT_CLARIFICATION_DESCRIPTION) -> dict:
    """The reserved decision-channel declaration handed to the provider.

    Applications pass their own ``description`` when they need the wording
    to stay stable (it is part of the prompt the model sees).
    """
    return {
        "type": "function",
        "function": {
            "name": CLARIFICATION_CHANNEL,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question, in the user's language."},
                    "reason": {"type": "string", "description": "Why you cannot proceed without it (short)."},
                    "options": {
                        "type": "array",
                        "maxItems": MAX_CLARIFICATION_OPTIONS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["id", "label"],
                        },
                    },
                    "allow_multiple": {
                        "type": "boolean",
                        "description": "True only when the request clearly allows several targets.",
                    },
                    "allow_free_text": {"type": "boolean"},
                },
                "required": ["question"],
            },
        },
    }


@dataclass
class _PartialCall:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class ToolCallAssembler:
    """Reassembles streamed tool-call fragments into whole calls.

    OpenAI-compatible providers stream a tool call as an ``index``, then an
    ``id`` and ``function.name`` in one fragment and ``function.arguments``
    spread over many. Fragments for different calls may interleave. Other
    providers are normalized into the same fragment shape by their adapter.
    """

    calls: dict[int, _PartialCall] = field(default_factory=dict)

    def feed(self, fragments: list[dict]) -> None:
        for fragment in fragments or []:
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index")
            if not isinstance(index, int):
                index = len(self.calls)
            call = self.calls.setdefault(index, _PartialCall(index=index))
            if fragment.get("id"):
                call.id = str(fragment["id"])
            function = fragment.get("function") or {}
            if function.get("name"):
                call.name += str(function["name"])
            if function.get("arguments"):
                call.arguments += str(function["arguments"])

    @property
    def started(self) -> bool:
        return bool(self.calls)

    def assemble(self) -> list[ToolCall]:
        assembled: list[ToolCall] = []
        for position, call in enumerate(sorted(self.calls.values(), key=lambda c: c.index)):
            call_id = call.id or f"call_{position}"
            if not call.name:
                raise InvalidModelOutput("tool_call_without_name", tool_call_id=call_id)
            raw = call.arguments.strip() or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as error:
                raise InvalidModelOutput("tool_arguments_not_json", str(error)[:120], tool_call_id=call_id) from error
            if not isinstance(arguments, dict):
                raise InvalidModelOutput("tool_arguments_not_object", tool_call_id=call_id)
            assembled.append(ToolCall(id=call_id, name=call.name, arguments=arguments))
        return assembled

    def raw_calls(self) -> list[dict]:
        """The calls exactly as the provider sent them, for the transcript."""
        return [
            {
                "id": call.id or f"call_{position}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for position, call in enumerate(sorted(self.calls.values(), key=lambda c: c.index))
        ]


def decide(text: str, tool_calls: list[ToolCall]) -> FinalDecision | ToolCallDecision | ClarificationDecision:
    """Normalize one finished model turn into a decision.

    A clarification wins over any tool calls in the same turn: a model that
    is unsure what the user meant should not also act on its best guess.
    """
    if tool_calls:
        for call in tool_calls:
            if call.name == CLARIFICATION_CHANNEL:
                try:
                    return ClarificationDecision(type="clarification", **call.arguments)
                except (ValidationError, TypeError) as error:
                    raise InvalidModelOutput("clarification_invalid", str(error)[:200], tool_call_id=call.id) from error
        return ToolCallDecision(tool_calls=tool_calls)
    if text.strip():
        return FinalDecision(content=text)
    raise InvalidModelOutput("empty_turn")
