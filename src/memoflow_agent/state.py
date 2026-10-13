"""Run vocabulary and the durable working state of a run.

The working state (:class:`RunState`) is everything the loop needs to pick a
run back up — after a pause for the user, or after the process that was
executing it died. It is what a checkpoint stores.

Applications extend :class:`RunState` and :class:`PendingAction` with their
own fields by subclassing (pydantic models); the kernel only reads and writes
the fields declared here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .exceptions import CheckpointIncompatible

CHECKPOINT_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    """Public run states. Only the runtime moves a run between them."""

    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
# `created` = ready to execute (new, or resumed). `running` = an executor holds
# it; seeing `running` when starting means the previous executor died.
EXECUTABLE_STATUSES = frozenset({RunStatus.CREATED, RunStatus.RUNNING})

# Every legal transition. `running → running` is a recovery (lease taken over).
_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.RUNNING, RunStatus.SUSPENDED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.SUSPENDED: frozenset({RunStatus.CREATED, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    def __init__(self, current: RunStatus, target: RunStatus):
        super().__init__(f"illegal run transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def check_transition(current: RunStatus, target: RunStatus) -> None:
    """Raise :class:`InvalidTransition` unless ``current → target`` is legal."""
    if target not in _TRANSITIONS[current]:
        raise InvalidTransition(current, target)


class StepType(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    CLARIFICATION = "clarification"
    AUTHORIZATION = "authorization"
    CONFIRMATION = "confirmation"


class WaitingReason(StrEnum):
    CLARIFICATION = "clarification"
    # Access to a class of resources or a capability (a grant).
    AUTHORIZATION = "authorization"
    # One concrete operation with its exact arguments (an approval), or the
    # user's verdict on something a tool prepared.
    CONFIRMATION = "confirmation"


class StopReason(StrEnum):
    """Why a run stopped, when that is not simply "the model answered"."""

    FINAL = "final"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_ERROR = "provider_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    WAITING_EXPIRED = "waiting_expired"
    INTERNAL_ERROR = "internal_error"


class PendingAction(BaseModel):
    """The one thing a suspended run is waiting on.

    Saved whole, so the decision the model already made is executed as made
    once the user answers — the model is not asked again, and nothing after
    it in the same batch has run.
    """

    reason: WaitingReason
    # The tool call that needs authorization / approval, or produced the
    # result the user is confirming.
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Tool calls from the same model turn that had not started yet.
    remaining_calls: list[dict[str, Any]] = Field(default_factory=list)
    # clarification
    question: str | None = None
    clarification: dict[str, Any] | None = None
    # authorization: the resource / capability that needs a grant
    resource: str | None = None
    # confirmation of a call before it runs (see memoflow_agent.policy.approvals)
    approval: dict[str, Any] | None = None


class RunState(BaseModel):
    """Working memory of one run.

    ``transcript`` holds the loop's own messages in OpenAI chat format —
    assistant turns that called tools and the tool results that answered
    them. It is the only place tool observations live, it is never logged,
    and it is dropped from the checkpoint when the run ends.
    """

    pending_class: ClassVar[type[PendingAction]] = PendingAction

    version: int = CHECKPOINT_SCHEMA_VERSION
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    # The user's words for this run: the original message plus any
    # clarification answers.
    user_texts: list[str] = Field(default_factory=list)
    # `name:arguments_hash` → count, for repeated-call detection.
    call_signatures: dict[str, int] = Field(default_factory=dict)
    model_turns: int = 0
    tool_calls: int = 0
    consecutive_failures: int = 0
    tokens_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    side_effects_started: bool = False
    pending: PendingAction | None = None
    # Tool calls the model already decided on that run before the next model
    # turn — set when a run resumes after authorization or approval.
    queued_calls: list[dict[str, Any]] = Field(default_factory=list)
    # tool_call_id → arguments hash of calls the user approved (consumed once).
    approved_calls: dict[str, str] = Field(default_factory=dict)
    # Set once a budget guard trips, so the final answer is labelled partial.
    partial_reason: str | None = None
    # Whether the client can show interactive cards (clarification,
    # authorization, approval). Non-interactive runs never suspend for them.
    interactive: bool = True
    # Set when material read in this run tried to instruct the assistant.
    security_flags: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_pending(cls, **fields: Any) -> PendingAction:
        return cls.pending_class(**fields)

    @classmethod
    def restore(cls, snapshot: dict | None):
        """Load a checkpoint snapshot, refusing ones from a newer schema."""
        if not snapshot:
            return cls()
        version = int(snapshot.get("version") or 1)
        if version > CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointIncompatible(
                f"checkpoint schema {version} is newer than supported {CHECKPOINT_SCHEMA_VERSION}"
            )
        return cls.model_validate(snapshot)
