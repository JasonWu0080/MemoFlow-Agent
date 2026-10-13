"""Applying the user's answer to a suspended run.

The same run continues: a clarification answer, an authorization decision
or an approval is fed back to the decision the model already made, and the
loop picks up from there. Nothing here trusts the client about what is
pending — the pending action comes from the checkpoint.

The functions operating on :class:`RunState` are shared by every
application (MemoFlow uses them with its own database side effects);
:func:`resume_run` is the complete flow over a generic run store.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from .exceptions import InvalidResume, RunNotFound, RunNotSuspended
from .policy import GrantDuration, GrantStore, new_grant, validate_approval
from .state import PendingAction, RunState, RunStatus, WaitingReason
from .util import stable_hash


class Resume(BaseModel):
    kind: Literal["clarification", "authorization", "confirmation"]
    # clarification
    option_ids: list[str] = Field(default_factory=list, max_length=6)
    text: str | None = Field(default=None, max_length=2000)
    # authorization
    granted: bool | None = None
    duration: GrantDuration = GrantDuration.ONCE
    # confirmation (approval)
    approval_id: str | None = None
    approved: bool | None = None

    @classmethod
    def answer(cls, text: str | None = None, *, option_ids: list[str] | None = None) -> Resume:
        return cls(kind="clarification", text=text, option_ids=list(option_ids or []))

    @classmethod
    def authorize(cls, granted: bool = True, *, duration: GrantDuration | str = GrantDuration.ONCE) -> Resume:
        return cls(kind="authorization", granted=granted, duration=GrantDuration(duration))

    @classmethod
    def approve(cls, approval_id: str, approved: bool = True) -> Resume:
        return cls(kind="confirmation", approval_id=approval_id, approved=approved)

    @classmethod
    def reject(cls, approval_id: str) -> Resume:
        return cls.approve(approval_id, approved=False)


def tool_result(state: RunState, call_id: str | None, payload: dict) -> None:
    if not call_id:
        return
    state.transcript.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload, ensure_ascii=False)})


def _pending(state: RunState) -> PendingAction:
    if state.pending is None:
        raise InvalidResume("not_waiting", "This run is not waiting for anything.")
    return state.pending


def check_pending(state: RunState, kind: str, waiting_reason: str | None = None) -> PendingAction:
    pending = state.pending
    if pending is None or pending.reason.value != kind or (waiting_reason is not None and waiting_reason != kind):
        raise InvalidResume("waiting_for_other", "This run is waiting for something else.")
    return pending


def answer_clarification(state: RunState, *, option_ids: list[str], text: str | None, joiner: str = "; ") -> str:
    """Validate the answer, record it as the clarification's result; returns the shown text."""
    pending = _pending(state)
    clarification = pending.clarification or {}
    options = {option["id"]: option for option in clarification.get("options", [])}
    unknown = [option_id for option_id in option_ids if option_id not in options]
    if unknown:
        raise InvalidResume("clarification_option_invalid", "Unknown option")
    if len(option_ids) > 1 and not clarification.get("allow_multiple"):
        raise InvalidResume("clarification_single_choice", "Choose one option")
    labels = [options[option_id]["label"] for option_id in option_ids]
    free_text = (text or "").strip()
    if not labels and not free_text:
        raise InvalidResume("clarification_empty", "An answer is required")
    shown = joiner.join(labels + ([free_text] if free_text else []))
    state.user_texts.append(shown)
    tool_result(state, pending.tool_call_id, {"ok": True, "data": {"answer": shown, "selected_option_ids": option_ids}})
    return shown


def apply_authorization(state: RunState, *, granted: bool, denial_message: str) -> None:
    """Granted: run the waiting call (and the rest of its batch). Denied: tell the model."""
    pending = _pending(state)
    if granted:
        state.queued_calls = [
            {"id": pending.tool_call_id, "name": pending.tool_name, "arguments": pending.arguments}
        ] + list(pending.remaining_calls)
    else:
        tool_result(
            state,
            pending.tool_call_id,
            {"ok": False, "error": {"code": "user_denied_authorization", "message": denial_message}},
        )
        state.queued_calls = list(pending.remaining_calls)


def finish_confirmation(state: RunState, observation: dict) -> None:
    """The user decided on what a tool prepared: that verdict is the tool's result."""
    pending = _pending(state)
    tool_result(state, pending.tool_call_id, observation)
    state.queued_calls = list(pending.remaining_calls)


def apply_approval(state: RunState, *, approval_id: str, approved: bool, principal_id: str | None) -> None:
    """Approve or reject one pending call. Approval is bound to the exact arguments and used once."""
    pending = _pending(state)
    validate_approval(
        pending.approval,
        approval_id=approval_id,
        principal_id=principal_id,
        tool_call_id=pending.tool_call_id,
        arguments=pending.arguments,
    )
    pending.approval = {**(pending.approval or {}), "status": "approved" if approved else "rejected"}
    if approved:
        state.approved_calls[pending.tool_call_id or ""] = stable_hash(pending.arguments)
        state.queued_calls = [
            {"id": pending.tool_call_id, "name": pending.tool_name, "arguments": pending.arguments}
        ] + list(pending.remaining_calls)
    else:
        tool_result(
            state,
            pending.tool_call_id,
            {"ok": False, "error": {"code": "user_rejected", "message": "The user rejected this action. Do not retry it."}},
        )
        state.queued_calls = list(pending.remaining_calls)


def reopen(store: Any, run: Any, state: RunState) -> None:
    """Make a resumed run executable again: next segment, status `created`, lease dropped."""
    state.pending = None
    store.update_run(run, commit=False, segment=int(getattr(run, "segment", 0) or 0) + 1)
    # `created` = ready to execute. `running` is only ever set by the loop
    # holding the lease, which is how it tells a recovery from a resume.
    store.set_status(run, RunStatus.CREATED, commit=False)
    store.update_run(run, commit=False, worker_id=None, lease_expires_at=None, cancel_requested_at=None)
    store.save_checkpoint(run, state, commit=False)
    store.commit()


def resume_run(
    store: Any,
    run_id: Any,
    resume: Resume,
    *,
    principal_id: str | None,
    grants: GrantStore | None = None,
    state_class: type[RunState] = RunState,
    clarification_joiner: str = "; ",
) -> Any:
    """Apply ``resume`` to a suspended run in a generic store. Returns the run."""
    run = store.load_run(run_id)
    if run is None or (run.principal_id is not None and run.principal_id != principal_id):
        raise RunNotFound(str(run_id))
    if store.get_status(run) is not RunStatus.SUSPENDED:
        raise RunNotSuspended(str(run_id))
    state = store.load_state(run, state_class)
    pending = check_pending(state, resume.kind, run.waiting_reason)
    if resume.kind == WaitingReason.CLARIFICATION.value:
        answer_clarification(state, option_ids=resume.option_ids, text=resume.text, joiner=clarification_joiner)
    elif resume.kind == WaitingReason.AUTHORIZATION.value:
        if resume.granted is None:
            raise InvalidResume("authorization_decision_required", "granted must be true or false")
        if resume.granted:
            if grants is None or principal_id is None:
                raise InvalidResume("no_grant_store", "This agent cannot record grants.")
            grants.add(
                new_grant(
                    principal_id=principal_id,
                    capability=pending.resource or "",
                    duration=resume.duration,
                    run_id=run.id,
                    session_id=run.session_id,
                )
            )
        apply_authorization(state, granted=bool(resume.granted), denial_message="The user declined access. Answer without it.")
    else:
        if pending.approval is None:
            raise InvalidResume("not_an_approval", "This run waits for an application-specific confirmation.")
        if resume.approval_id is None or resume.approved is None:
            raise InvalidResume("approval_decision_required", "approval_id and approved are required")
        apply_approval(state, approval_id=resume.approval_id, approved=resume.approved, principal_id=principal_id)
    reopen(store, run, state)
    return run
