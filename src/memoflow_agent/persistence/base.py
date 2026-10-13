"""The run store contract.

A store is the source of truth for runs. Semantics every implementation
must provide:

* **Lease ownership** — a run is executed only by the holder of its lease.
  ``acquire_lease`` is a single atomic compare-and-set ("lease expired or
  mine"), so two workers racing for a run cannot both win; a dead worker's
  lease simply expires. ``renew_lease`` raises :class:`LeaseLost` when the
  lease was taken over, and the loop stops without another write.
* **Event ordering** — persistent events get a per-run ``seq`` that only the
  lease holder assigns, strictly increasing, stored before delivery.
* **Checkpoints** — one current checkpoint per run, rewritten in place, with
  ``schema_version``; an incompatible one is refused, never half-loaded.
* **Unit of work** — methods taking ``commit=False`` stage their write so
  the loop can make a status change and the event describing it durable
  together; ``commit()`` makes staged writes durable, ``rollback()`` drops
  them. Stores without transactions (in-memory) apply writes immediately.
* **Idempotency** — tool calls with non-repeatable side effects are recorded
  under an idempotency key; a recovered run finds the record and replays
  the result instead of running the tool again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..state import RunState, RunStatus


@dataclass
class RunRecord:
    id: str
    status: RunStatus = RunStatus.CREATED
    segment: int = 0
    principal_id: str | None = None
    session_id: str | None = None
    idempotency_key: str | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    waiting_reason: str | None = None
    stop_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    output: str | None = None
    version: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    paused_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] | None = None


class RunStore(Protocol):
    # runs
    def create_run(self, *, state: RunState, principal_id: str | None = None, session_id: str | None = None,
                   idempotency_key: str | None = None, metadata: dict | None = None) -> RunRecord: ...
    def load_run(self, run_id: str) -> RunRecord | None: ...
    def find_by_idempotency_key(self, principal_id: str | None, key: str) -> RunRecord | None: ...
    def get_status(self, run: Any) -> RunStatus: ...
    def set_status(self, run: Any, status: RunStatus, *, waiting_reason: str | None = None,
                   stop_reason: str | None = None, error_code: str | None = None,
                   error_message: str | None = None, commit: bool = True) -> None: ...
    def update_run(self, run: Any, *, commit: bool = True, **fields: Any) -> None: ...
    def request_cancel(self, run: Any) -> None: ...
    def cancel_requested(self, run_id: Any) -> bool: ...
    # leases
    def acquire_lease(self, run_id: Any, worker_id: str) -> bool: ...
    def renew_lease(self, run: Any, worker_id: str) -> None: ...
    def release_lease(self, run_id: Any, worker_id: str) -> None: ...
    def expired_runs(self, *, limit: int = 20) -> list[Any]: ...
    def stale_suspended_runs(self, *, older_than_seconds: float, limit: int = 100) -> list[Any]: ...
    # checkpoints
    def load_state(self, run: Any, state_class: type[RunState] = RunState) -> RunState: ...
    def save_checkpoint(self, run: Any, state: RunState, *, step_id: Any = None,
                        manifest: dict | None = None, commit: bool = True) -> None: ...
    def clear_working_state(self, run: Any, state: RunState) -> None: ...
    # steps (trace)
    def begin_step(self, run: Any, step_type: str, *, name: str | None, budget_snapshot: dict | None = None) -> Any: ...
    def finish_step(self, step: Any, *, status: str, duration_ms: int | None = None, token_usage: dict | None = None,
                    error_code: str | None = None, commit: bool = True) -> None: ...
    # events
    def event_sink(self, run: Any) -> Any: ...
    def events_after(self, run_id: Any, after_seq: int, *, limit: int = 500) -> list[dict]: ...
    # tool idempotency
    def tool_record(self, key: str) -> dict | None: ...
    def save_tool_record(self, key: str, record: dict) -> None: ...
    # unit of work
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


def checkpoint_envelope(run: RunRecord, state: RunState, *, event_sequence: int, step_id: Any = None) -> dict:
    """The generic checkpoint document (schema_version + state + positions)."""
    pending = state.pending
    return {
        "schema_version": state.version,
        "run_id": run.id,
        "run_state": str(run.status),
        "segment": run.segment,
        "model_turn": state.model_turns,
        "event_sequence": event_sequence,
        "step_id": step_id,
        "suspension_state": pending.model_dump(mode="json") if pending is not None else None,
        "pending_tool_calls": list(state.queued_calls),
        "budget_state": {
            "model_turns": state.model_turns,
            "tool_calls": state.tool_calls,
            "tokens_used": state.tokens_used,
            "input_tokens_used": state.input_tokens_used,
            "output_tokens_used": state.output_tokens_used,
            "consecutive_failures": state.consecutive_failures,
        },
        "state": state.model_dump(mode="json"),
    }
