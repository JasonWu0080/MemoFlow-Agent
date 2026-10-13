"""In-memory run store: the zero-setup default.

Everything lives in this process. Runs survive suspension and resume within
the process; **nothing survives a restart** and there is no cross-process
recovery. Use :class:`~memoflow_agent.persistence.postgres.PostgresStore`
for durable execution.

Writes apply immediately (``commit`` arguments are accepted and ignored);
every operation is atomic under one lock.
"""

from __future__ import annotations

import copy
import threading
import uuid
from datetime import timedelta
from typing import Any

from ..events import EPHEMERAL_EVENT_TYPES, envelope
from ..exceptions import LeaseLost
from ..state import EXECUTABLE_STATUSES, TERMINAL_STATUSES, RunState, RunStatus, check_transition
from ..util import utcnow
from .base import RunRecord, checkpoint_envelope


class _Sink:
    def __init__(self, store: InMemoryStore, run: RunRecord):
        self.store = store
        self.run = run
        with store._lock:
            self.seq = len(store._events.get(run.id, []))

    def emit(self, event_type: str, payload: dict, *, commit: bool = True) -> dict:
        if str(event_type) in EPHEMERAL_EVENT_TYPES:
            return envelope(self.run.id, self.seq, event_type, payload)
        with self.store._lock:
            events = self.store._events.setdefault(self.run.id, [])
            self.seq = len(events) + 1
            event = envelope(self.run.id, self.seq, event_type, payload)
            events.append(copy.deepcopy(event))
        return event


class InMemoryStore:
    def __init__(self, *, lease_seconds: float = 45.0) -> None:
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[dict]] = {}
        self._steps: dict[str, list[dict]] = {}
        self._tool_records: dict[str, dict] = {}

    # --- runs -----------------------------------------------------------------------

    def create_run(self, *, state: RunState, principal_id=None, session_id=None, idempotency_key=None, metadata=None) -> RunRecord:
        now = utcnow()
        run = RunRecord(
            id=uuid.uuid4().hex,
            principal_id=principal_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._runs[run.id] = run
            self.save_checkpoint(run, state)
        return run

    def load_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(str(run_id))

    def find_by_idempotency_key(self, principal_id, key: str) -> RunRecord | None:
        with self._lock:
            for run in self._runs.values():
                if run.idempotency_key == key and run.principal_id == principal_id:
                    return run
        return None

    def get_status(self, run: RunRecord) -> RunStatus:
        return run.status

    def set_status(self, run: RunRecord, status: RunStatus, *, waiting_reason=None, stop_reason=None,
                   error_code=None, error_message=None, commit: bool = True) -> None:
        with self._lock:
            check_transition(run.status, status)
            now = utcnow()
            run.status = status
            run.version += 1
            run.updated_at = now
            run.waiting_reason = waiting_reason if status is RunStatus.SUSPENDED else None
            if status is RunStatus.SUSPENDED:
                run.paused_at = now
            if stop_reason is not None:
                run.stop_reason = stop_reason
            if error_code is not None:
                run.error_code = error_code[:64]
            if error_message is not None:
                run.error_message = error_message[:2000]
            if status in TERMINAL_STATUSES:
                run.completed_at = now
                run.worker_id = None
                run.lease_expires_at = None

    def update_run(self, run: RunRecord, *, commit: bool = True, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(run, key, value)
            run.updated_at = utcnow()

    def request_cancel(self, run: RunRecord) -> None:
        with self._lock:
            if run.cancel_requested_at is None:
                run.cancel_requested_at = utcnow()

    def cancel_requested(self, run_id) -> bool:
        run = self.load_run(run_id)
        return run is not None and run.cancel_requested_at is not None

    # --- leases -----------------------------------------------------------------------

    def acquire_lease(self, run_id, worker_id: str) -> bool:
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is None or run.status not in EXECUTABLE_STATUSES:
                return False
            now = utcnow()
            free = run.worker_id in (None, worker_id) or run.lease_expires_at is None or run.lease_expires_at < now
            if not free:
                return False
            run.worker_id = worker_id
            run.lease_expires_at = now + timedelta(seconds=self.lease_seconds)
            run.version += 1
            return True

    def renew_lease(self, run: RunRecord, worker_id: str) -> None:
        with self._lock:
            if run.worker_id != worker_id:
                raise LeaseLost(f"run {run.id}")
            run.lease_expires_at = utcnow() + timedelta(seconds=self.lease_seconds)

    def release_lease(self, run_id, worker_id: str) -> None:
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is not None and run.worker_id == worker_id:
                run.worker_id = None
                run.lease_expires_at = None

    def expired_runs(self, *, limit: int = 20) -> list[str]:
        now = utcnow()
        grace = timedelta(seconds=self.lease_seconds)
        with self._lock:
            return [
                run.id
                for run in self._runs.values()
                if run.status in EXECUTABLE_STATUSES
                and (run.lease_expires_at is None or run.lease_expires_at < now)
                and run.created_at is not None
                and run.created_at < now - grace
            ][:limit]

    def stale_suspended_runs(self, *, older_than_seconds: float, limit: int = 100) -> list[RunRecord]:
        cutoff = utcnow() - timedelta(seconds=older_than_seconds)
        with self._lock:
            return [
                run for run in self._runs.values()
                if run.status is RunStatus.SUSPENDED and run.paused_at is not None and run.paused_at < cutoff
            ][:limit]

    # --- checkpoints -------------------------------------------------------------------

    def load_state(self, run: RunRecord, state_class: type[RunState] = RunState) -> RunState:
        with self._lock:
            checkpoint = copy.deepcopy(run.checkpoint)
        return state_class.restore((checkpoint or {}).get("state"))

    def save_checkpoint(self, run: RunRecord, state: RunState, *, step_id=None, manifest=None, commit: bool = True) -> None:
        with self._lock:
            sequence = len(self._events.get(run.id, []))
            document = checkpoint_envelope(run, state, event_sequence=sequence, step_id=step_id)
            if manifest is not None:
                document["context_manifest"] = manifest
            run.checkpoint = document

    def clear_working_state(self, run: RunRecord, state: RunState) -> None:
        state.transcript = []
        state.pending = None
        self.save_checkpoint(run, state)

    # --- steps --------------------------------------------------------------------------

    def begin_step(self, run: RunRecord, step_type: str, *, name: str | None, budget_snapshot: dict | None = None) -> dict:
        with self._lock:
            steps = self._steps.setdefault(run.id, [])
            step = {
                "run_id": run.id,
                "sequence": len(steps) + 1,
                "type": step_type,
                "name": name,
                "status": "running",
                "started_at": utcnow().isoformat(),
                "budget_snapshot": budget_snapshot,
            }
            steps.append(step)
            return step

    def finish_step(self, step: dict, *, status: str, duration_ms=None, token_usage=None, error_code=None, commit: bool = True) -> None:
        with self._lock:
            step.update(status=status, completed_at=utcnow().isoformat())
            if duration_ms is not None:
                step["duration_ms"] = duration_ms
            if token_usage is not None:
                step["token_usage"] = token_usage
            if error_code is not None:
                step["error_code"] = error_code

    def steps(self, run_id) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._steps.get(str(run_id), []))

    # --- events ---------------------------------------------------------------------------

    def event_sink(self, run: RunRecord) -> _Sink:
        return _Sink(self, run)

    def events_after(self, run_id, after_seq: int, *, limit: int = 500) -> list[dict]:
        with self._lock:
            return copy.deepcopy([event for event in self._events.get(str(run_id), []) if event["seq"] > after_seq][:limit])

    # --- tool idempotency -----------------------------------------------------------------

    def tool_record(self, key: str) -> dict | None:
        with self._lock:
            record = self._tool_records.get(key)
            return copy.deepcopy(record) if record is not None else None

    def save_tool_record(self, key: str, record: dict) -> None:
        with self._lock:
            self._tool_records[key] = copy.deepcopy(record)

    # --- unit of work ----------------------------------------------------------------------

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None
