"""PostgreSQL run store (optional: ``pip install "memoflow-agent[postgres]"``).

Durable runs, checkpoints, leases, events and idempotency records in the
generic schema of ``schema.sql``. Semantics are those documented in
:mod:`memoflow_agent.persistence.base`:

* ``acquire_lease`` is one conditional ``UPDATE … WHERE lease expired OR
  mine`` — exactly one of several racing workers wins;
* event ``seq`` values are assigned by the lease holder and protected by the
  ``(run_id, seq)`` primary key;
* methods called with ``commit=False`` stay in the open transaction until
  :meth:`commit`, so a status change and the event describing it become
  durable together.

One store instance owns one connection and serializes access with a lock.
Give each worker thread its own instance (``PostgresStore(dsn)``) when runs
should execute in parallel.
"""

from __future__ import annotations

import threading
import uuid
from importlib import resources
from typing import Any

from ..events import EPHEMERAL_EVENT_TYPES, envelope
from ..exceptions import ConfigurationError, LeaseLost
from ..policy import Grant, GrantDuration
from ..state import EXECUTABLE_STATUSES, TERMINAL_STATUSES, RunState, RunStatus, check_transition
from ..util import utcnow
from .base import RunRecord, checkpoint_envelope

try:  # pragma: no cover - import guard
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

_RUN_COLUMNS = (
    "id, status, segment, principal_id, session_id, idempotency_key, worker_id, lease_expires_at, "
    "cancel_requested_at, waiting_reason, stop_reason, error_code, error_message, output, version, metadata, "
    "checkpoint, created_at, updated_at, paused_at, completed_at"
)
_UPDATABLE = {
    "segment", "worker_id", "lease_expires_at", "cancel_requested_at", "output", "metadata", "session_id",
}


def _record(row: dict) -> RunRecord:
    data = dict(row)
    data["status"] = RunStatus(data["status"])
    data["metadata"] = data.get("metadata") or {}
    return RunRecord(**data)


class _Sink:
    def __init__(self, store: PostgresStore, run: RunRecord):
        self.store = store
        self.run = run
        with store._lock:
            self.seq = int(store._scalar("SELECT COALESCE(MAX(seq), 0) AS seq FROM memoflow_agent_events WHERE run_id = %s", (run.id,)))

    def emit(self, event_type: str, payload: dict, *, commit: bool = True) -> dict:
        if str(event_type) in EPHEMERAL_EVENT_TYPES:
            return envelope(self.run.id, self.seq, event_type, payload)
        with self.store._lock:
            self.seq += 1
            event = envelope(self.run.id, self.seq, event_type, payload)
            self.store._exec(
                "INSERT INTO memoflow_agent_events (run_id, seq, type, envelope) VALUES (%s, %s, %s, %s)",
                (self.run.id, self.seq, str(event_type), Jsonb(event)),
            )
            if commit:
                self.store._conn.commit()
        return event


class PostgresStore:
    def __init__(self, dsn: str | None = None, *, connection: Any = None, lease_seconds: float = 45.0) -> None:
        if psycopg is None:
            raise ConfigurationError('PostgresStore needs psycopg: pip install "memoflow-agent[postgres]"')
        if connection is None and not dsn:
            raise ConfigurationError("PostgresStore needs a dsn or a connection")
        self._conn = connection or psycopg.connect(str(dsn), row_factory=dict_row)
        self._conn.row_factory = dict_row
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()

    # --- plumbing ------------------------------------------------------------------

    def _exec(self, sql: str, params: tuple = ()) -> Any:
        return self._conn.execute(sql, params)

    def _one(self, sql: str, params: tuple = ()) -> dict | None:
        return self._exec(sql, params).fetchone()

    def _scalar(self, sql: str, params: tuple = ()) -> Any:
        row = self._one(sql, params)
        assert row is not None
        return next(iter(row.values()))

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        return list(self._exec(sql, params).fetchall())

    def create_schema(self) -> None:
        sql = resources.files("memoflow_agent.persistence").joinpath("schema.sql").read_text(encoding="utf-8")
        with self._lock:
            self._exec(sql)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def _done(self, commit: bool) -> None:
        if commit:
            self._conn.commit()

    # --- runs ------------------------------------------------------------------------

    def create_run(self, *, state: RunState, principal_id=None, session_id=None, idempotency_key=None, metadata=None) -> RunRecord:
        run_id = uuid.uuid4().hex
        with self._lock:
            row = self._one(
                f"INSERT INTO memoflow_agent_runs (id, status, principal_id, session_id, idempotency_key, metadata) "
                f"VALUES (%s, %s, %s, %s, %s, %s) RETURNING {_RUN_COLUMNS}",
                (run_id, RunStatus.CREATED.value, principal_id, session_id, idempotency_key, Jsonb(dict(metadata or {}))),
            )
            assert row is not None
            run = _record(row)
            self.save_checkpoint(run, state, commit=False)
            self._conn.commit()
        return run

    def load_run(self, run_id) -> RunRecord | None:
        with self._lock:
            row = self._one(f"SELECT {_RUN_COLUMNS} FROM memoflow_agent_runs WHERE id = %s", (str(run_id),))
            self._conn.commit()
        return _record(row) if row else None

    def find_by_idempotency_key(self, principal_id, key: str) -> RunRecord | None:
        with self._lock:
            row = self._one(
                f"SELECT {_RUN_COLUMNS} FROM memoflow_agent_runs WHERE idempotency_key = %s AND principal_id IS NOT DISTINCT FROM %s",
                (key, principal_id),
            )
            self._conn.commit()
        return _record(row) if row else None

    def get_status(self, run: RunRecord) -> RunStatus:
        return run.status

    def set_status(self, run: RunRecord, status: RunStatus, *, waiting_reason=None, stop_reason=None,
                   error_code=None, error_message=None, commit: bool = True) -> None:
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
        with self._lock:
            # The lease columns belong to acquire/renew/release; a status
            # change only clears them when the run ends.
            terminal = status in TERMINAL_STATUSES
            self._exec(
                "UPDATE memoflow_agent_runs SET status=%s, version=version+1, updated_at=%s, waiting_reason=%s, paused_at=%s, "
                "stop_reason=%s, error_code=%s, error_message=%s, completed_at=%s, "
                "worker_id = CASE WHEN %s THEN NULL ELSE worker_id END, "
                "lease_expires_at = CASE WHEN %s THEN NULL ELSE lease_expires_at END WHERE id=%s",
                (status.value, now, run.waiting_reason, run.paused_at, run.stop_reason, run.error_code, run.error_message,
                 run.completed_at, terminal, terminal, run.id),
            )
            self._done(commit)

    def update_run(self, run: RunRecord, *, commit: bool = True, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        for key, value in fields.items():
            setattr(run, key, value)
        if not fields:
            return
        assignments = ", ".join(f"{key} = %s" for key in fields)
        values = tuple(Jsonb(value) if key == "metadata" else value for key, value in fields.items())
        with self._lock:
            self._exec(f"UPDATE memoflow_agent_runs SET {assignments}, updated_at = now() WHERE id = %s", (*values, run.id))
            self._done(commit)

    def request_cancel(self, run: RunRecord) -> None:
        with self._lock:
            self._exec("UPDATE memoflow_agent_runs SET cancel_requested_at = COALESCE(cancel_requested_at, now()) WHERE id = %s", (run.id,))
            self._conn.commit()
        run.cancel_requested_at = run.cancel_requested_at or utcnow()

    def cancel_requested(self, run_id) -> bool:
        with self._lock:
            row = self._one("SELECT cancel_requested_at FROM memoflow_agent_runs WHERE id = %s", (str(run_id),))
        return bool(row and row["cancel_requested_at"] is not None)

    # --- leases ----------------------------------------------------------------------------

    def acquire_lease(self, run_id, worker_id: str) -> bool:
        with self._lock:
            row = self._one(
                "UPDATE memoflow_agent_runs SET worker_id = %s, lease_expires_at = now() + make_interval(secs => %s), "
                "version = version + 1 WHERE id = %s AND status = ANY(%s) AND (worker_id IS NULL OR worker_id = %s "
                "OR lease_expires_at IS NULL OR lease_expires_at < now()) RETURNING id",
                (worker_id, self.lease_seconds, str(run_id), [s.value for s in EXECUTABLE_STATUSES], worker_id),
            )
            self._conn.commit()
        return row is not None

    def renew_lease(self, run: RunRecord, worker_id: str) -> None:
        with self._lock:
            row = self._one(
                "UPDATE memoflow_agent_runs SET lease_expires_at = now() + make_interval(secs => %s) "
                "WHERE id = %s AND worker_id = %s RETURNING id",
                (self.lease_seconds, run.id, worker_id),
            )
            self._conn.commit()
        if row is None:
            raise LeaseLost(f"run {run.id}")

    def release_lease(self, run_id, worker_id: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE memoflow_agent_runs SET worker_id = NULL, lease_expires_at = NULL WHERE id = %s AND worker_id = %s",
                (str(run_id), worker_id),
            )
            self._conn.commit()

    def expired_runs(self, *, limit: int = 20) -> list[str]:
        with self._lock:
            rows = self._all(
                "SELECT id FROM memoflow_agent_runs WHERE status = ANY(%s) AND (lease_expires_at IS NULL OR lease_expires_at < now()) "
                "AND created_at < now() - make_interval(secs => %s) ORDER BY created_at LIMIT %s",
                ([s.value for s in EXECUTABLE_STATUSES], self.lease_seconds, limit),
            )
            self._conn.commit()
        return [row["id"] for row in rows]

    def stale_suspended_runs(self, *, older_than_seconds: float, limit: int = 100) -> list[RunRecord]:
        with self._lock:
            rows = self._all(
                f"SELECT {_RUN_COLUMNS} FROM memoflow_agent_runs WHERE status = %s AND paused_at < now() - make_interval(secs => %s) LIMIT %s",
                (RunStatus.SUSPENDED.value, older_than_seconds, limit),
            )
            self._conn.commit()
        return [_record(row) for row in rows]

    # --- checkpoints ---------------------------------------------------------------------------

    def load_state(self, run: RunRecord, state_class: type[RunState] = RunState) -> RunState:
        with self._lock:
            row = self._one("SELECT checkpoint FROM memoflow_agent_runs WHERE id = %s", (run.id,))
            self._conn.commit()
        checkpoint = (row or {}).get("checkpoint") or {}
        return state_class.restore(checkpoint.get("state"))

    def save_checkpoint(self, run: RunRecord, state: RunState, *, step_id=None, manifest=None, commit: bool = True) -> None:
        with self._lock:
            sequence = int(self._scalar("SELECT COALESCE(MAX(seq), 0) AS seq FROM memoflow_agent_events WHERE run_id = %s", (run.id,)))
            document = checkpoint_envelope(run, state, event_sequence=sequence, step_id=step_id)
            if manifest is not None:
                document["context_manifest"] = manifest
            run.checkpoint = document
            self._exec("UPDATE memoflow_agent_runs SET checkpoint = %s, updated_at = now() WHERE id = %s", (Jsonb(document), run.id))
            self._done(commit)

    def clear_working_state(self, run: RunRecord, state: RunState) -> None:
        state.transcript = []
        state.pending = None
        self.save_checkpoint(run, state, commit=False)

    # --- steps ---------------------------------------------------------------------------------

    def begin_step(self, run: RunRecord, step_type: str, *, name: str | None, budget_snapshot: dict | None = None) -> dict:
        with self._lock:
            row = self._one(
                "INSERT INTO memoflow_agent_steps (run_id, sequence, type, name, status, budget_snapshot) "
                "SELECT %s, COALESCE(MAX(sequence), 0) + 1, %s, %s, 'running', %s FROM memoflow_agent_steps WHERE run_id = %s "
                "RETURNING id, sequence",
                (run.id, step_type, name, Jsonb(budget_snapshot) if budget_snapshot is not None else None, run.id),
            )
            self._conn.commit()
        assert row is not None
        return {"id": row["id"], "sequence": row["sequence"], "run_id": run.id}

    def finish_step(self, step: dict, *, status: str, duration_ms=None, token_usage=None, error_code=None, commit: bool = True) -> None:
        with self._lock:
            self._exec(
                "UPDATE memoflow_agent_steps SET status=%s, completed_at=now(), duration_ms=COALESCE(%s, duration_ms), "
                "token_usage=COALESCE(%s, token_usage), error_code=COALESCE(%s, error_code) WHERE id=%s",
                (status, duration_ms, Jsonb(token_usage) if token_usage is not None else None, error_code, step["id"]),
            )
            self._done(commit)

    def steps(self, run_id) -> list[dict]:
        with self._lock:
            rows = self._all("SELECT sequence, type, name, status, error_code, token_usage FROM memoflow_agent_steps WHERE run_id=%s ORDER BY sequence", (str(run_id),))
            self._conn.commit()
        return rows

    # --- events --------------------------------------------------------------------------------

    def event_sink(self, run: RunRecord) -> _Sink:
        return _Sink(self, run)

    def events_after(self, run_id, after_seq: int, *, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._all(
                "SELECT envelope FROM memoflow_agent_events WHERE run_id = %s AND seq > %s ORDER BY seq LIMIT %s",
                (str(run_id), after_seq, limit),
            )
            self._conn.commit()
        return [row["envelope"] for row in rows]

    # --- tool idempotency ------------------------------------------------------------------------

    def tool_record(self, key: str) -> dict | None:
        with self._lock:
            row = self._one("SELECT record FROM memoflow_agent_tool_records WHERE key = %s", (key,))
        return row["record"] if row else None

    def save_tool_record(self, key: str, record: dict) -> None:
        # Committed at once: the record must be durable *before* the side
        # effect runs, or a crash could hide that it started.
        with self._lock:
            self._exec(
                "INSERT INTO memoflow_agent_tool_records (key, record) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET record = EXCLUDED.record, updated_at = now()",
                (key, Jsonb(record)),
            )
            self._conn.commit()


class PostgresGrantStore:
    """Grants in ``memoflow_agent_grants``; re-read on every policy evaluation."""

    def __init__(self, store: PostgresStore) -> None:
        self.store = store

    def add(self, grant: Grant) -> Grant:
        with self.store._lock:
            self.store._exec(
                "INSERT INTO memoflow_agent_grants (id, principal_id, capability, duration, run_id, session_id, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (grant.id, grant.principal_id, grant.capability, grant.duration.value,
                 None if grant.run_id is None else str(grant.run_id),
                 None if grant.session_id is None else str(grant.session_id), grant.expires_at),
            )
            self.store._conn.commit()
        return grant

    def revoke(self, grant_id: str) -> bool:
        with self.store._lock:
            row = self.store._one(
                "UPDATE memoflow_agent_grants SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL RETURNING id", (grant_id,)
            )
            self.store._conn.commit()
        return row is not None

    def active(self, *, principal_id: str, capability: str, run_id: Any, session_id: Any) -> bool:
        with self.store._lock:
            row = self.store._one(
                "SELECT 1 FROM memoflow_agent_grants WHERE principal_id = %s AND capability = %s AND revoked_at IS NULL "
                "AND (expires_at IS NULL OR expires_at > now()) AND ("
                "duration = 'persistent' OR (duration = 'session' AND session_id = %s) OR (duration = 'once' AND run_id = %s)) LIMIT 1",
                (principal_id, capability, None if session_id is None else str(session_id), None if run_id is None else str(run_id)),
            )
            self.store._conn.commit()
        return row is not None

    def list(self, principal_id: str) -> list[Grant]:
        with self.store._lock:
            rows = self.store._all("SELECT * FROM memoflow_agent_grants WHERE principal_id = %s ORDER BY created_at", (principal_id,))
            self.store._conn.commit()
        return [
            Grant(
                id=row["id"], principal_id=row["principal_id"], capability=row["capability"], duration=GrantDuration(row["duration"]),
                run_id=row["run_id"], session_id=row["session_id"], expires_at=row["expires_at"], revoked_at=row["revoked_at"],
            )
            for row in rows
        ]
