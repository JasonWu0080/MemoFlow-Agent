"""Framework events, their envelope, and an in-process fan-out.

Every event a run emits has the same envelope::

    {"id": "<run_id>:<seq>", "run_id": "r1", "seq": 7, "type": "tool.started",
     "ts": "2026-10-12T08:00:00.123Z", "payload": {...}}

Persistent events get a per-run, monotonically increasing ``seq`` assigned by
the lease holder and are stored before they are delivered, so a run's history
can be replayed from any cursor. Ephemeral events (``model.delta``) are *not*
stored: they carry the ``seq`` of the last persisted event and a character
``offset``. A reconnecting client gets the text so far as one snapshot — the
framework never promises token-by-token replay of deltas.

Applications that already have a wire protocol map these names onto theirs
with an :class:`EventMapper` (MemoFlow maps them onto its V4 protocol).
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    RUN_RESUMED = "run.resumed"
    RUN_SUSPENDED = "run.suspended"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    MODEL_STARTED = "model.started"
    MODEL_DELTA = "model.delta"
    # Text streamed during a turn that turned out to be a tool-calling turn
    # (or that was abandoned): clients discard it, it is not the answer.
    MODEL_RESET = "model.reset"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    CLARIFICATION_REQUIRED = "clarification.required"
    AUTHORIZATION_REQUIRED = "authorization.required"
    APPROVAL_REQUIRED = "approval.required"
    USAGE_UPDATED = "usage.updated"


EPHEMERAL_EVENT_TYPES = frozenset({EventType.MODEL_DELTA.value})
# After one of these the run produces nothing until someone acts.
QUIESCENT_EVENT_TYPES = frozenset(
    {
        EventType.RUN_SUSPENDED.value,
        EventType.RUN_COMPLETED.value,
        EventType.RUN_FAILED.value,
        EventType.RUN_CANCELLED.value,
    }
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def envelope(run_id: Any, seq: int, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{run_id}:{seq}",
        "run_id": run_id,
        "seq": seq,
        "type": str(event_type),
        "ts": now_iso(),
        "payload": payload,
    }


def parse_last_event_id(value: str | None, run_id: Any = None) -> int:
    """``Last-Event-ID`` → the sequence to resume after (0 when unusable)."""
    if not value:
        return 0
    head, _, tail = value.strip().rpartition(":") if ":" in value else ("", "", value.strip())
    try:
        if head:
            if run_id is not None and head != str(run_id):
                return 0
            return max(0, int(tail))
        return max(0, int(tail))
    except ValueError:
        return 0


@dataclass(frozen=True)
class Event:
    """A delivered event, as the public :class:`~memoflow_agent.Agent` API yields it."""

    id: str
    run_id: Any
    seq: int
    type: str
    ts: str
    payload: dict[str, Any]

    @classmethod
    def from_envelope(cls, data: Mapping[str, Any]) -> Event:
        return cls(
            id=str(data.get("id")),
            run_id=data.get("run_id"),
            seq=int(data.get("seq") or 0),
            type=str(data.get("type")),
            ts=str(data.get("ts")),
            payload=dict(data.get("payload") or {}),
        )

    @property
    def persistent(self) -> bool:
        return self.type not in EPHEMERAL_EVENT_TYPES

    @property
    def text(self) -> str:
        """The text of a ``model.delta`` event ('' for other events)."""
        return str(self.payload.get("text") or "") if self.type == EventType.MODEL_DELTA else ""


class EventSink(Protocol):
    """Where the loop sends events. Returns the envelope, or None if suppressed."""

    def emit(self, event_type: str, payload: dict[str, Any], *, commit: bool = True) -> dict[str, Any] | None: ...


class EventMapper:
    """Renames kernel events onto an application protocol.

    ``names`` maps a kernel type to the application's type, or to ``None`` to
    suppress it. Types not in the map (application-defined events) pass
    through unchanged.
    """

    def __init__(self, sink: EventSink, names: Mapping[str, str | None]):
        self.sink = sink
        self.names = dict(names)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sink, name)

    def emit(self, event_type: str, payload: dict[str, Any], *, commit: bool = True) -> dict[str, Any] | None:
        mapped = self.names.get(str(event_type), str(event_type))
        if mapped is None:
            return None
        return self.sink.emit(mapped, payload, commit=commit)


class EventBroker:
    """Same-process fan-out from the executing worker to live subscribers.

    Subscribers get a bounded queue; a subscriber that stops reading loses
    events rather than holding memory — it replays from the store when it
    reconnects. The store, not this broker, is the source of truth.
    """

    _QUEUE_SIZE = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[Any, list[queue.Queue]] = defaultdict(list)

    def subscribe(self, run_id: Any) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        with self._lock:
            self._subscribers[run_id].append(subscriber)
        return subscriber

    def unsubscribe(self, run_id: Any, subscriber: queue.Queue) -> None:
        with self._lock:
            listeners = self._subscribers.get(run_id)
            if not listeners:
                return
            try:
                listeners.remove(subscriber)
            except ValueError:
                pass
            if not listeners:
                self._subscribers.pop(run_id, None)

    def publish(self, run_id: Any, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._subscribers.get(run_id, ()))
        for subscriber in listeners:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass

    def has_subscribers(self, run_id: Any) -> bool:
        with self._lock:
            return bool(self._subscribers.get(run_id))
