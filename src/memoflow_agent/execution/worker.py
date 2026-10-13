"""Background execution: runs outlive the request that started them.

A run submitted to a :class:`Worker` keeps going when the client goes away;
the client reconnects and replays from its last event. If the process dies,
the run's lease expires and a recovery sweep (in this or another process)
resubmits it; the loop resumes from its checkpoint, and side-effecting tool
calls are made idempotent by their key, so a recovered run never repeats one.

Deliberately not a message queue: the store already holds run state and
arbitrates leases; a broker would be a second source of truth.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("memoflow_agent.worker")


def make_worker_id() -> str:
    return f"{socket.gethostname()[:40]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class Worker:
    """Thread pool + lease-recovery sweep. Subclasses implement :meth:`drive`."""

    def __init__(
        self,
        *,
        threads: int = 8,
        recovery_interval: float = 15.0,
        recover: bool = True,
        worker_id: str | None = None,
    ) -> None:
        self.worker_id = worker_id or make_worker_id()
        self.threads = threads
        self.recovery_interval = recovery_interval
        self.recover = recover
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._active: set[Any] = set()
        self._stop = threading.Event()
        self._recovery: threading.Thread | None = None

    # --- subclass API ----------------------------------------------------------------

    def drive(self, run_id: Any, context: Any = None) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_idle(self, run_id: Any) -> None:
        """Called after a run's execution finished in this worker (for any reason)."""
        return None

    def recover_once(self) -> int:  # pragma: no cover - abstract
        """Resubmit runs whose lease expired; return how many were resubmitted."""
        raise NotImplementedError

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._executor is not None:
                return
            self._executor = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="agent-run")
            self._stop.clear()
            if self.recover:
                self._recovery = threading.Thread(target=self._recovery_loop, name="agent-recovery", daemon=True)
                self._recovery.start()

    def shutdown(self, *, wait: bool = False) -> None:
        self._stop.set()
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            # Running loops keep their leases until they expire; another
            # process (or this one after restart) recovers them.
            executor.shutdown(wait=wait, cancel_futures=True)

    def is_active(self, run_id: Any) -> bool:
        with self._lock:
            return run_id in self._active

    def submit(self, run_id: Any, context: Any = None) -> bool:
        if self._executor is None:
            self.start()
        with self._lock:
            if run_id in self._active or self._executor is None:
                return False
            self._active.add(run_id)
            executor = self._executor
        try:
            executor.submit(self._run, run_id, context)
        except RuntimeError:
            with self._lock:
                self._active.discard(run_id)
            return False
        return True

    def _run(self, run_id: Any, context: Any) -> None:
        try:
            self.drive(run_id, context)
        except Exception:
            logger.exception("agent run %s: worker crashed", run_id)
        finally:
            with self._lock:
                self._active.discard(run_id)
            self.on_idle(run_id)

    def _recovery_loop(self) -> None:
        interval = max(2.0, self.recovery_interval)
        while not self._stop.wait(interval):
            try:
                count = self.recover_once()
                if count:
                    logger.warning("recovered %d agent run(s) with expired leases", count)
            except Exception:
                logger.warning("agent recovery sweep failed", exc_info=True)


class StoreWorker(Worker):
    """A worker over a generic run store; ``run_factory(run_id, worker_id)`` returns the event iterator."""

    def __init__(
        self,
        store: Any,
        run_factory: Callable[[Any, str], Any],
        *,
        broker: Any = None,
        waiting_ttl_seconds: float = 72 * 3600,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.store = store
        self.run_factory = run_factory
        self.broker = broker
        self.waiting_ttl_seconds = waiting_ttl_seconds

    def drive(self, run_id: Any, context: Any = None) -> None:
        for event in self.run_factory(run_id, self.worker_id):
            if self.broker is not None:
                self.broker.publish(run_id, event)

    def on_idle(self, run_id: Any) -> None:
        if self.broker is not None:
            self.broker.publish(run_id, {"type": "_worker_idle", "run_id": run_id})

    def recover_once(self) -> int:
        from ..state import RunState, RunStatus, StopReason

        resubmitted = 0
        for run_id in self.store.expired_runs():
            if self.submit(run_id):
                resubmitted += 1
        for run in self.store.stale_suspended_runs(older_than_seconds=self.waiting_ttl_seconds):
            state = self.store.load_state(run, RunState)
            self.store.set_status(run, RunStatus.CANCELLED, stop_reason=StopReason.WAITING_EXPIRED.value, commit=False)
            self.store.clear_working_state(run, state)
            self.store.commit()
        return resubmitted
