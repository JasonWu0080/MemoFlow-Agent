"""Stores, leases, checkpoints and crash recovery — in memory and on PostgreSQL."""

from __future__ import annotations

import time

import pytest

from memoflow_agent import Agent, InMemoryStore, Principal, Resume, RunStatus, tool
from memoflow_agent.exceptions import LeaseLost, RunNotSuspended
from memoflow_agent.execution import StoreWorker
from memoflow_agent.state import InvalidTransition, RunState
from memoflow_agent.testing import ScriptedModel, calls, text


@pytest.fixture(params=["memory", "postgres"])
def store(request):
    if request.param == "memory":
        store = InMemoryStore(lease_seconds=1.0)
        store.connect_another = lambda: store  # type: ignore[attr-defined]
        return store
    return request.getfixturevalue("postgres_store")


def test_runs_checkpoints_and_events_round_trip(store):
    state = RunState(user_texts=["hello"])
    run = store.create_run(state=state, principal_id="alice", idempotency_key="k1")
    assert store.find_by_idempotency_key("alice", "k1").id == run.id
    assert store.find_by_idempotency_key("bob", "k1") is None
    loaded = store.load_run(run.id)
    assert loaded.status is RunStatus.CREATED
    assert store.load_state(loaded).user_texts == ["hello"]
    sink = store.event_sink(loaded)
    first = sink.emit("run.started", {"a": 1})
    delta = sink.emit("model.delta", {"text": "x"})
    second = sink.emit("tool.started", {"b": 2})
    assert (first["seq"], delta["seq"], second["seq"]) == (1, 1, 2)
    store.commit()
    assert [event["seq"] for event in store.events_after(run.id, 0)] == [1, 2]
    assert [event["type"] for event in store.events_after(run.id, 1)] == ["tool.started"]


def test_only_one_worker_holds_a_lease_and_an_expired_lease_moves(store):
    run = store.create_run(state=RunState())
    other = store.connect_another()
    assert store.acquire_lease(run.id, "w1")
    assert not other.acquire_lease(run.id, "w2")
    time.sleep(1.1)
    assert other.acquire_lease(run.id, "w2")
    with pytest.raises(LeaseLost):
        store.renew_lease(store.load_run(run.id), "w1")


def test_illegal_transitions_are_refused(store):
    run = store.create_run(state=RunState())
    store.set_status(run, RunStatus.RUNNING)
    store.set_status(run, RunStatus.COMPLETED)
    with pytest.raises(InvalidTransition):
        store.set_status(run, RunStatus.RUNNING)


def test_suspended_runs_survive_a_new_agent_instance(store):
    """Durable resume: the run is resumed by a different Agent (≈ another process)."""
    model = ScriptedModel([calls(("request_clarification", {"question": "Which city?"}))])
    first = Agent(model, store=store, principal=Principal("alice")).run_sync("weather")
    assert first.suspended
    later = Agent(ScriptedModel([text("Oslo it is.")]), store=store.connect_another(), principal=Principal("alice"))
    done = later.resume_sync(first.run_id, Resume.answer("Oslo"))
    assert done.completed and done.output == "Oslo it is."
    with pytest.raises(RunNotSuspended):
        later.resume_sync(first.run_id, Resume.answer("again"))


def test_a_crashed_run_is_recovered_from_its_checkpoint_without_repeating_side_effects(store):
    effects = []

    @tool(side_effect="non_idempotent")
    def charge(amount: int) -> dict:
        "Charge a card."
        effects.append(amount)
        return {"charged": amount}

    model = ScriptedModel([calls(("charge", {"amount": 5}), ("charge", {"amount": 7})), text("Charged 12.")])
    agent = Agent(model, tools=[charge], store=store)
    run_id = agent.start("charge 5 and 7")
    events = agent.execute(run_id)
    for event in events:
        if event.type == "tool.completed":
            break  # the worker dies right after the first charge
    events.close()
    assert effects == [5]
    assert store.load_run(run_id).status is RunStatus.RUNNING

    time.sleep(1.1)  # its lease expires
    survivor = Agent(model, tools=[charge], store=store.connect_another())
    recovered = list(survivor.execute(run_id))
    types = [event.type for event in recovered if event.type != "model.delta"]
    assert types[0] == "run.resumed" and recovered[0].payload["recovered"] is True
    assert types[-1] == "run.completed"
    # The first charge was replayed from its idempotency record; the second ran once.
    assert effects == [5, 7]


def test_a_side_effect_with_unknown_outcome_is_not_retried(store):
    effects = []

    @tool(side_effect="non_idempotent")
    def transfer(amount: int) -> dict:
        "Transfer money."
        effects.append(amount)
        raise KeyboardInterrupt  # the process dies inside the side effect

    model = ScriptedModel([calls(("transfer", {"amount": 100})), text("I could not confirm the transfer.")])
    agent = Agent(model, tools=[transfer], store=store)
    run_id = agent.start("send 100")
    with pytest.raises(KeyboardInterrupt):
        list(agent.execute(run_id))
    time.sleep(1.1)
    survivor = Agent(model, tools=[transfer], store=store.connect_another())
    events = list(survivor.execute(run_id))
    failed = [event for event in events if event.type == "tool.failed"]
    assert effects == [100] and failed[0].payload["error_code"] == "requires_reconciliation"
    assert events[-1].type == "run.completed"


def test_a_store_worker_executes_and_recovers_runs(store):
    model = ScriptedModel([text("background answer")])
    agent = Agent(model, store=store)
    run_id = agent.start("q")
    worker = StoreWorker(store, lambda rid, wid: agent.execute(rid), recover=False, threads=2)
    assert worker.submit(run_id)
    deadline = time.monotonic() + 5
    while worker.is_active(run_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.shutdown(wait=True)
    assert store.load_run(run_id).status is RunStatus.COMPLETED


def test_suspended_runs_expire_and_drop_working_state(store):
    model = ScriptedModel([calls(("request_clarification", {"question": "?"}))])
    agent = Agent(model, store=store, tools=[])
    result = agent.run_sync("x")
    time.sleep(1.1)
    worker = StoreWorker(store, lambda rid, wid: iter(()), recover=False, waiting_ttl_seconds=1)
    worker.recover_once()
    run = store.load_run(result.run_id)
    assert run.status is RunStatus.CANCELLED and run.stop_reason == "waiting_expired"
    assert store.load_state(run).transcript == []


def test_postgres_grants_are_reread_every_call(postgres_store):
    from memoflow_agent.persistence.postgres import PostgresGrantStore
    from memoflow_agent.policy import new_grant

    grants = PostgresGrantStore(postgres_store)
    grant = grants.add(new_grant(principal_id="a", capability="docs.read", duration="persistent"))
    assert grants.active(principal_id="a", capability="docs.read", run_id=None, session_id=None)
    assert grants.revoke(grant.id) and not grants.active(principal_id="a", capability="docs.read", run_id=None, session_id=None)
