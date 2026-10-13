"""07 · Durable runs: suspend in one process, resume in another.

Uses PostgreSQL when MEMOFLOW_AGENT_POSTGRES_DSN is set
(pip install "memoflow-agent[postgres]"), otherwise an in-memory store
shared between two Agent instances to show the same flow.
"""

import os

from _common import default_model
from memoflow_agent import Agent, InMemoryStore, Principal, Resume


def make_store():
    dsn = os.getenv("MEMOFLOW_AGENT_POSTGRES_DSN")
    if not dsn:
        return InMemoryStore()
    from memoflow_agent.persistence.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.create_schema()
    return store


def main(model=None, *, later_model=None, store=None):
    store = store or make_store()
    user = Principal("user-123")
    first = Agent(model=default_model(model), store=store, principal=user)
    result = first.run_sync("Plan a weekend trip. Ask me where I want to go before planning.")
    print("status:", result.status.value, "| pending:", (result.pending or {}).get("question"))

    # …the process may restart here; only the run id is needed to continue.
    second = Agent(model=default_model(later_model or model), store=store, principal=user)
    if result.suspended:
        result = second.resume_sync(result.run_id, Resume.answer("Kyoto"))
    print(result.output)
    print("events stored:", len(second.events(result.run_id)))
    return result


if __name__ == "__main__":
    main()
