"""Shared fixtures. The default test run is offline: no network, no keys."""

from __future__ import annotations

import os
import uuid

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any real HTTP request in a default test run is a bug."""
    if os.getenv("MEMOFLOW_AGENT_LIVE_TESTS") == "1":
        yield
        return

    def refuse(self, request, *args, **kwargs):  # pragma: no cover - only on a bug
        raise AssertionError(f"unexpected network call to {request.url.host}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    yield


@pytest.fixture()
def postgres_store():
    """A PostgresStore in a throwaway schema; skipped without a DSN."""
    dsn = os.getenv("MEMOFLOW_AGENT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set MEMOFLOW_AGENT_TEST_POSTGRES_DSN to run PostgreSQL tests")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.rows import dict_row

    from memoflow_agent.persistence.postgres import PostgresStore

    schema = f"mfa_test_{uuid.uuid4().hex[:10]}"
    admin = psycopg.connect(dsn, autocommit=True)
    admin.execute(f"CREATE SCHEMA {schema}")
    connections = []

    def connect():
        connection = psycopg.connect(dsn, row_factory=dict_row, options=f"-c search_path={schema}")
        connections.append(connection)
        return PostgresStore(connection=connection, lease_seconds=1.0)

    store = connect()
    store.create_schema()
    store.connect_another = connect  # type: ignore[attr-defined]
    try:
        yield store
    finally:
        for connection in connections:
            connection.close()
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()
