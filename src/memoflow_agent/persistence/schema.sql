-- memoflow-agent PostgreSQL schema (schema_version 1).
-- Generic tables for durable runs; independent of any application's schema.
-- Apply with PostgresStore.create_schema() or run this file yourself.

CREATE TABLE IF NOT EXISTS memoflow_agent_runs (
    id                  TEXT PRIMARY KEY,
    status              TEXT        NOT NULL,
    segment             INTEGER     NOT NULL DEFAULT 0,
    principal_id        TEXT,
    session_id          TEXT,
    idempotency_key     TEXT,
    worker_id           TEXT,
    lease_expires_at    TIMESTAMPTZ,
    cancel_requested_at TIMESTAMPTZ,
    waiting_reason      TEXT,
    stop_reason         TEXT,
    error_code          TEXT,
    error_message       TEXT,
    output              TEXT,
    version             BIGINT      NOT NULL DEFAULT 0,
    metadata            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    checkpoint          JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    paused_at           TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS memoflow_agent_runs_idempotency
    ON memoflow_agent_runs (principal_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS memoflow_agent_runs_recoverable
    ON memoflow_agent_runs (status, lease_expires_at);

CREATE TABLE IF NOT EXISTS memoflow_agent_events (
    run_id     TEXT        NOT NULL REFERENCES memoflow_agent_runs (id) ON DELETE CASCADE,
    seq        INTEGER     NOT NULL,
    type       TEXT        NOT NULL,
    envelope   JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS memoflow_agent_steps (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT        NOT NULL REFERENCES memoflow_agent_runs (id) ON DELETE CASCADE,
    sequence        INTEGER     NOT NULL,
    type            TEXT        NOT NULL,
    name            TEXT,
    status          TEXT        NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    duration_ms     INTEGER,
    token_usage     JSONB,
    error_code      TEXT,
    budget_snapshot JSONB,
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS memoflow_agent_tool_records (
    key        TEXT PRIMARY KEY,
    record     JSONB       NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memoflow_agent_grants (
    id           TEXT PRIMARY KEY,
    principal_id TEXT        NOT NULL,
    capability   TEXT        NOT NULL,
    duration     TEXT        NOT NULL,
    run_id       TEXT,
    session_id   TEXT,
    expires_at   TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memoflow_agent_grants_lookup
    ON memoflow_agent_grants (principal_id, capability) WHERE revoked_at IS NULL;
