# Architecture

[中文](zh-CN/architecture.md)

## Layers

```text
Applications ─ Application adapters (hooks)
        │
        ▼
Agent kernel
  agent.py        Agent: wiring + sync/async API
  loop.py         AgentLoop (the only loop) + Application hook interface
  decisions.py    FINAL / TOOL_CALL / CLARIFICATION, tool-call assembly
  budget.py       guards on persisted counters
  context.py      token estimation, layered compaction
  tools/          contract, @tool, ToolExecutor pipeline
  policy.py       outcomes, capability policy, grants, approvals
  events.py       envelope, kernel event names, mapper, broker
  resume.py       answering a suspended run (shared state transitions)
  search.py       OFF / HOSTED / ISOLATED decision + privacy gate
        │
        ▼
Infrastructure interfaces
  models/         ModelProvider (OpenAI, Anthropic, DeepSeek, yours)
  persistence/    RunStore (InMemory, PostgreSQL, yours)
  execution/      Worker (thread pool + lease recovery)
  observability   UsageObserver
```

The kernel imports no web framework, no ORM and no application module.

## The loop

```text
start segment ─ set RUNNING (RUNNING already ⇒ recovery: run.resumed{recovered} + model.reset)
  └─ loop: cancelled? → queued calls? → model turn
       model turn: budget check → app.prepare_turn (context, capacity) → begin step
                   → provider.stream_turn (deltas, provider events, ModelTurn)
                   → account usage → app.after_model_turn → finish step → checkpoint
                   → decide():
                       FINAL         → app.complete → COMPLETED → run.completed
                       CLARIFICATION → pending → SUSPENDED → run.suspended
                       TOOL_CALL     → for each call: checkpoint intent → budget → executor
                                         authorization → pending → SUSPENDED
                                         confirmation  → approval → SUSPENDED
                                         result.pending (app proposal) → SUSPENDED
                                         else observation → transcript → checkpoint
```

Runtime-only outcomes (never chosen by the model): `SUSPENDED`, `FAILED`, `CANCELLED`, budget exhaustion.

## Run state machine

`created → running → {suspended, completed, failed, cancelled}`, `suspended → {created (resume), cancelled}`,
`running → running` (recovery). Every transition is checked (`state.check_transition`).

## Events

Persistent events get a per-run `seq` from the lease holder and are stored before delivery. `model.delta`
is ephemeral (carries the last persisted `seq` and a character `offset`). Applications with an existing
wire protocol rename events with `EventMapper` (MemoFlow maps onto its V4 protocol and suppresses
kernel-only events).

## ADR-K1 — a synchronous loop

The loop is a synchronous generator. Reasons: (1) it must run next to synchronous database code in thread
pools (MemoFlow's production setup); (2) applications bind context variables (metering, tracing) around
each step, which a synchronous generator preserves trivially; (3) one loop instead of a sync and an async
copy. The async API (`Agent.run/stream/resume`) drives the loop in a worker thread with the caller's
context copied. Async tools run on one private event loop per run. Trade-off: a CPU-bound tool blocks its
run's thread; use async tools or threads for slow I/O.

## ADR-K2 — authorization ≠ confirmation

A grant (authorization) covers a capability for a duration; an approval (confirmation) covers one call
and the hash of its arguments, expires, and is consumed. Applications may also suspend *after* a tool
ran (a proposal the user confirms through the app) via `ToolResult.pending`.

## ADR-K3 — durable tool intent

Before each tool call the decided calls are checkpointed as `queued_calls`. A recovered run executes them
with the same call ids, so idempotency records apply; it does not ask the model to decide again.

## How MemoFlow embeds the kernel

MemoFlow keeps accounts, notes, memory, billing (Credits holds per run segment), Web Search V5, its
database tables and its Flutter V4 protocol in an adapter: `MemoFlowApplication` (hooks),
`MemoFlowRunStore` (its tables), `MemoFlowExecutionHooks` (audit rows, proposal replay), an
`EventMapper` onto V4 names, and its own model gateway wrapped as a provider.
