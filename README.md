# MemoFlow Agent

**Small Core. Reliable Agents.**

A lightweight, extensible Python AI agent framework with model-driven tool calling, durable execution,
permission-aware tools, and human-in-the-loop workflows.

MemoFlow Agent is the agent kernel of the AI assistant in the [MemoFlow](#about) app, extracted as a
standalone package. The MemoFlow app runs this same loop through an application adapter; there is no
private fork of it.

[中文文档](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/README.zh-CN.md)

> Status: **v0.1.0, alpha.** The API may still change before 1.0.

---

## Features

- **One agent loop, model-driven.** Each turn the model decides: answer, call tools, or ask a clarifying
  question. Decisions come only from the provider's native tool-calling protocol, never from parsing prose.
- **Deterministic runtime.** Permissions, budgets and run state are decided by code, not by the model.
- **Typed tools.** `@tool` turns a Python function into a strict JSON-schema tool. Unknown arguments are
  rejected. Sync and `async` tools are both supported.
- **Permission policy.** Capability-based: principals, grants (once / session / persistent), revocation
  that takes effect on the very next call.
- **Human-in-the-loop.** Approvals are bound to one tool call and the hash of its exact arguments. They
  expire, can be used only once, and cannot be replayed.
- **Clarification.** The model can pause the run to ask the user a question. The same run resumes with the answer.
- **Budgets.** Model turns, tool calls, tokens (total, input, output), wall time, consecutive failures and
  repeated identical calls. Counters survive checkpoints.
- **Streaming events.** One event envelope with per-run sequence numbers, built for replay.
- **Durable execution (optional).** PostgreSQL store with checkpoints, leases, crash recovery and
  idempotent side effects.
- **Providers:** OpenAI, Anthropic and DeepSeek. Adding another takes about 50 lines.

Not in v0.1: multi-agent orchestration, workflow builders, built-in RAG or vector stores, browser or shell
tools. See [docs/roadmap.md](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/roadmap.md).

## Installation

```bash
pip install memoflow-agent
# durable runs on PostgreSQL:
pip install "memoflow-agent[postgres]"
```

Python 3.11+. Core dependencies: `pydantic` and `httpx`.

## Quickstart

```python
import asyncio
from memoflow_agent import Agent

async def main():
    agent = Agent(model="openai:<model-id>")       # or "anthropic:<model-id>", "deepseek:<model-id>"
    result = await agent.run("Explain how AI agents work.")
    print(result.output)

asyncio.run(main())
```

Set the provider key in `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY`. The id after the colon
is passed to the provider unchanged; the framework does not keep a model list.

Prefer synchronous code? `agent.run_sync(...)` and `agent.stream_sync(...)` do the same thing.

## Tool calling

```python
from memoflow_agent import Agent, tool

@tool(risk="low")
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression."""
    ...

agent = Agent(model="openai:<model-id>", tools=[calculate])
result = agent.run_sync("What is 17 * 23?")
```

Every call goes through one pipeline: lookup → schema validation → policy → idempotency → execution
(retry policy, timeout for async tools) → output cap → injection scan → audit. Tool metadata includes
`risk`, `side_effect` (`read_only`, `idempotent`, `non_idempotent`, `human_approved`), `permission`,
`requires_approval`, `timeout` and `retry`.

## Streaming

```python
async for event in agent.stream("Write a haiku."):
    if event.type == "model.delta":
        print(event.text, end="")
```

Event types: `run.started`, `run.resumed`, `run.suspended`, `run.completed`, `run.failed`,
`run.cancelled`, `model.delta`, `model.reset`, `tool.started`, `tool.completed`, `tool.failed`,
`clarification.required`, `authorization.required`, `approval.required`. Every event except `model.delta`
is stored with a sequence number. Deltas are not stored and are never promised to replay token by token.

## Model providers

| Prefix | Class | API |
| --- | --- | --- |
| `openai:` | `OpenAIProvider` | Chat Completions, streaming, native tools |
| `anthropic:` | `AnthropicProvider` | Messages API, streaming, native tool use |
| `deepseek:` | `DeepSeekProvider` | OpenAI-compatible Chat Completions |

Capabilities are declared, not guessed: `ModelCapabilities(tool_calling=True, streaming=True)` by default,
and everything else (`structured_output`, `native_web_search`, `vision`, `reasoning`) is off until you turn
it on. Passing tools to a model whose capabilities exclude tool calling raises `ConfigurationError`. Bring
your own provider by passing any object with a `stream_turn()` method, or with `register_provider()`.

## Permissions

```python
from memoflow_agent import Agent, CapabilityPolicy, InMemoryGrantStore, Principal, Resume, tool

@tool(permission="calendar.read")
def read_calendar(day: str) -> list[str]: ...

grants = InMemoryGrantStore()
policy = CapabilityPolicy(grants=grants, grantable=frozenset({"calendar.read"}))
agent = Agent(model="openai:<model-id>", tools=[read_calendar], policy=policy, grants=grants,
              principal=Principal("user-123"))

result = agent.run_sync("What's on my calendar tomorrow?")
if result.suspended and result.pending["type"] == "authorization.required":
    result = agent.resume_sync(result.run_id, Resume.authorize(True, duration="session"))
```

The model cannot grant access to itself. Grants are re-read on every call, so a revoked grant stops the
next call. Once untrusted tool output tries to give the agent instructions, tools with side effects are
blocked for the rest of that run.

## Human-in-the-loop

```python
@tool(side_effect="non_idempotent", requires_approval=True)
def send_email(to: str, body: str) -> dict: ...

result = agent.run_sync("Email ana@example.com that the report is ready.")
if result.suspended and result.pending["type"] == "approval.required":
    result = agent.resume_sync(result.run_id, Resume.approve(result.pending["approval_id"]))
```

**Authorization** grants access to a kind of resource or capability. **Approval** approves one call with
its exact arguments. Changed arguments, a second use, an expired approval or a different principal are all
rejected.

## Durable execution

```python
from memoflow_agent.persistence.postgres import PostgresStore

store = PostgresStore("postgresql://…")
store.create_schema()
agent = Agent(model="openai:<model-id>", store=store)
```

- Runs, checkpoints (versioned; incompatible ones are refused), events and leases live in PostgreSQL.
- A run is executed only by the holder of its lease. If a worker dies, its lease expires and
  `StoreWorker` (or any `agent.execute(run_id)`) resumes the run from its checkpoint.
- Before each tool call, the calls the model decided on are checkpointed. A recovered run executes exactly
  those calls instead of asking the model again.
- **Side-effect semantics:** `non_idempotent` and `human_approved` tools run **at most once** per
  `(run, tool call)`. After a crash, a completed call is replayed from its record. A call that started but
  never recorded a result is **not retried**: it reports `requires_reconciliation`, or you can provide a
  `reconcile` hook that checks the external system. Exactly-once cannot be guaranteed when the outcome
  cannot be verified, and the framework does not pretend it can.

The default `InMemoryStore` needs no setup and supports suspend and resume within one process. It does not
survive a restart.

## Architecture

```text
Your application ── Application hooks (messages, billing, payloads)
        │
        ▼
  AgentLoop (single loop) ── decisions · budget · context · events
        │
        ├── ToolExecutor ── Policy (grants, approvals) ── your tools
        ├── ModelProvider (OpenAI / Anthropic / DeepSeek / yours)
        └── RunStore (InMemory / PostgreSQL / yours) ── Worker (optional)
```

The loop is synchronous by design (ADR-K1): it runs next to synchronous database code in a thread pool.
The async API drives it in a worker thread. See [docs/architecture.md](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/architecture.md).

## Documentation

- [Architecture](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/architecture.md)
- [API reference](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/api.md)
- [Security model](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/security.md)
- [Roadmap](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/docs/roadmap.md)
- [Examples](https://github.com/JasonWu0080/MemoFlow-Agent/tree/main/examples): basic agent, tools, streaming, multiple providers, permissions, approval,
  persistence, knowledge agent

## Contributing

See [CONTRIBUTING.md](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/CONTRIBUTING.md). Report security issues privately as described in
[SECURITY.md](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/SECURITY.md).

## License

Copyright 2026 Nanjing Junhe E-Commerce Co., Ltd. Licensed under the Apache License 2.0; see
[LICENSE](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/LICENSE) and [NOTICE](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/NOTICE). The license does not cover the names "MemoFlow" and
"MemoFlow Agent"; see [TRADEMARK.md](https://github.com/JasonWu0080/MemoFlow-Agent/blob/main/TRADEMARK.md).

## About

MemoFlow is a personal knowledge and AI meeting app. Its assistant is built on this kernel: MemoFlow
keeps only its business logic (accounts, notes, billing, its wire protocol) in an adapter around it.
