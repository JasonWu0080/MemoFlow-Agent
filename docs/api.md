# API reference (v0.1)

[中文](zh-CN/api.md)

## `Agent(model, *, tools=(), instructions=None, policy=None, store=None, budget=None, principal=None, grants=None, max_output_tokens=4096, input_budget=None, context_builder=None, retries=2, usage_observers=(), clarification=True, executor_options=None, model_options=None)`

| Method | Returns |
| --- | --- |
| `await run(prompt, **opts)` / `run_sync(...)` | `RunResult` |
| `stream(prompt, **opts)` (async iterator) / `stream_sync(...)` | `Event`s |
| `await resume(run_id, Resume)` / `resume_sync(...)` / `stream_resume(...)` / `stream_resume_sync(...)` | `RunResult` / `Event`s |
| `start(prompt, *, history=None, session_id=None, interactive=True, idempotency_key=None)` | run id |
| `execute(run_id)` | `Event`s (drives a created / resumed / recovered run) |
| `cancel(run_id)`, `get_run(run_id)`, `events(run_id, after_seq=0)` | — |

`model`: `"openai:<id>"`, `"anthropic:<id>"`, `"deepseek:<id>"`, or a provider object.
`policy` defaults to `CapabilityPolicy(grants=grants)`; `store` to `InMemoryStore()`; `budget` to `BudgetLimits()`.

## `RunResult`
`run_id, status (RunStatus), output, partial, stop_reason, error_code, pending, usage, events`;
properties `completed`, `suspended`. `pending` is `{"type": "<event type>", ...payload}` when suspended.

## `Resume`
`Resume.answer(text=None, option_ids=[])`, `Resume.authorize(granted=True, duration="once"|"session"|"persistent")`,
`Resume.approve(approval_id, approved=True)`, `Resume.reject(approval_id)`.

## Tools
`@tool(fn=None, *, name, description, risk, side_effect, permission, requires_approval, timeout, retry, public_fields, audit_fields, interactive_only)`
→ `ToolSpec`. Handlers may return `ToolResult`, a dict, a pydantic model or a scalar. Raise `ToolError(code, message, retryable=False)`
for failures the model should hear about. A parameter named `ctx` receives the `ToolContext`
(`run_id, state, principal, session_id, scope, is_cancelled`). `ToolSpec.metadata["reconcile"]` may hold
`reconcile(ctx, arguments) -> ToolResult | None` for side effects with unknown outcome.

## Policy
`Allow(scope=None)`, `Deny(code, message)`, `RequireAuthorization(resource, reason)`, `RequireConfirmation(reason)`;
`AllowAll`, `CapabilityPolicy(grants, grantable, approve_risks, lock_on_injection=True)`; `Principal(id, capabilities)`;
`GrantDuration`, `InMemoryGrantStore`, `PostgresGrantStore`, `new_grant`, `new_approval`, `validate_approval`.
Any object with `evaluate(spec, arguments, ctx)` (or a callable) is a policy.

## Budget
`BudgetLimits(max_model_turns=8, max_tool_calls=12, max_run_tokens=240_000, max_run_seconds=300, max_consecutive_failures=3, max_identical_calls=3, max_input_tokens=None, max_output_tokens=None)`;
`AgentBudget(max_tokens=…, max_seconds=…, …)` builds one.

## Providers
`ModelProvider` protocol: `name`, `model`, `capabilities`, `stream_turn(request, *, is_cancelled) -> Iterator[TextDelta | ProviderEvent | ModelTurn]`.
`StreamingProvider` base (implement `raw_frames`), `RetryingProvider`, `generate(provider, request)`,
`resolve_model`, `register_provider`, `ModelCapabilities`, `ModelRequest`, `ModelTurn`.

## Stores
`RunStore` protocol (see `persistence/base.py` for the semantics contract), `InMemoryStore(lease_seconds=45)`,
`PostgresStore(dsn | connection=…, lease_seconds=45)` with `create_schema()`.

## Embedding: `AgentLoop` and `Application`
`AgentLoop(run, state, store, events, app, executor, provider, limits, worker_id, cancellations=None, …).execute()`
yields event envelopes. Subclass `Application` to own messages, payloads, billing and context.

## Events
`Event(id, run_id, seq, type, ts, payload)`; `.text` for deltas; `EventType`; `EventMapper`; `EventBroker`.

## Errors
`MemoFlowAgentError` → `ConfigurationError`, `ModelProviderError(code)`, `ContextTooLong`, `BudgetExceeded(reason)`,
`CapacityExhausted`, `LeaseLost`, `RunNotFound`, `RunNotSuspended`, `InvalidResume(code)`, `CheckpointIncompatible`.

## Testing
`memoflow_agent.testing`: `ScriptedModel([...])`, `text()`, `calls()`, `fail()`, `Raw(frames)`.
