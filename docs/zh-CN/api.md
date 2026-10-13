# API 参考（v0.1）

[English](../api.md)

## `Agent(model, *, tools=(), instructions=None, policy=None, store=None, budget=None, principal=None, grants=None, max_output_tokens=4096, input_budget=None, context_builder=None, retries=2, usage_observers=(), clarification=True, executor_options=None, model_options=None)`

| 方法 | 返回 |
| --- | --- |
| `await run(prompt, **opts)` / `run_sync(...)` | `RunResult` |
| `stream(prompt, **opts)`（异步迭代器）/ `stream_sync(...)` | `Event` 序列 |
| `await resume(run_id, Resume)` / `resume_sync(...)` / `stream_resume(...)` / `stream_resume_sync(...)` | `RunResult` / `Event` 序列 |
| `start(prompt, *, history=None, session_id=None, interactive=True, idempotency_key=None)` | run id |
| `execute(run_id)` | `Event` 序列（驱动已创建 / 已恢复 / 需崩溃恢复的 run） |
| `cancel(run_id)`、`get_run(run_id)`、`events(run_id, after_seq=0)` | — |

`model`：`"openai:<id>"`、`"anthropic:<id>"`、`"deepseek:<id>"`，或一个 Provider 对象。
`policy` 默认 `CapabilityPolicy(grants=grants)`；`store` 默认 `InMemoryStore()`；`budget` 默认 `BudgetLimits()`。
未传 `session_id` 时，每个 run 自成一个会话（"会话级"授权只覆盖该 run 及其恢复）。

## `RunResult`
`run_id, status (RunStatus), output, partial, stop_reason, error_code, pending, usage, events`；
属性 `completed`、`suspended`。挂起时 `pending` 为 `{"type": "<事件类型>", ...载荷}`。

## `Resume`
`Resume.answer(text=None, option_ids=[])`、`Resume.authorize(granted=True, duration="once"|"session"|"persistent")`、
`Resume.approve(approval_id, approved=True)`、`Resume.reject(approval_id)`。

## 工具
`@tool(fn=None, *, name, description, risk, side_effect, permission, requires_approval, timeout, retry, public_fields, audit_fields, interactive_only)`
→ `ToolSpec`。handler 可以返回 `ToolResult`、dict、pydantic 模型或标量。需要让模型知道的失败请抛出
`ToolError(code, message, retryable=False)`。名为 `ctx` 的参数会接收 `ToolContext`
（`run_id, state, principal, session_id, scope, is_cancelled`）。`ToolSpec.metadata["reconcile"]` 可放
`reconcile(ctx, arguments) -> ToolResult | None`，用于结果未知的副作用。

## 策略
`Allow(scope=None)`、`Deny(code, message)`、`RequireAuthorization(resource, reason)`、`RequireConfirmation(reason)`；
`AllowAll`、`CapabilityPolicy(grants, grantable, approve_risks, lock_on_injection=True)`；`Principal(id, capabilities)`；
`GrantDuration`、`InMemoryGrantStore`、`PostgresGrantStore`、`new_grant`、`new_approval`、`validate_approval`。
任何带 `evaluate(spec, arguments, ctx)` 的对象（或可调用对象）都可以作为策略。

## 预算
`BudgetLimits(max_model_turns=8, max_tool_calls=12, max_run_tokens=240_000, max_run_seconds=300, max_consecutive_failures=3, max_identical_calls=3, max_input_tokens=None, max_output_tokens=None)`；
`AgentBudget(max_tokens=…, max_seconds=…, …)` 用于构造它。

## Provider
`ModelProvider` 协议：`name`、`model`、`capabilities`、`stream_turn(request, *, is_cancelled) -> Iterator[TextDelta | ProviderEvent | ModelTurn]`。
`StreamingProvider` 基类（实现 `raw_frames`）、`RetryingProvider`、`generate(provider, request)`、
`resolve_model`、`register_provider`、`ModelCapabilities`、`ModelRequest`、`ModelTurn`。

## 存储
`RunStore` 协议（语义约定见 `persistence/base.py`）、`InMemoryStore(lease_seconds=45)`、
`PostgresStore(dsn | connection=…, lease_seconds=45)`，带 `create_schema()`。

## 嵌入：`AgentLoop` 与 `Application`
`AgentLoop(run, state, store, events, app, executor, provider, limits, worker_id, cancellations=None, …).execute()`
产出事件信封。继承 `Application` 来接管消息、载荷、计费和上下文。

## 事件
`Event(id, run_id, seq, type, ts, payload)`；增量文本用 `.text`；`EventType`；`EventMapper`；`EventBroker`。

## 错误
`MemoFlowAgentError` → `ConfigurationError`、`ModelProviderError(code)`、`ContextTooLong`、`BudgetExceeded(reason)`、
`CapacityExhausted`、`LeaseLost`、`RunNotFound`、`RunNotSuspended`、`InvalidResume(code)`、`CheckpointIncompatible`。

## 测试
`memoflow_agent.testing`：`ScriptedModel([...])`、`text()`、`calls()`、`fail()`、`Raw(frames)`。
