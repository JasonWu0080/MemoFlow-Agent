# 架构

[English](../architecture.md)

## 分层

```text
应用 ─ 应用适配层（钩子）
        │
        ▼
Agent 内核
  agent.py        Agent：组装 + 同步/异步 API
  loop.py         AgentLoop（唯一的循环）+ Application 钩子接口
  decisions.py    FINAL / TOOL_CALL / CLARIFICATION，工具调用组装
  budget.py       基于持久化计数器的护栏
  context.py      token 估算、分层压缩
  tools/          契约、@tool、ToolExecutor 管线
  policy.py       结果类型、能力策略、授权、审批
  events.py       信封、内核事件名、映射器、广播
  resume.py       回答挂起的 run（共享的状态转换）
  search.py       OFF / HOSTED / ISOLATED 判定 + 隐私闸门
        │
        ▼
基础设施接口
  models/         ModelProvider（OpenAI、Anthropic、DeepSeek、自定义）
  persistence/    RunStore（InMemory、PostgreSQL、自定义）
  execution/      Worker（线程池 + 租约恢复）
  observability   UsageObserver
```

内核不导入任何 Web 框架、ORM 或应用模块。

## 循环

```text
开始 segment ─ 置为 RUNNING（已经是 RUNNING ⇒ 恢复：run.resumed{recovered} + model.reset）
  └─ 循环：已取消？→ 有排队的调用？→ 模型回合
       模型回合：预算检查 → app.prepare_turn（上下文、额度）→ 开始步骤
                 → provider.stream_turn（增量文本、Provider 事件、ModelTurn）
                 → 统计用量 → app.after_model_turn → 结束步骤 → 检查点
                 → decide()：
                     FINAL         → app.complete → COMPLETED → run.completed
                     CLARIFICATION → pending → SUSPENDED → run.suspended
                     TOOL_CALL     → 对每个调用：写入意图检查点 → 预算 → 执行器
                                       需要授权 → pending → SUSPENDED
                                       需要确认 → 审批 → SUSPENDED
                                       result.pending（应用提案）→ SUSPENDED
                                       否则 observation → transcript → 检查点
```

只由运行时产生的结果（模型永远无法选择）：`SUSPENDED`、`FAILED`、`CANCELLED`、预算耗尽。

## Run 状态机

`created → running → {suspended, completed, failed, cancelled}`，`suspended → {created（恢复）, cancelled}`，
`running → running`（崩溃恢复）。每次状态转换都会校验（`state.check_transition`）。

## 事件

持久化事件由租约持有者分配每个 run 内的 `seq`，先存储再投递。`model.delta` 是瞬时事件（携带最近一次持久化的
`seq` 和字符 `offset`）。已有线上协议的应用用 `EventMapper` 给事件改名（MemoFlow 映射到它的 V4 协议，并屏蔽仅内核有的事件）。

## ADR-K1 —— 同步循环

循环是一个同步生成器。原因：(1) 它必须能和同步数据库代码一起在线程池中运行（MemoFlow 的生产架构）；
(2) 应用会在每一步周围绑定上下文变量（计量、追踪），同步生成器天然保留这一点；(3) 只维护一个循环，而不是同步、异步两份。
异步 API（`Agent.run/stream/resume`）在工作线程中驱动该循环，并复制调用方的上下文。异步工具在每个 run 私有的事件循环上运行。
代价：CPU 密集型工具会阻塞其所在 run 的线程；慢 I/O 请用异步工具或线程。

## ADR-K2 —— 授权 ≠ 确认

授权（grant）在一段时间内覆盖某项能力；审批（确认）只覆盖一次调用及其参数哈希，会过期，且用后即失效。
应用也可以在工具运行*之后*挂起（由应用让用户确认的提案），方式是 `ToolResult.pending`。

## ADR-K3 —— 持久化的工具调用意图

每次工具调用前，已决定的调用会作为 `queued_calls` 写入检查点。恢复后的 run 用相同的调用 ID 执行它们，幂等记录因此生效；不会再让模型重新决定。

## MemoFlow 如何嵌入内核

MemoFlow 把账户、笔记、记忆、计费（每个 run segment 的 Credits 预留）、Web Search V5、数据库表和 Flutter V4 协议放在适配层：
`MemoFlowApplication`（钩子）、`MemoFlowRunStore`（它的表）、`MemoFlowExecutionHooks`（审计行、提案重放）、
映射到 V4 名称的 `EventMapper`，以及包装成 Provider 的自有模型网关。
