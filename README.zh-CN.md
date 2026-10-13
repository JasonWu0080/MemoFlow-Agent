# MemoFlow Agent

**Small Core. Reliable Agents.（小内核，可靠的智能体。）**

一个轻量、可扩展的 Python AI Agent 框架：模型驱动的原生工具调用、可持久化执行、权限感知的工具，以及人在回路（Human-in-the-Loop）工作流。

MemoFlow Agent 是 [MemoFlow](#关于) App 中 AI 助手的 Agent 内核，抽取为独立的开源包。MemoFlow App 通过应用适配层运行的就是同一个 Agent Loop，没有私有分叉。

[English](README.md)

> 状态：**v0.1.0，Alpha**。1.0 之前 API 可能调整。

---

## 特性

- **单一 Agent Loop，模型驱动**：每一轮由模型决定直接回答、调用工具，还是向用户提一个澄清问题。决策只来自 Provider 的原生工具调用协议，绝不靠解析模型文本。
- **确定性运行时**：权限、预算、Run 状态都由代码决定，不由模型决定。
- **类型化工具**：`@tool` 把 Python 函数变成严格 JSON Schema 的工具，未知参数一律拒绝；同步和 `async` 工具都支持。
- **权限策略**：基于能力（Capability）：主体（Principal）、授权（单次 / 会话 / 持久）、撤销后下一次调用立即生效。
- **人在回路**：审批绑定到具体的一次工具调用及其参数哈希，有过期时间、只能用一次、不可重放。
- **澄清**：模型可以暂停 Run 向用户提问，用户回答后在同一个 Run 中继续。
- **预算**：模型轮数、工具调用数、Token（总量 / 输入 / 输出）、墙钟时间、连续失败、重复的相同调用；计数随检查点保存，恢复后不清零。
- **流式事件**：统一的事件信封，每个 Run 内序号单调递增，可重放。
- **可选的持久化执行**：PostgreSQL 存储，支持检查点、租约、崩溃恢复和幂等的副作用。
- **Provider**：OpenAI、Anthropic、DeepSeek；新增一个 Provider 大约 50 行代码。

v0.1 不包含：多 Agent 编排、可视化工作流、内置 RAG / 向量库、浏览器或 Shell 工具。见 [docs/roadmap.md](docs/zh-CN/roadmap.md)。

## 安装

```bash
pip install memoflow-agent
# 需要 PostgreSQL 持久化：
pip install "memoflow-agent[postgres]"
```

需要 Python 3.11+。核心依赖只有 `pydantic` 和 `httpx`。

## 快速开始

```python
import asyncio
from memoflow_agent import Agent

async def main():
    agent = Agent(model="openai:<model-id>")       # 或 "anthropic:<model-id>"、"deepseek:<model-id>"
    result = await agent.run("用两句话解释 AI Agent 是怎么工作的。")
    print(result.output)

asyncio.run(main())
```

在 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY` 中设置对应密钥。冒号后的模型 ID 原样传给 Provider，框架不维护模型列表。同步代码可用 `agent.run_sync(...)` / `agent.stream_sync(...)`。

## 工具调用

```python
from memoflow_agent import Agent, tool

@tool(risk="low")
def calculate(expression: str) -> float:
    """计算一个算术表达式。"""
    ...

agent = Agent(model="openai:<model-id>", tools=[calculate])
result = agent.run_sync("17 × 23 等于多少？")
```

所有工具调用都走同一条管线：查找 → Schema 校验 → 策略 → 幂等 → 执行（重试策略；异步工具有超时）→ 输出截断 → 注入扫描 → 审计。工具元数据包括 `risk`、`side_effect`（`read_only` / `idempotent` / `non_idempotent` / `human_approved`）、`permission`、`requires_approval`、`timeout`、`retry`。

## 流式输出

```python
async for event in agent.stream("写一首俳句。"):
    if event.type == "model.delta":
        print(event.text, end="")
```

事件类型：`run.started`、`run.resumed`、`run.suspended`、`run.completed`、`run.failed`、`run.cancelled`、`model.delta`、`model.reset`、`tool.started`、`tool.completed`、`tool.failed`、`clarification.required`、`authorization.required`、`approval.required`。除 `model.delta` 外，所有事件都带序号持久化；delta 不持久化，框架不承诺逐 Token 重放。

## 模型 Provider

| 前缀 | 类 | 接口 |
| --- | --- | --- |
| `openai:` | `OpenAIProvider` | Chat Completions，流式，原生工具 |
| `anthropic:` | `AnthropicProvider` | Messages API，流式，原生 tool use |
| `deepseek:` | `DeepSeekProvider` | OpenAI 兼容的 Chat Completions |

模型能力是声明出来的，不是猜出来的：默认只有 `tool_calling` 与 `streaming`，`structured_output`、`native_web_search`、`vision`、`reasoning` 需要显式开启。给不支持工具调用的模型传工具会直接抛出 `ConfigurationError`。任何实现了 `stream_turn()` 的对象都可以作为 Provider 传入，也可以用 `register_provider()` 注册前缀。

## 权限

```python
from memoflow_agent import Agent, CapabilityPolicy, InMemoryGrantStore, Principal, Resume, tool

@tool(permission="calendar.read")
def read_calendar(day: str) -> list[str]: ...

grants = InMemoryGrantStore()
policy = CapabilityPolicy(grants=grants, grantable=frozenset({"calendar.read"}))
agent = Agent(model="openai:<model-id>", tools=[read_calendar], policy=policy, grants=grants,
              principal=Principal("user-123"))

result = agent.run_sync("我明天有什么安排？")
if result.suspended and result.pending["type"] == "authorization.required":
    result = agent.resume_sync(result.run_id, Resume.authorize(True, duration="session"))
```

模型无法给自己授权。授权每次调用都重新读取，撤销后下一次调用即失效。一旦不可信的工具输出试图向 Agent 下达指令，本次 Run 中所有有副作用的工具都会被锁定。

## 人在回路

```python
@tool(side_effect="non_idempotent", requires_approval=True)
def send_email(to: str, body: str) -> dict: ...

result = agent.run_sync("给 ana@example.com 发邮件说报告好了。")
if result.suspended and result.pending["type"] == "approval.required":
    result = agent.resume_sync(result.run_id, Resume.approve(result.pending["approval_id"]))
```

**授权（Authorization）** 授予对某类资源或能力的访问；**审批（Approval / Confirmation）** 批准某一次具体调用及其参数。参数被改动、重复使用、过期或由其他主体提交的审批都会被拒绝。

## 持久化执行

```python
from memoflow_agent.persistence.postgres import PostgresStore

store = PostgresStore("postgresql://…")
store.create_schema()
agent = Agent(model="openai:<model-id>", store=store)
```

- Run、检查点（带版本，不兼容的版本拒绝恢复）、事件和租约都存放在 PostgreSQL 中。
- 只有持有租约的 Worker 才能执行 Run。Worker 崩溃后租约过期，`StoreWorker`（或任何 `agent.execute(run_id)`）会从检查点恢复。
- 每次工具调用之前，模型已经决定好的调用会先写入检查点。恢复后的 Run 执行的正是这些调用，不会重新询问模型。
- **副作用语义**：`non_idempotent` 和 `human_approved` 工具对同一个 `(run, tool call)` **最多执行一次**。崩溃后，已完成的调用从记录中回放；开始了却没有记录结果的调用**不会重试**，而是报告 `requires_reconciliation`，也可以提供 `reconcile` 钩子去外部系统核对结果。在无法验证结果时不可能保证 Exactly-Once，框架也不假装能做到。

默认的 `InMemoryStore` 零配置，支持同一进程内的挂起与恢复，但进程重启后数据不保留。

## 架构

```text
你的应用 ── Application hooks（消息、计费、事件载荷）
    │
    ▼
AgentLoop（唯一的循环）── 决策 · 预算 · 上下文 · 事件
    │
    ├── ToolExecutor ── Policy（授权、审批）── 你的工具
    ├── ModelProvider（OpenAI / Anthropic / DeepSeek / 自定义）
    └── RunStore（InMemory / PostgreSQL / 自定义）── Worker（可选）
```

Loop 有意设计为同步的（ADR-K1）：这样能和同步数据库代码一起在线程池里运行；异步 API 在工作线程中驱动它。详见 [docs/architecture.md](docs/zh-CN/architecture.md)。

## 文档

- [架构](docs/zh-CN/architecture.md)
- [API 参考](docs/zh-CN/api.md)
- [安全模型](docs/zh-CN/security.md)
- [路线图](docs/zh-CN/roadmap.md)
- [示例](examples/)：基础 Agent、工具、流式、多 Provider、权限、审批、持久化、知识库 Agent

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.zh-CN.md)。安全问题请按 [SECURITY.md](SECURITY.zh-CN.md) 私下报告。

## 许可证

Copyright 2026 Nanjing Junhe E-Commerce Co., Ltd.。以 Apache License 2.0 授权，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。许可证不涵盖 "MemoFlow" 与 "MemoFlow Agent" 这两个名称，见 [TRADEMARK.zh-CN.md](TRADEMARK.zh-CN.md)。

## 关于

MemoFlow 是一款个人知识与 AI 会议 App。它的 AI 助手构建在这个内核之上：MemoFlow 只把自己的业务（账户、笔记、计费、客户端协议）放在内核外的适配层中。
