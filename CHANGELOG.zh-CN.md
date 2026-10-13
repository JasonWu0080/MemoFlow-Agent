# 更新日志

[English](CHANGELOG.md)

所有值得注意的变化都记录在这里。格式遵循 Keep a Changelog，版本号遵循 SemVer。

## [未发布]

## [0.1.0] —— 2026-10-14

第一个公开发布版本：从 MemoFlow AI Agent V4 中抽取的 Agent 内核。

### 新增
- `AgentLoop`：唯一的、模型驱动的循环（FINAL / TOOL_CALL / CLARIFICATION），预算收尾、无效输出纠正、取消、挂起、失败处理；供嵌入应用使用的 `Application` 钩子。
- `Agent`：同步与异步的 run / stream / resume / cancel。
- 工具：`@tool`、`ToolSpec`、`ToolRegistry`，带钩子的唯一 `ToolExecutor` 管线、重试策略、异步超时、输出截断、注入扫描、带对账的幂等记录。
- 策略：`CapabilityPolicy`、主体、授权（单次 / 会话 / 持久）、绑定参数哈希的审批（含过期与防重放）。
- 上下文：token 估算、分层压缩、`ContextTooLong`。
- Provider：OpenAI、Anthropic、DeepSeek（httpx、流式、原生工具调用）；`RetryingProvider`。
- 持久化：`InMemoryStore`、`PostgresStore`（通用 schema）、`PostgresGrantStore`；带版本的检查点；租约；带恢复扫描的 `StoreWorker`。
- 持久化的工具调用意图：已决定的调用在执行前写入检查点，恢复后的 run 会重放这些调用。
- `memoflow_agent.testing.ScriptedModel`，用于离线测试。
