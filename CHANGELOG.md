# Changelog

[中文](CHANGELOG.zh-CN.md)

All notable changes are documented here. The format follows Keep a Changelog; versions follow SemVer.

## [Unreleased]

## [0.1.0] — 2026-10-14

First public release: the agent kernel extracted from MemoFlow AI Agent V4.

### Added
- `AgentLoop`: single model-driven loop (FINAL / TOOL_CALL / CLARIFICATION), budget wrap-up, invalid-output
  correction, cancellation, suspension, failure handling; `Application` hooks for embedding apps.
- `Agent`: sync and async run / stream / resume / cancel.
- Tools: `@tool`, `ToolSpec`, `ToolRegistry`, single `ToolExecutor` pipeline with hooks, retry policy,
  async timeouts, output cap, injection scan, idempotency records with reconciliation.
- Policy: `CapabilityPolicy`, principals, grants (once / session / persistent), approvals bound to
  arguments hash with expiry and replay protection.
- Context: token estimation, layered compaction, `ContextTooLong`.
- Providers: OpenAI, Anthropic, DeepSeek (httpx, streaming, native tool calling); `RetryingProvider`.
- Persistence: `InMemoryStore`, `PostgresStore` (generic schema), `PostgresGrantStore`; versioned
  checkpoints; leases; `StoreWorker` with recovery sweep.
- Durable tool intent: decided calls are checkpointed before execution so recovered runs replay them.
- `memoflow_agent.testing.ScriptedModel` for offline tests.

[Unreleased]: https://github.com/JasonWu0080/MemoFlow-Agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JasonWu0080/MemoFlow-Agent/releases/tag/v0.1.0
