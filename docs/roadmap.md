# Roadmap

[中文](zh-CN/roadmap.md)

Not in v0.1 by design (small core): multi-agent orchestration and hand-off, visual workflow builder,
dashboard / SaaS, built-in RAG or vector database, LLM gateway, full MCP server, browser automation,
default shell or filesystem-write tools, billing platform.

Candidates after v0.1 (feedback welcome):
- MCP client adapter that registers remote tools through the same `ToolExecutor` pipeline.
- OpenAI Responses API provider; structured-output helpers where capabilities declare support.
- Connection pooling for `PostgresStore`; async stores.
- Pluggable tokenizer helpers.
