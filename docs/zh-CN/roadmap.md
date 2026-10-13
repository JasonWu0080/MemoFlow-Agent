# 路线图

[English](../roadmap.md)

v0.1 刻意不包含（小内核原则）：多 Agent 编排与交接、可视化工作流构建器、控制台 / SaaS、内置 RAG 或向量数据库、
LLM 网关、完整的 MCP Server、浏览器自动化、默认的 Shell 或文件写入工具、计费平台。

v0.1 之后的候选（欢迎反馈）：
- MCP 客户端适配器：通过同一条 `ToolExecutor` 管线注册远程工具。
- OpenAI Responses API Provider；在能力声明支持时提供结构化输出辅助。
- `PostgresStore` 连接池；异步存储。
- 可插拔的分词器辅助。
