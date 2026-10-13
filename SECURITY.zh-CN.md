# 安全政策

[English](SECURITY.md)

## 报告漏洞

请**不要为安全问题开公开 issue**。请通过本仓库 GitHub 上的"Report a vulnerability"（私密安全公告）报告，或发送给维护者在仓库主页上列出的安全联系人。请附上问题描述、受影响版本，以及（如有可能）最小复现。

我们争取在 3 个工作日内确认收到，并与你商定披露时间表。

## 支持的版本

| 版本 | 是否支持 |
| --- | --- |
| 0.1.x | ✅ |

## 范围与已知限制

属于范围内：任何能让模型、工具结果或第三方绕过策略、审批、预算或 run 状态的问题；跨主体访问 run 或授权；工具 observation 泄露到日志。

已记录的限制（本身不算漏洞，但欢迎改进它们的报告）：

- 提示词注入扫描器（`memoflow_agent.safety.scan_for_injection`）是启发式的。
- 外发查询隐私闸门（`check_query_privacy`）检测不到短于其分片长度（默认 12 个字符）的私有片段。
- 应用启用 Provider 托管搜索时，无法逐条检查查询；框架只决定是否声明该工具。
- 崩溃后结果无法验证的副作用会报告为 `requires_reconciliation`，不会重试——不保证恰好一次。

完整的安全模型见 [docs/zh-CN/security.md](docs/zh-CN/security.md)。
