# 参与贡献 MemoFlow Agent

[English](CONTRIBUTING.md)

感谢你的帮助。本项目看重**小内核**：在写较大的改动之前，请先开 issue 讨论，尤其是会增加依赖或新的公开 API 的改动。

## 开发环境

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 离线：不访问网络，不需要 API key
ruff check src tests examples
mypy
```

可选的测试套件：

| 套件 | 运行方式 |
| --- | --- |
| PostgreSQL 存储 | `MEMOFLOW_AGENT_TEST_POSTGRES_DSN=postgresql://… pytest tests/test_persistence_recovery.py`（使用一次性 schema） |
| 真实 Provider | `MEMOFLOW_AGENT_LIVE_TESTS=1`，加上 `MEMOFLOW_AGENT_LIVE_{OPENAI,ANTHROPIC,DEEPSEEK}_MODEL` 和对应的 API key。PR 永远不要求运行。 |

## 基本规则

- **只有一个循环。** 新行为要么进 `AgentLoop`，要么放在 `Application` 钩子后面——绝不新增第二个循环。
- **决策只来自原生工具调用。** 不用正则或解析模型文本来决定是否运行工具。
- **权限是确定性的。** 模型不能授予权限、批准操作或修改 run 状态。涉及安全的改动需要在 `tests/test_security.py` 中补测试。
- **每次工具调用都经过 `ToolExecutor`。** 不给"受信任"的工具开捷径。
- **不要在注释或文档里承诺无法验证的事**（例如恰好一次的副作用）。
- 每个改动都要有测试；示例必须持续通过（`tests/test_examples.py`）。
- 默认安装保持轻量依赖（`pydantic`、`httpx`）。

## Pull Request

1. Fork、建分支、写清楚提交信息。
2. `pytest`、`ruff`、`mypy` 全绿；在 `CHANGELOG.md` 的 *Unreleased* 下更新。
3. 提交贡献即表示你同意该贡献按 Apache License 2.0 授权（许可证第 5 节）。欢迎附上 DCO 签名（`git commit -s`）。

## 行为准则

参与本项目须遵守 [CODE_OF_CONDUCT.zh-CN.md](CODE_OF_CONDUCT.zh-CN.md)。
