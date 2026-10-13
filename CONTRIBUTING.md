# Contributing to MemoFlow Agent

[中文](CONTRIBUTING.zh-CN.md)

Thank you for helping. This project values a **small core**: please open an issue to discuss a feature
before writing a large change, especially anything that adds a dependency or a new public API.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # offline: no network, no API keys
ruff check src tests examples
mypy
```

Optional suites:

| Suite | How |
| --- | --- |
| PostgreSQL store | `MEMOFLOW_AGENT_TEST_POSTGRES_DSN=postgresql://… pytest tests/test_persistence_recovery.py` (uses a throwaway schema) |
| Live providers | `MEMOFLOW_AGENT_LIVE_TESTS=1` plus `MEMOFLOW_AGENT_LIVE_{OPENAI,ANTHROPIC,DEEPSEEK}_MODEL` and the matching API key. Never required for a PR. |

## Rules of the road

- **One loop.** New behaviour goes into `AgentLoop` or behind an `Application` hook — never a second loop.
- **Decisions come from native tool calling.** No regex or prose parsing decides whether a tool runs.
- **Authority is deterministic.** The model must not be able to grant permissions, approve actions or
  change run state. Security-relevant changes need a test in `tests/test_security.py`.
- **Every tool call goes through `ToolExecutor`.** No shortcut paths for "trusted" tools.
- **Don't promise what you can't verify** in code comments or docs (e.g. exactly-once side effects).
- Tests for every change; examples must keep passing (`tests/test_examples.py`).
- Keep the default install dependency-light (`pydantic`, `httpx`).

## Pull requests

1. Fork, branch, commit with clear messages.
2. `pytest`, `ruff`, `mypy` green; update `CHANGELOG.md` under *Unreleased*.
3. By submitting a contribution you agree it is licensed under the Apache License 2.0 (section 5 of the
   license). A DCO sign-off (`git commit -s`) is appreciated.

## Code of Conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
