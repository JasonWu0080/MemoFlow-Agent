# Security Policy

[中文](SECURITY.zh-CN.md)

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately via GitHub's
"Report a vulnerability" (private security advisory) on this repository, or to the security contact
listed by the maintainers on the repository profile. Include a description, affected versions, and a
minimal reproduction if possible.

We aim to acknowledge reports within 3 business days and to agree on a disclosure timeline with you.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | ✅ |

## Scope and known limitations

In scope: anything that lets a model, a tool result or a third party bypass the policy, approvals,
budgets or run state; cross-principal access to runs or grants; leaks of tool observations into logs.

Documented limitations (not vulnerabilities by themselves, but reports that improve them are welcome):

- The prompt-injection scanner (`memoflow_agent.safety.scan_for_injection`) is a heuristic.
- The outbound-query privacy gate (`check_query_privacy`) does not detect private fragments shorter than
  its shingle size (12 characters by default).
- Provider-hosted search, when an application enables it, cannot be inspected per query; the framework
  only decides whether it is declared at all.
- Side effects whose outcome cannot be verified after a crash are reported as `requires_reconciliation`,
  not retried — exactly-once is not guaranteed.

See [docs/security.md](docs/security.md) for the full security model.
