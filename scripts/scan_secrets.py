"""Fail if any tracked file in the package looks like it contains a credential.

Prints file and line numbers only — never the matched value.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = {
    "openai_or_generic_sk_key": re.compile(r"\bsk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{24,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    "postgres_url_with_password": re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^:\s/]+:[^@\s]{3,}@(?!localhost|127\.0\.0\.1)"),
}
SKIP_DIRS = {".git", ".venv", "venv", "dist", "build", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
ALLOW = {("scripts/scan_secrets.py", "*")}


def main() -> int:
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts) or path.suffix in {".png", ".whl", ".gz"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if (relative, "*") in ALLOW:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{number}: {name}")
    for finding in findings:
        print(finding)
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
