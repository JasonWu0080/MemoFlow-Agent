"""Heuristic safety checks: prompt-injection signals and query privacy.

**These are heuristics, not guarantees.** They catch common phrasings of
instructions planted in untrusted material and obvious copies of private
text in outbound queries. They reduce risk; they do not prove absence of
injection or leakage. Applications with stronger requirements plug in their
own scanner / gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("ignore_instructions", r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts?|rules)\b"),
        ("disregard_instructions", r"\bdisregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|system)\b"),
        ("role_override", r"\byou\s+are\s+now\s+(?:a|an|the)\b|\bact\s+as\s+(?:the\s+)?(?:system|developer|admin)\b"),
        ("system_prompt", r"\b(?:reveal|print|show)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions)\b"),
        ("fake_system_tag", r"<\s*/?\s*(?:system|developer)\s*>|\[\s*system\s*\]"),
        ("tool_coercion", r"\b(?:call|invoke|run)\s+the\s+\w+\s+tool\b.*\b(?:now|immediately)\b"),
        ("zh_ignore", r"忽略(?:之前|以上|上面|前面)(?:的)?(?:所有)?(?:指令|提示|规则)"),
        ("zh_role", r"你现在是|从现在起你是|扮演(?:系统|管理员|开发者)"),
        ("zh_system_prompt", r"(?:输出|显示|泄露)(?:你的)?系统提示"),
    )
)


def scan_for_injection(text: str) -> list[str]:
    """Names of injection patterns found in ``text`` (empty when none)."""
    if not text:
        return []
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


@dataclass(frozen=True)
class PrivacyVerdict:
    vetoed: bool
    reason_code: str | None = None


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)")
_SECRET = re.compile(r"\b(?:sk|pk|rk|ghp|xox[abprs])[-_][A-Za-z0-9_-]{12,}\b")


def _shingles(text: str, size: int) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    return {compact[index : index + size] for index in range(max(0, len(compact) - size + 1))}


def check_query_privacy(
    query: str,
    *,
    private_texts: list[str],
    user_texts: list[str] | tuple[str, ...] = (),
    shingle_size: int = 12,
) -> PrivacyVerdict:
    """Mechanical gate for a query about to leave the process.

    Vetoes queries containing personal data patterns, secrets, or any
    ``shingle_size``-character run copied from private material that does not
    also appear in the user's own words. Known limitation: private fragments
    shorter than ``shingle_size`` characters are not detected.
    """
    if _SECRET.search(query):
        return PrivacyVerdict(True, "secret")
    if _EMAIL.search(query) or _PHONE.search(query):
        return PrivacyVerdict(True, "personal_data")
    query_shingles = _shingles(query, shingle_size)
    if not query_shingles:
        return PrivacyVerdict(False)
    user_shingles: set[str] = set()
    for text in user_texts:
        user_shingles |= _shingles(text, shingle_size)
    for text in private_texts:
        leaked = (query_shingles & _shingles(text, shingle_size)) - user_shingles
        if leaked:
            return PrivacyVerdict(True, "private_content")
    return PrivacyVerdict(False)
