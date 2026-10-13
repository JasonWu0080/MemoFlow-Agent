"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def stable_hash(value: Any) -> str:
    """SHA-256 of a canonical JSON encoding (sorted keys, UTF-8)."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)
