"""Permission policy: may this tool call run, here, now, for this principal?

Evaluated by the executor before every handler. The model's arguments are
inputs to the decision, never authority for it — a model asking for a wider
scope gets an authorization request, not access.

Outcomes:

* :class:`Allow` — run the handler (optionally with a resolved ``scope``);
* :class:`Deny` — tell the model why, so it can answer without the tool;
* :class:`RequireAuthorization` — the principal lacks a *grant* for a class of
  resources or a capability; suspend and ask the user;
* :class:`RequireConfirmation` — the principal may do this kind of thing, but
  this *one* operation, with these exact arguments, needs a human's approval.

Authorization and confirmation are deliberately different: a grant outlives
one call (once / session / persistent), an approval is bound to one tool call
and the hash of its arguments, and is consumed when used.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from .exceptions import InvalidResume
from .util import stable_hash, utcnow


@dataclass(frozen=True)
class Allow:
    scope: Any = None


@dataclass(frozen=True)
class Deny:
    code: str
    message: str


@dataclass(frozen=True)
class RequireAuthorization:
    resource: str
    reason: str = ""


@dataclass(frozen=True)
class RequireConfirmation:
    reason: str = ""
    scope: Any = None


PolicyOutcome = Allow | Deny | RequireAuthorization | RequireConfirmation


class Policy(Protocol):
    def evaluate(self, spec: Any, arguments: Any, ctx: Any) -> PolicyOutcome: ...


class AllowAll:
    """Allows every offered tool. Only sensible for read-only demos."""

    def evaluate(self, spec: Any, arguments: Any, ctx: Any) -> PolicyOutcome:
        return Allow()


# --- principals and grants ------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Who a run acts for. Comes from the application's authentication, never the model."""

    id: str
    # Capabilities held unconditionally (e.g. from the user's role).
    capabilities: frozenset[str] = frozenset()


class GrantDuration(StrEnum):
    ONCE = "once"  # this run only
    SESSION = "session"  # this session / conversation, until revoked
    PERSISTENT = "persistent"  # until revoked


@dataclass(frozen=True)
class Grant:
    id: str
    principal_id: str
    capability: str
    duration: GrantDuration
    run_id: Any = None
    session_id: Any = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    def applies(self, *, principal_id: str, capability: str, run_id: Any, session_id: Any, now: datetime) -> bool:
        if self.revoked_at is not None or self.principal_id != principal_id or self.capability != capability:
            return False
        if self.expires_at is not None and self.expires_at <= now:
            return False
        if self.duration is GrantDuration.ONCE:
            return self.run_id is not None and self.run_id == run_id
        if self.duration is GrantDuration.SESSION:
            return self.session_id is not None and self.session_id == session_id
        return True


class GrantStore(Protocol):
    def add(self, grant: Grant) -> Grant: ...

    def revoke(self, grant_id: str) -> bool: ...

    def active(self, *, principal_id: str, capability: str, run_id: Any, session_id: Any) -> bool: ...

    def list(self, principal_id: str) -> list[Grant]: ...


class InMemoryGrantStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._grants: dict[str, Grant] = {}

    def add(self, grant: Grant) -> Grant:
        with self._lock:
            self._grants[grant.id] = grant
        return grant

    def revoke(self, grant_id: str) -> bool:
        with self._lock:
            grant = self._grants.get(grant_id)
            if grant is None or grant.revoked_at is not None:
                return False
            self._grants[grant_id] = replace(grant, revoked_at=utcnow())
            return True

    def active(self, *, principal_id: str, capability: str, run_id: Any, session_id: Any) -> bool:
        now = utcnow()
        with self._lock:
            grants = list(self._grants.values())
        return any(
            grant.applies(principal_id=principal_id, capability=capability, run_id=run_id, session_id=session_id, now=now)
            for grant in grants
        )

    def list(self, principal_id: str) -> list[Grant]:
        with self._lock:
            return [grant for grant in self._grants.values() if grant.principal_id == principal_id]


def new_grant(
    *,
    principal_id: str,
    capability: str,
    duration: GrantDuration | str,
    run_id: Any = None,
    session_id: Any = None,
    once_ttl: timedelta = timedelta(days=3),
) -> Grant:
    duration = GrantDuration(duration)
    return Grant(
        id=uuid.uuid4().hex,
        principal_id=principal_id,
        capability=capability,
        duration=duration,
        # Bound only to what the duration names.
        run_id=run_id if duration is GrantDuration.ONCE else None,
        session_id=session_id if duration is GrantDuration.SESSION else None,
        # A one-run grant cannot outlive a forgotten suspended run.
        expires_at=utcnow() + once_ttl if duration is GrantDuration.ONCE else None,
    )


@dataclass
class CapabilityPolicy:
    """A capability-based policy that covers most applications.

    * A tool with ``permission=None`` needs no capability.
    * A capability is held when the principal has it statically, or an active
      grant exists — re-read on **every** call, so a revoked grant stops the
      very next call.
    * Missing capabilities listed in ``grantable`` are requested from the user
      (``RequireAuthorization``) in interactive runs; others are denied.
    * Tools with ``requires_approval`` (or whose risk is in
      ``approve_risks`` and that have side effects) need a confirmation of the
      exact call.
    * Once untrusted material tried to instruct the assistant, tools with
      side effects are refused for the rest of the run (``lock_on_injection``).
    """

    grants: GrantStore = field(default_factory=InMemoryGrantStore)
    grantable: frozenset[str] = frozenset()
    approve_risks: frozenset[str] = frozenset()
    lock_on_injection: bool = True

    def evaluate(self, spec: Any, arguments: Any, ctx: Any) -> PolicyOutcome:
        state = ctx.state
        principal: Principal | None = getattr(ctx, "principal", None)
        effect = spec.side_effect
        side_effecting = getattr(effect, "needs_idempotency_key", False)
        if side_effecting and state.pending is not None:
            return Deny("pending_action", "Wait for the user to answer the previous request.")
        if side_effecting and self.lock_on_injection and state.security_flags.get("injection_detected"):
            return Deny(
                "writes_blocked_untrusted_material",
                "Material read in this run contained instructions aimed at the assistant, so "
                "actions with side effects are disabled for this request. Tell the user.",
            )
        capability = spec.permission
        if capability:
            held = principal is not None and (
                capability in principal.capabilities
                or self.grants.active(
                    principal_id=principal.id,
                    capability=capability,
                    run_id=getattr(ctx, "run_id", None),
                    session_id=getattr(ctx, "session_id", None),
                )
            )
            if not held:
                if capability in self.grantable and principal is not None and state.interactive:
                    return RequireAuthorization(resource=capability, reason=f"{spec.name} needs {capability}")
                return Deny("not_authorized", f"You are not allowed to use {spec.name} here. Answer without it.")
        needs_approval = spec.requires_approval or (
            str(spec.risk) in self.approve_risks and str(effect) != "read_only"
        )
        if needs_approval:
            if not state.interactive:
                return Deny("approval_unavailable", "This action needs the user's approval, which this client cannot show.")
            return RequireConfirmation(reason=f"{spec.name} needs approval")
        return Allow()


# --- approvals --------------------------------------------------------------------


def new_approval(
    *,
    run_id: Any,
    tool_call_id: str,
    tool_name: str,
    arguments: dict,
    principal_id: str | None,
    ttl: timedelta = timedelta(hours=24),
) -> dict:
    """The approval record a suspended run stores in its pending action."""
    return {
        "approval_id": uuid.uuid4().hex,
        "run_id": str(run_id),
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "arguments_hash": stable_hash(arguments),
        "principal_id": principal_id,
        "expires_at": (utcnow() + ttl).isoformat(),
        "status": "pending",
    }


def validate_approval(
    approval: dict | None,
    *,
    approval_id: str,
    principal_id: str | None,
    tool_call_id: str | None,
    arguments: dict,
    now: datetime | None = None,
) -> None:
    """Refuse an approval that is unknown, used, expired, for someone else or tampered with."""
    if not approval or approval.get("approval_id") != approval_id:
        raise InvalidResume("approval_mismatch", "That approval does not belong to this run's pending action.")
    if approval.get("status") != "pending":
        raise InvalidResume("approval_used", "That approval was already used.")
    if approval.get("principal_id") is not None and approval.get("principal_id") != principal_id:
        raise InvalidResume("approval_principal", "Only the principal the run acts for may approve it.")
    if approval.get("tool_call_id") != tool_call_id:
        raise InvalidResume("approval_mismatch", "Approval is for another tool call.")
    if approval.get("arguments_hash") != stable_hash(arguments):
        raise InvalidResume("approval_tampered", "The arguments changed after the approval was requested.")
    expires = approval.get("expires_at")
    if expires and datetime.fromisoformat(expires) <= (now or utcnow()):
        raise InvalidResume("approval_expired", "That approval has expired.")


def capabilities(*names: str | Iterable[str]) -> frozenset[str]:
    flat: list[str] = []
    for name in names:
        flat.extend([name] if isinstance(name, str) else list(name))
    return frozenset(flat)
