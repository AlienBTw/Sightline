"""Hazard status transitions for fix-and-verify."""

from __future__ import annotations

from typing import Literal

from sightline.errors import SightlineError

Status = Literal[
    "candidate",
    "confirmed",
    "open",
    "fix_pending",
    "verified",
    "closed",
    "false_positive",
    "needs_human",
]

ALLOWED: dict[str, set[str]] = {
    "candidate": {"confirmed", "needs_human", "false_positive"},
    "confirmed": {"open", "needs_human", "false_positive"},
    "open": {"fix_pending", "needs_human", "false_positive"},
    "fix_pending": {"verified", "open", "false_positive"},
    "verified": {"closed"},
    "closed": set(),
    "false_positive": set(),
    "needs_human": {"open", "false_positive", "confirmed"},
}


def transition(current: str, new: str) -> str:
    allowed = ALLOWED.get(current)
    if allowed is None:
        raise SightlineError(
            error_code="hazard_status_unknown",
            component="hazards.state_machine",
            message=f"Unknown status {current!r}",
            hint="Use a defined hazard status",
        )
    if new not in allowed:
        raise SightlineError(
            error_code="hazard_transition_illegal",
            component="hazards.state_machine",
            message=f"Cannot transition {current} → {new}",
            hint=f"Allowed from {current}: {sorted(allowed)}",
        )
    return new
