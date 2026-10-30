"""Structured fail-loud errors for API and UI."""

from __future__ import annotations

from typing import Any


class SightlineError(Exception):
    """Raised when a configured dependency or step fails. Never swallowed into empty success."""

    def __init__(
        self,
        *,
        error_code: str,
        component: str,
        message: str,
        hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.error_code = error_code
        self.component = component
        self.message = message
        self.hint = hint
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "component": self.component,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }
