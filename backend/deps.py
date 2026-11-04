"""Shared app context and auth dependencies."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jwt
from fastapi import Depends, HTTPException, Request, WebSocket

from .settings import Settings


@dataclass
class User:
    sub: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class Ctx:
    settings: Settings
    db: object
    tokens: object
    limiter: object
    hub: object
    push: object
    manager: object
    store: object                     # sightline_pipeline.storage.EventStore
    started: float = field(default_factory=time.time)
    used_tickets: dict = field(default_factory=dict)


def ctx(request: Request) -> Ctx:
    return request.app.state.ctx


def client_ip(request: Request | WebSocket, settings: Settings) -> str:
    host = request.client.host if request.client else "unknown"
    if settings.trust_proxy and host in ("127.0.0.1", "::1", "localhost"):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()[:64]
    return host


def _unauth(detail="not authenticated"):
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def user_from_claims(c: Ctx, claims: dict) -> User:
    u = c.settings.users.get(claims.get("sub"))
    if u is None or u["role"] != claims.get("role", u["role"]):
        raise _unauth("user no longer exists")
    return User(claims["sub"], u["role"])


def _bearer(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    if h[:7].lower() == "bearer ":
        return h[7:].strip()
    return None


def current_user(request: Request, c: Ctx = Depends(ctx)) -> User:
    tok = _bearer(request)
    if not tok:
        raise _unauth()
    try:
        claims = c.tokens.decode(tok, "access")
    except jwt.ExpiredSignatureError:
        raise _unauth("token expired")
    except jwt.InvalidTokenError:
        raise _unauth("invalid token")
    return user_from_claims(c, claims)


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "the demo account cannot change cameras")
    return user


def media_user(request: Request, c: Ctx = Depends(ctx)) -> User:
    """Bearer access token, or a signed short-lived `st` media token scoped to this URL path
    (for <Image>/<Video> sources and the dev console's <video> tags)."""
    if _bearer(request):
        return current_user(request, c)
    st = request.query_params.get("st")
    if not st:
        raise _unauth()
    try:
        claims = c.tokens.decode(st, "media")
    except jwt.InvalidTokenError:
        raise _unauth("invalid or expired media link")
    scope = claims.get("scp") or ""
    if not scope or not request.url.path.startswith(scope):
        raise HTTPException(403, "media link not valid for this resource")
    u = c.settings.users.get(claims["sub"])
    if u is None:
        raise _unauth("user no longer exists")
    return User(claims["sub"], u["role"])


def signed(c: Ctx, user: User, scope: str, path: str) -> str:
    tok, _ = c.tokens.media(user.sub, scope)
    return f"{path}?st={tok}"
