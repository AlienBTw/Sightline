"""POST /api/auth/login | refresh | logout, GET /api/auth/me."""
from __future__ import annotations

import logging

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..deps import Ctx, User, client_ip, ctx, current_user
from ..security import verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
log = logging.getLogger("sightline.auth")


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=2048)


def _pair(c: Ctx, sub: str, role: str, family: str | None = None) -> dict:
    at, _ = c.tokens.access(sub, role)
    rt, rc = c.tokens.refresh(sub, role, family)
    c.db.add_refresh(rc)
    return {"access_token": at, "refresh_token": rt, "token_type": "bearer", "expires_in": c.tokens.access_ttl,
            "refresh_expires_in": c.tokens.refresh_ttl, "user": {"username": sub, "role": role}}


def _limit(c: Ctx, key: str, n: int, window: float):
    ok, retry = c.limiter.allow(key, n, window)
    if not ok:
        raise HTTPException(429, "too many login attempts, try again later",
                            headers={"Retry-After": str(int(retry) + 1)})


@router.post("/login")
async def login(body: LoginIn, request: Request, c: Ctx = Depends(ctx)):
    ip = client_ip(request, c.settings)
    s = c.settings
    _limit(c, f"login:ip:{ip}", s.login_per_minute, 60)
    _limit(c, f"login:iph:{ip}", s.login_per_hour, 3600)
    _limit(c, f"login:user:{body.username.lower()}", 10, 900)
    u = s.users.get(body.username)
    ok = await run_in_threadpool(verify_password, body.password, u["hash"] if u else None)
    if not ok:
        log.warning("login failed ip=%s", ip)
        raise HTTPException(401, "invalid username or password")
    log.info("login ok user=%s ip=%s", body.username, ip)
    return _pair(c, body.username, u["role"])


@router.post("/refresh")
def refresh(body: RefreshIn, request: Request, c: Ctx = Depends(ctx)):
    ok, retry = c.limiter.allow(f"refresh:{client_ip(request, c.settings)}", 30, 60)
    if not ok:
        raise HTTPException(429, "too many requests", headers={"Retry-After": str(int(retry) + 1)})
    try:
        claims = c.tokens.decode(body.refresh_token, "refresh")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "invalid or expired refresh token")
    row = c.db.refresh_row(claims["jti"])
    if row is None:
        raise HTTPException(401, "unknown refresh token")
    if not c.db.revoke_refresh(claims["jti"]):
        c.db.revoke_family(row["family"])          # reuse of a rotated token: kill the whole session
        log.warning("refresh token reuse detected user=%s", claims["sub"])
        raise HTTPException(401, "refresh token already used; please log in again")
    u = c.settings.users.get(claims["sub"])
    if u is None:
        raise HTTPException(401, "user no longer exists")
    return _pair(c, claims["sub"], u["role"], family=row["family"])


@router.post("/logout")
def logout(body: RefreshIn, c: Ctx = Depends(ctx)):
    try:
        claims = c.tokens.decode(body.refresh_token, "refresh")
        row = c.db.refresh_row(claims["jti"])
        if row:
            c.db.revoke_family(row["family"])
    except jwt.InvalidTokenError:
        pass
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"username": user.sub, "role": user.role,
            "permissions": ["read", "events:status", "cameras:start_stop_virtual"] +
                           (["cameras:rtsp", "cameras:delete", "cameras:edit"] if user.is_admin else [])}
