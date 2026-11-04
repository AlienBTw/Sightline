"""Health, push devices, real-time WebSocket."""
from __future__ import annotations

import asyncio
import shutil
import time

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sightline_pipeline.escalation import api_key

from .. import __version__
from ..deps import Ctx, User, ctx, current_user
from ..push import valid_token

router = APIRouter(tags=["ops"])


@router.get("/health")
def health(request: Request):
    """Public liveness for the watchdog / uptime monitor; numbers only, nothing sensitive."""
    c: Ctx = request.app.state.ctx
    ok = c.db.ping()
    sts = [c.manager.status(cam["id"])["state"] for cam in c.db.cameras()] if ok else []
    body = {"status": "ok" if ok else "degraded", "version": __version__, "time": time.time(),
            "uptime_s": round(time.time() - c.started),
            "cameras": {"running": sts.count("running"), "error": sts.count("error")}}
    return JSONResponse(body, status_code=200 if ok else 503, headers={"Cache-Control": "no-store"})


@router.get("/api/health")
def health_detail(c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    cams = {cam["id"]: c.manager.status(cam["id"]) for cam in c.db.cameras()}
    du = shutil.disk_usage(c.store.root)
    events_ok = True
    try:
        c.store.list(limit=1)
    except Exception:
        events_ok = False
    return {"status": "ok" if c.db.ping() and events_ok else "degraded", "version": __version__,
            "uptime_s": round(time.time() - c.started), "db": c.db.ping(), "events_db": events_ok,
            "disk_free_gb": round(du.free / 1024 ** 3, 1),
            "capacity": {"active": c.manager.active_count(), "max": c.manager.capacity, "gpus": c.manager.gpus,
                         "models_loaded": c.manager.pool.loaded},
            "cameras": cams if user.is_admin else {k: {"state": v["state"]} for k, v in cams.items()},
            "push": {"enabled": c.push.enabled, **({k: c.push.stats[k] for k in ("sent", "errors", "suppressed")})},
            "escalation_key_configured": bool(api_key()), "ws_clients": c.hub.count()}


# ------------------------------------------------------------------ push devices
class DeviceIn(BaseModel):
    token: str = Field(min_length=10, max_length=250)
    platform: str | None = Field("android", max_length=20)


@router.post("/api/devices", status_code=201)
def register_device(body: DeviceIn, c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    if not valid_token(body.token):
        raise HTTPException(422, "not an Expo push token (ExponentPushToken[...])")
    c.db.add_device(body.token, user.sub, body.platform)
    return {"ok": True}


@router.get("/api/devices")
def my_devices(c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    return {"items": [{"token": d["token"][:22] + "...", "platform": d["platform"], "last_seen": d["last_seen"]}
                      for d in c.db.devices(user.sub)]}


@router.delete("/api/devices")
def unregister_device(token: str = Query(min_length=10, max_length=250), c: Ctx = Depends(ctx),
                      user: User = Depends(current_user)):
    return {"ok": True, "removed": c.db.delete_device(token, user.sub)}


@router.post("/api/devices/test")
async def test_push(c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    ok, _ = c.limiter.allow(f"pushtest:{user.sub}", 3, 60)
    if not ok:
        raise HTTPException(429, "too many test pushes")
    devs = c.db.devices(user.sub)
    if not devs:
        raise HTTPException(404, "no registered device for this user")
    rec = {"event_id": "evt_test", "top_factor": "F1", "risk_max": 0.9, "camera_id": "test"}
    res = []
    for d in devs:
        r = await run_in_threadpool(c.push.client.post, c.push.url, json=[c.push.message(d["token"], rec)])
        res.append(r.status_code)
    return {"sent": len(devs), "http": res}


# ------------------------------------------------------------------ WebSocket
@router.post("/api/ws/ticket")
def ws_ticket(c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    """One-time 60 s ticket so the access token never goes into a URL."""
    t, _ = c.tokens.ws_ticket(user.sub, user.role)
    return {"ticket": t, "expires_in": 60, "url": f"/api/ws?ticket={t}"}


@router.websocket("/api/ws")
async def ws(websocket: WebSocket, ticket: str = Query("")):
    c: Ctx = websocket.app.state.ctx
    try:
        claims = c.tokens.decode(ticket, "ws")
    except jwt.InvalidTokenError:
        await websocket.close(code=4401)
        return
    now = time.time()
    for k in [k for k, exp in c.used_tickets.items() if exp < now]:
        c.used_tickets.pop(k, None)
    if claims["jti"] in c.used_tickets or claims["sub"] not in c.settings.users:
        await websocket.close(code=4401)
        return
    c.used_tickets[claims["jti"]] = claims["exp"]
    if c.hub.count(claims["sub"]) >= c.settings.ws_per_user or c.hub.count() >= c.settings.ws_total:
        await websocket.close(code=4429)
        return
    await websocket.accept()
    cid, q = c.hub.add(claims["sub"])
    try:
        await websocket.send_json({"type": "hello", "user": claims["sub"],
                                   "cameras": [{"camera_id": cam["id"], **{k: v for k, v in c.manager.status(cam["id"]).items()
                                                                           if k in ("state", "active_factors")}}
                                               for cam in c.db.cameras()]})

        async def reader():
            while True:
                m = await websocket.receive_text()
                if m == "ping":
                    await websocket.send_text('{"type":"pong"}')

        rd = asyncio.create_task(reader())
        try:
            while not rd.done():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    await websocket.send_text('{"type":"ping"}')   # keeps ngrok / mobile NAT alive
        finally:
            rd.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        c.hub.remove(cid)
