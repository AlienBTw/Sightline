"""Cameras: registry CRUD, start/stop, live preview."""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as FPath
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from sightline_pipeline.ingest import redact, source_kind

from ..cameras import CapacityError
from ..deps import Ctx, User, ctx, current_user, media_user, require_admin, signed
from ..preview import render_preview

router = APIRouter(prefix="/api", tags=["cameras"])
log = logging.getLogger("sightline.api.cameras")
CAM_ID = r"^cam_[a-z0-9_]{1,40}$"


# ------------------------------------------------------------------ helpers
def cam_out(c: Ctx, cam: dict, user: User) -> dict:
    cid = cam["id"]
    st = c.manager.status(cid)
    scope = f"/api/cameras/{cid}/"
    out = {"id": cid, "name": cam["name"], "kind": cam["kind"], "loop": cam["loop"], "enabled": cam["enabled"],
           "max_runtime_s": cam["max_runtime_s"], "created_at": cam["created_at"], "video": cam["meta"].get("video"),
           "status": st,
           "preview": {"snapshot_url": signed(c, user, scope, f"{scope}snapshot.jpg"),
                       "mjpeg_url": signed(c, user, scope, f"{scope}live.mjpeg"),
                       "expires_in": c.tokens.media_ttl}}
    if cam["kind"] != "virtual" and user.is_admin:
        out["source"] = redact(cam["uri"])          # credentials never leave the server
        out["rtsp_transport"] = cam["meta"].get("rtsp_transport", "tcp")
    return out


def _cam_or_404(c: Ctx, cid: str) -> dict:
    cam = c.db.camera(cid)
    if cam is None:
        raise HTTPException(404, "camera not found")
    return cam


def _new_id() -> str:
    return "cam_" + secrets.token_hex(3)


def _start(c: Ctx, cid: str) -> tuple[bool, str | None]:
    try:
        c.manager.start(cid)
        return True, None
    except CapacityError as e:
        return False, str(e)


# ------------------------------------------------------------------ registry
class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    uri: str = Field(min_length=8, max_length=500, description="rtsp://, rtsps://, http(s):// (MJPEG/HLS)")
    start: bool = True
    rtsp_transport: Literal["tcp", "udp", "auto"] = "tcp"


class CameraPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=60)
    uri: str | None = Field(None, min_length=8, max_length=500)
    loop: bool | None = None
    max_runtime_s: float | None = Field(None, ge=0, le=7 * 24 * 3600)


def _check_uri(uri: str) -> str:
    if any(ch.isspace() for ch in uri) or source_kind(uri) == "file":
        raise HTTPException(422, "uri must be an rtsp://, rtsps:// or http(s):// stream URL")
    return "rtsp" if source_kind(uri) == "rtsp" else "stream"


@router.get("/cameras")
def list_cameras(c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    return {"items": [cam_out(c, cam, user) for cam in c.db.cameras()],
            "capacity": {"active": c.manager.active_count(), "max": c.manager.capacity, "gpus": c.manager.gpus}}


@router.post("/cameras", status_code=201)
def add_camera(body: CameraIn, c: Ctx = Depends(ctx), user: User = Depends(require_admin)):
    kind = _check_uri(body.uri)
    cid = _new_id()
    c.db.add_camera({"id": cid, "name": body.name.strip(), "kind": kind, "uri": body.uri, "loop": False,
                     "enabled": False, "created_by": user.sub, "meta": {"rtsp_transport": body.rtsp_transport}})
    started, why = _start(c, cid) if body.start else (False, None)
    out = cam_out(c, c.db.camera(cid), user)
    out["started"], out["start_error"] = started, why
    c.hub.publish({"type": "camera_added", "camera_id": cid})
    return out


@router.get("/cameras/{cid}")
def get_camera(cid: str = FPath(pattern=CAM_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    return cam_out(c, _cam_or_404(c, cid), user)


@router.patch("/cameras/{cid}")
def patch_camera(body: CameraPatch, cid: str = FPath(pattern=CAM_ID), c: Ctx = Depends(ctx),
                 user: User = Depends(require_admin)):
    cam = _cam_or_404(c, cid)
    upd = body.model_dump(exclude_none=True)
    if "uri" in upd:
        if cam["kind"] == "virtual":
            raise HTTPException(422, "the source of a virtual camera is its file")
        _check_uri(upd["uri"])
    if "loop" in upd and cam["kind"] != "virtual":
        upd.pop("loop")
    if upd.get("max_runtime_s") == 0:
        upd["max_runtime_s"] = None
    c.db.update_camera(cid, **upd)
    return cam_out(c, c.db.camera(cid), user)


@router.delete("/cameras/{cid}")
def delete_camera(cid: str = FPath(pattern=CAM_ID), c: Ctx = Depends(ctx), user: User = Depends(require_admin)):
    cam = _cam_or_404(c, cid)
    c.manager.stop(cid, wait_s=10)
    c.db.delete_camera(cid)
    if cam["kind"] == "virtual":
        p = Path(cam["uri"]).resolve()
        if c.settings.uploads_dir.resolve() in p.parents and p.exists():
            p.unlink()                           # only a file stored under the upload dir, never the seeded demo video
    c.hub.publish({"type": "camera_removed", "camera_id": cid})
    return {"ok": True, "events_kept": True}


def _may_control(cam: dict, user: User):
    if not user.is_admin and cam["kind"] != "virtual":
        raise HTTPException(403, "the demo account can only start and stop the sample camera")


@router.post("/cameras/{cid}/start")
def start_camera(cid: str = FPath(pattern=CAM_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    cam = _cam_or_404(c, cid)
    _may_control(cam, user)
    ok, why = _start(c, cid)
    if not ok:
        raise HTTPException(409, why)
    return cam_out(c, c.db.camera(cid), user)


@router.post("/cameras/{cid}/stop")
def stop_camera(cid: str = FPath(pattern=CAM_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    cam = _cam_or_404(c, cid)
    _may_control(cam, user)
    c.manager.stop(cid)
    return cam_out(c, c.db.camera(cid), user)


# ------------------------------------------------------------------ live preview
def _worker_or_503(c: Ctx, cid: str):
    _cam_or_404(c, cid)
    w = c.manager.worker(cid)
    if w is None or w.latest is None:
        raise HTTPException(503, "camera is not running (start it first)", headers={"Retry-After": "2"})
    return w


@router.get("/cameras/{cid}/snapshot.jpg")
async def snapshot(cid: str = FPath(pattern=CAM_ID), w: int = Query(480, ge=160, le=1280), overlay: bool = True,
                   c: Ctx = Depends(ctx), user: User = Depends(media_user)):
    """Latest frame as JPEG with the live circles drawn. For camera-grid thumbnails: fetch once per
    screen visit, do not poll it (ngrok free = 20k requests/month); use live.mjpeg for live view."""
    wk = _worker_or_503(c, cid)
    cam = c.db.camera(cid)
    jpg, age = await run_in_threadpool(render_preview, wk, cam["name"], w, overlay, c.settings.preview_quality)
    return Response(jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Frame-Age": f"{age:.2f}",
                             "X-Active-Factors": ",".join(wk.active_factors)})


@router.get("/cameras/{cid}/live.mjpeg")
async def mjpeg(request: Request, cid: str = FPath(pattern=CAM_ID), fps: float = Query(1.0, gt=0, le=10),
                w: int = Query(480, ge=160, le=1280), overlay: bool = True, c: Ctx = Depends(ctx),
                user: User = Depends(media_user)):
    """Live view: ONE long HTTP response (multipart/x-mixed-replace MJPEG) with circles drawn
    server-side. Default 1 fps at 480 px (~15-25 KB/frame); it ends after mjpeg_max_seconds so an
    idle phone cannot drain the ngrok quota. Show it in react-native-webview as <img src=signed url>."""
    _worker_or_503(c, cid)
    fps = min(fps, c.settings.mjpeg_max_fps)
    cam = c.db.camera(cid)
    t_end = time.time() + c.settings.mjpeg_max_seconds

    async def gen():
        last = None
        while time.time() < t_end:
            if await request.is_disconnected():
                break
            wk = c.manager.worker(cid)
            if wk is None:
                break
            if wk.latest is not None and wk.latest[0] != last:
                last = wk.latest[0]
                jpg, _ = await run_in_threadpool(render_preview, wk, cam["name"], w, overlay, c.settings.preview_quality)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() +
                       b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(1.0 / fps)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ ONVIF (optional): add a camera without its RTSP URL
class OnvifDiscoverIn(BaseModel):
    timeout_s: float = Field(3.0, ge=0.5, le=10)
    hosts: list[str] = Field(default_factory=list, max_length=32, description="extra hosts to probe by unicast")


class OnvifDeviceIn(BaseModel):
    xaddr: str = Field(min_length=8, max_length=300, description="http://<ip>/onvif/device_service")
    username: str | None = Field(None, max_length=64)
    password: str | None = Field(None, max_length=128)


class OnvifAddIn(OnvifDeviceIn):
    profile_token: str | None = Field(None, max_length=64, description="default: the first (main) profile")
    name: str | None = Field(None, min_length=1, max_length=60)
    start: bool = True
    rtsp_transport: Literal["tcp", "udp", "auto"] = "tcp"


def _onvif_profiles(body: OnvifDeviceIn) -> dict:
    from .. import onvif
    try:
        return onvif.profiles(body.xaddr, body.username, body.password)
    except onvif.OnvifError as e:
        raise HTTPException(502, str(e))


@router.post("/onvif/discover")
def onvif_discover(body: OnvifDiscoverIn, user: User = Depends(require_admin)):
    """WS-Discovery on the backend's LAN (multicast 239.255.255.250:3702, plus unicast to `hosts`)."""
    from .. import onvif
    devs = onvif.discover(body.timeout_s, hosts=[h for h in body.hosts if h and not any(ch.isspace() for ch in h)])
    return {"items": [{k: d.get(k) for k in ("name", "hardware", "xaddrs", "endpoint", "from")} for d in devs]}


@router.post("/onvif/profiles")
def onvif_profiles(body: OnvifDeviceIn, user: User = Depends(require_admin)):
    return _onvif_profiles(body)


@router.post("/onvif/add", status_code=201)
def onvif_add(body: OnvifAddIn, c: Ctx = Depends(ctx), user: User = Depends(require_admin)):
    from .. import onvif
    info = _onvif_profiles(body)
    profs = [p for p in info["profiles"] if p.get("uri")]
    if body.profile_token:
        profs = [p for p in profs if p["token"] == body.profile_token]
    if not profs:
        raise HTTPException(404, "no ONVIF profile with an RTSP stream URI")
    p = profs[0]
    uri = onvif.with_credentials(p["uri"], body.username, body.password)
    name = body.name or f"{info['device'].get('Model') or 'ONVIF camera'} {p['name']}"[:60]
    out = add_camera(CameraIn(name=name, uri=uri, start=body.start, rtsp_transport=body.rtsp_transport), c, user)
    out["onvif"] = {"xaddr": info["xaddr"], "profile": p["token"], "video": p.get("video")}
    return out
