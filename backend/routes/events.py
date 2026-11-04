"""Events: list/filter/paginate (pipeline SQLite index), detail, status changes, authed media."""
from __future__ import annotations

import sqlite3
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as FPath
from pydantic import BaseModel

from sightline_pipeline.config import FACTOR_LABELS

from .. import media
from ..deps import Ctx, User, ctx, current_user, media_user, signed

router = APIRouter(prefix="/api/events", tags=["events"])
EVT_ID = r"^evt_[A-Za-z0-9_\-]{1,120}$"
STATUSES = ("open", "acknowledged", "dismissed", "fix_pending")
MEDIA = {"clip.mp4": "video/mp4", "clip_circles.mp4": "video/mp4", "thumb.jpg": "image/jpeg"}


def _conn(c: Ctx):
    con = sqlite3.connect(c.store.db_path, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def summary(c: Ctx, row: dict, user: User | None) -> dict:
    eid = row["event_id"]
    factors = row["factors"] if isinstance(row["factors"], list) else [f for f in (row.get("factors") or "").split(",") if f]
    out = {"event_id": eid, "camera_id": row["camera_id"], "started_at": row["started_at"],
           "ended_at": row.get("ended_at"), "risk_max": row.get("risk_max"), "top_factor": row.get("top_factor"),
           "top_factor_label": FACTOR_LABELS.get(row.get("top_factor") or "", None), "factors": factors,
           "factor_labels": {f: FACTOR_LABELS.get(f, f) for f in factors}, "status": row.get("status") or "open",
           "escalated": bool(row.get("escalated")) if "escalated" in row else bool((row.get("escalation") or {}).get("calls"))}
    verified = row.get("verified_by")
    if not verified:
        esc = row.get("escalation") if isinstance(row.get("escalation"), dict) else {}
        gates = esc.get("verify_gates") if isinstance(esc.get("verify_gates"), dict) else {}
        for fac in (gates.get("factors") or {}).values():
            if isinstance(fac, dict) and fac.get("verified_by"):
                verified = fac.get("verified_by")
                break
        if not verified:
            legacy = esc.get("f5_vlm_gate") if isinstance(esc.get("f5_vlm_gate"), dict) else None
            if isinstance(legacy, dict):
                verified = legacy.get("verified_by")
    if verified:
        out["verified_by"] = verified
    if user is not None:
        out["thumb_url"] = signed(c, user, f"/api/events/{eid}/", f"/api/events/{eid}/thumb.jpg")
    return out


@router.get("")
def list_events(camera_id: str | None = Query(None, max_length=50), status: str | None = Query(None),
                factor: str | None = Query(None, pattern=r"^F\d{1,2}$"), min_risk: float | None = Query(None, ge=0, le=1),
                since: str | None = Query(None, max_length=40), until: str | None = Query(None, max_length=40),
                limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0),
                c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    """Newest first. `since`/`until` are ISO timestamps compared to started_at."""
    where, args = [], []
    if camera_id:
        where.append("camera_id = ?"); args.append(camera_id)
    if status:
        sts = [s for s in status.split(",") if s in STATUSES]
        if not sts:
            raise HTTPException(422, f"status must be one of {', '.join(STATUSES)}")
        where.append(f"COALESCE(status, 'open') IN ({','.join('?' * len(sts))})"); args += sts
    if factor:
        where.append("(',' || factors || ',') LIKE ?"); args.append(f"%,{factor},%")
    if min_risk is not None:
        where.append("risk_max >= ?"); args.append(min_risk)
    if since:
        where.append("started_at >= ?"); args.append(since)
    if until:
        where.append("started_at < ?"); args.append(until)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    with _conn(c) as con:
        total = con.execute(f"SELECT COUNT(*) FROM events{w}", args).fetchone()[0]
        rows = [dict(r) for r in con.execute(f"SELECT * FROM events{w} ORDER BY started_at DESC, event_id DESC "
                                             f"LIMIT ? OFFSET ?", [*args, limit, offset])]
    items = [summary(c, r, user) for r in rows]
    nxt = offset + len(items)
    return {"items": items, "total": total, "limit": limit, "offset": offset,
            "next_offset": nxt if nxt < total else None}


def _row(c: Ctx, eid: str) -> dict:
    with _conn(c) as con:
        r = con.execute("SELECT * FROM events WHERE event_id = ?", (eid,)).fetchone()
    if r is None:
        raise HTTPException(404, "event not found")
    return dict(r)


@router.get("/{eid}")
def get_event(eid: str = FPath(pattern=EVT_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    row = _row(c, eid)
    rec = c.store.get(eid) or {}
    d = c.store.root / "events" / eid
    rec.pop("source", None)                      # local file paths / stream URLs stay on the server
    rec.update(summary(c, {**rec, **row}, user))
    scope = f"/api/events/{eid}/"
    rec["media"] = {k.split(".")[0]: {"url": signed(c, user, scope, scope + k), "ready": media.ready(d / k, 0.0 if k.endswith(".jpg") else 2.0)}
                    for k in MEDIA}
    rec["media"]["expires_in"] = c.tokens.media_ttl
    rec["actions"] = c.db.q("SELECT status, sub AS by, at FROM event_actions WHERE event_id = ? ORDER BY at", (eid,))
    return rec


class StatusIn(BaseModel):
    status: str


def _set_status(c: Ctx, eid: str, status: str, user: User) -> dict:
    if status not in STATUSES:
        raise HTTPException(422, f"status must be one of {', '.join(STATUSES)}")
    _row(c, eid)
    c.store.set_status(eid, status)
    c.db.x("INSERT INTO event_actions (event_id, status, sub, at) VALUES (?,?,?,?)", (eid, status, user.sub, time.time()))
    c.hub.publish({"type": "event_status", "event_id": eid, "status": status, "by": user.sub})
    return {"event_id": eid, "status": status}


@router.post("/{eid}/status")
def set_status(body: StatusIn, eid: str = FPath(pattern=EVT_ID), c: Ctx = Depends(ctx),
               user: User = Depends(current_user)):
    return _set_status(c, eid, body.status, user)


@router.post("/{eid}/ack")
def ack(eid: str = FPath(pattern=EVT_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    return _set_status(c, eid, "acknowledged", user)


@router.post("/{eid}/dismiss")
def dismiss(eid: str = FPath(pattern=EVT_ID), c: Ctx = Depends(ctx), user: User = Depends(current_user)):
    return _set_status(c, eid, "dismissed", user)


@router.api_route("/{eid}/{name}", methods=["GET", "HEAD"])
def event_media(request: Request, eid: str = FPath(pattern=EVT_ID),
                name: str = FPath(pattern=r"^(clip\.mp4|clip_circles\.mp4|thumb\.jpg)$"),
                c: Ctx = Depends(ctx), user: User = Depends(media_user)):
    """clip.mp4 / clip_circles.mp4 / thumb.jpg with HTTP Range (206) support."""
    p = c.store.root / "events" / eid / name
    if not p.is_file():
        raise HTTPException(404, "media not found")
    if not media.ready(p, 0.0 if name.endswith(".jpg") else 2.0):
        raise HTTPException(503, "clip is still being written", headers={"Retry-After": "2"})
    return media.file_response(request, p, MEDIA[name])
