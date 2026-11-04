"""FastAPI app factory.

    uvicorn --factory backend.app:create_app --host 127.0.0.1 --port 8000 --no-access-log

Run from the repo root in WSL with ~/vjepa-venv (the real pipeline needs torch + ~/vjepa2).
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from . import __version__
from .cameras import CameraManager, RealPipelineFactory
from .db import DB
from .deps import Ctx, client_ip
from .hub import Hub
from .push import ExpoPush
from .routes import auth, cameras, events, misc
from .routes.events import summary
from .security import RateLimiter, Tokens
from .settings import Settings

STATIC = Path(__file__).resolve().parent / "static" / "console"
log = logging.getLogger("sightline.backend")

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
       "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")


def setup_logging(log_dir: Path | None):
    root = logging.getLogger("sightline")
    if getattr(root, "_sl_configured", False):
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    if not log_dir:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / "backend.log", maxBytes=10 * 1024 * 1024, backupCount=5,
                                                  encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root._sl_configured = True


def create_app(settings: Settings | None = None, factory=None, push_transport=None) -> FastAPI:
    s = settings or Settings.from_env()
    setup_logging(s.log_dir)
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.backend_dir.mkdir(parents=True, exist_ok=True)
    if factory is None:
        factory = RealPipelineFactory(s.pipeline_config)
    store = factory.store
    db = DB(s.backend_dir / "sightline_backend.db")
    hub = Hub()
    push = ExpoPush(db, s.expo_push_url, s.expo_access_token, s.push_enabled, s.push_cooldown_s, transport=push_transport)

    def on_event(rec):
        hub.publish({"type": "event", "event": summary(ctx_, rec, None)})
        push.notify(rec)

    manager = CameraManager(db, factory, s.gpus, s.cameras_per_gpu, on_event=on_event,
                            on_status=lambda cid, st: hub.publish({"type": "camera", "camera_id": cid,
                                                                   "state": st["state"], "error": st.get("last_error")}),
                            on_alert_state=lambda cid, f, inc: hub.publish({"type": "alert_state", "camera_id": cid,
                                                                            "factors": f, "incident": inc}))
    ctx_ = Ctx(settings=s, db=db, tokens=Tokens(s.jwt_secret, s.access_ttl_s, s.refresh_ttl_s, s.media_ttl_s),
               limiter=RateLimiter(), hub=hub, push=push, manager=manager, store=store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind(asyncio.get_running_loop())
        _seed_demo_camera(ctx_)
        if s.autostart:
            started = manager.autostart()
            if started:
                log.info("autostarted cameras: %s", started)
        log.info("backend up: version=%s users=%d (%s) gpus=%s cap=%d", __version__, len(s.users),
                 ",".join(sorted({u["role"] for u in s.users.values()})) or "none", s.gpus, manager.capacity)
        if not s.users:
            log.warning("no login configured: set SIGHTLINE_DEMO_USER + SIGHTLINE_DEMO_PASSWORD_HASH")
        yield
        log.info("shutting down: stopping camera workers")
        await asyncio.get_running_loop().run_in_executor(None, manager.shutdown, 25.0)
        push.close()

    app = FastAPI(title="Sightline API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs" if s.api_docs else None, redoc_url=None,
                  openapi_url="/api/openapi.json" if s.api_docs else None)
    app.state.ctx = ctx_

    @app.middleware("http")
    async def guard(request: Request, call_next):
        t0 = time.perf_counter()
        path = request.url.path
        if path.startswith("/api/"):
            ok, retry = ctx_.limiter.allow(f"api:{client_ip(request, s)}", s.api_per_minute, 60)
            if not ok:
                return JSONResponse({"detail": "rate limit"}, status_code=429, headers={"Retry-After": str(int(retry) + 1)})
        try:
            resp = await call_next(request)
        except Exception:
            log.exception("unhandled error on %s %s", request.method, path)
            resp = JSONResponse({"detail": "internal error"}, status_code=500)
        h = resp.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https":
            h.setdefault("Strict-Transport-Security", "max-age=31536000")
        if path.startswith("/console"):
            h.setdefault("Content-Security-Policy", CSP)
        if path.startswith("/api/") and h.get("content-type", "").startswith("application/json"):
            h.setdefault("Cache-Control", "no-store")
        # never log query strings: they carry signed media tokens / ws tickets
        ms = (time.perf_counter() - t0) * 1000
        if not (path.endswith("snapshot.jpg") and resp.status_code == 200):
            log.info("%s %s %s %.0fms ip=%s", request.method, path, resp.status_code, ms, client_ip(request, s))
        return resp

    if s.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=s.allowed_hosts)
    if s.cors_origins:   # the native app needs no CORS; only list extra web origins here
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=False,
                           allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                           allow_headers=["Authorization", "Content-Type", "X-Filename", "ngrok-skip-browser-warning"],
                           max_age=600)

    for r in (auth.router, cameras.router, events.router, misc.router):
        app.include_router(r)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/console/")

    @app.get("/console/", include_in_schema=False)
    def console():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/console/{name}", include_in_schema=False)
    def console_asset(name: str):
        if name not in ("app.js", "style.css"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})

    return app


def _seed_demo_camera(c: Ctx):
    """A ready virtual camera for judges (stopped by default; press start)."""
    s = c.settings
    if not s.seed_demo_camera or not s.demo_video.exists():
        return
    if c.db.camera("cam_demo_workshop") is None:
        c.db.add_camera({"id": "cam_demo_workshop", "name": "Demo workshop (sample video)", "kind": "virtual",
                         "uri": str(s.demo_video), "loop": True, "enabled": False,
                         "max_runtime_s": s.upload_max_runtime_s or None, "created_by": "seed",
                         "meta": {"video": {"note": "CC BY 4.0 test footage, see data/demo/ATTRIBUTION.md"}}})
