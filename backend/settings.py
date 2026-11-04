"""Backend settings from environment variables (and the repo .env via python-dotenv).

Secrets are only ever read by env var name; nothing here prints or logs their values.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ROLES = ("admin", "demo")


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _path(v: str | None, default: Path) -> Path:
    if not v:
        return default
    p = Path(os.path.expanduser(v))
    return p if p.is_absolute() else REPO / p


@dataclass
class Settings:
    jwt_secret: str
    users: dict = field(default_factory=dict)          # username -> {"hash": str, "role": str}
    access_ttl_s: int = 900
    refresh_ttl_s: int = 7 * 24 * 3600
    media_ttl_s: int = 900
    backend_dir: Path = REPO / "data" / "backend"       # backend SQLite (cameras, tokens, devices)
    uploads_dir: Path = REPO / "data" / "uploads"
    events_dir: Path | None = None                       # None = the pipeline's storage.data_dir
    log_dir: Path | None = None
    cors_origins: list = field(default_factory=list)
    allowed_hosts: list = field(default_factory=list)
    trust_proxy: bool = True                             # ngrok forwards from 127.0.0.1 with X-Forwarded-For
    api_docs: bool = False
    # uploads
    upload_max_mb: int = 300
    upload_max_seconds: float = 600.0
    upload_min_seconds: float = 2.0
    upload_max_dim: int = 3840
    uploads_per_hour: int = 12
    upload_chunk_mb: int = 8
    max_virtual_cameras: int = 30
    upload_max_runtime_s: float = 1800.0                 # a looping upload camera stops itself after this
    # cameras / GPUs
    gpus: list = field(default_factory=lambda: ["cuda:0", "cuda:1"])
    cameras_per_gpu: int = 3
    autostart: bool = True                               # restart cameras that were running when the backend stopped
    seed_demo_camera: bool = True
    demo_video: Path = REPO / "data" / "demo" / "sightline_demo.mp4"
    pipeline_config: str | None = None
    # live preview
    preview_quality: int = 60
    mjpeg_max_fps: float = 5.0
    mjpeg_max_seconds: int = 300                         # one live-view session; the app reconnects on demand
    # push
    push_enabled: bool = True
    expo_access_token: str | None = None
    push_cooldown_s: float = 60.0
    expo_push_url: str = "https://exp.host/--/api/v2/push/send"
    # limits
    login_per_minute: int = 5
    login_per_hour: int = 30
    api_per_minute: int = 600
    ws_per_user: int = 5
    ws_total: int = 50

    @property
    def max_active_cameras(self) -> int:
        return max(1, len(self.gpus)) * max(1, self.cameras_per_gpu)

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file and not _bool(os.environ.get("SIGHTLINE_NO_DOTENV"), False):
            try:
                from dotenv import load_dotenv
                load_dotenv(REPO / ".env", override=False)
            except ImportError:
                pass
        e = os.environ.get
        secret = e("SIGHTLINE_JWT_SECRET", "")
        if len(secret) < 32:
            if _bool(e("SIGHTLINE_DEV"), False):
                secret = secrets.token_urlsafe(48)       # dev only: tokens die with the process
            else:
                raise RuntimeError("SIGHTLINE_JWT_SECRET is missing or shorter than 32 characters "
                                   "(generate one with: python -m backend.hash_password --secret)")
        users = {}
        for role, uvar, hvar in (("demo", "SIGHTLINE_DEMO_USER", "SIGHTLINE_DEMO_PASSWORD_HASH"),
                                 ("admin", "SIGHTLINE_ADMIN_USER", "SIGHTLINE_ADMIN_PASSWORD_HASH")):
            u, h = (e(uvar) or "").strip(), (e(hvar) or "").strip()
            if u and h:
                users[u] = {"hash": h, "role": role}
        csv = lambda v: [x.strip() for x in (v or "").split(",") if x.strip()]
        return cls(
            jwt_secret=secret, users=users,
            access_ttl_s=_int("SIGHTLINE_ACCESS_TTL_S", 900),
            refresh_ttl_s=_int("SIGHTLINE_REFRESH_TTL_S", 7 * 24 * 3600),
            media_ttl_s=_int("SIGHTLINE_MEDIA_TTL_S", 900),
            backend_dir=_path(e("SIGHTLINE_BACKEND_DIR"), REPO / "data" / "backend"),
            uploads_dir=_path(e("SIGHTLINE_UPLOAD_DIR"), REPO / "data" / "uploads"),
            events_dir=_path(e("SIGHTLINE_EVENTS_DIR"), None) if e("SIGHTLINE_EVENTS_DIR") else None,
            log_dir=_path(e("SIGHTLINE_LOG_DIR"), None) if e("SIGHTLINE_LOG_DIR") else None,
            cors_origins=csv(e("SIGHTLINE_CORS_ORIGINS")),
            allowed_hosts=csv(e("SIGHTLINE_ALLOWED_HOSTS")),
            trust_proxy=_bool(e("SIGHTLINE_TRUST_PROXY"), True),
            api_docs=_bool(e("SIGHTLINE_API_DOCS"), False),
            upload_max_mb=_int("SIGHTLINE_UPLOAD_MAX_MB", 300),
            upload_max_seconds=_float("SIGHTLINE_UPLOAD_MAX_SECONDS", 600.0),
            uploads_per_hour=_int("SIGHTLINE_UPLOADS_PER_HOUR", 12),
            max_virtual_cameras=_int("SIGHTLINE_MAX_VIRTUAL_CAMERAS", 30),
            upload_max_runtime_s=_float("SIGHTLINE_UPLOAD_MAX_RUNTIME_S", 1800.0),
            gpus=csv(e("SIGHTLINE_GPUS")) or ["cuda:0", "cuda:1"],
            cameras_per_gpu=_int("SIGHTLINE_CAMERAS_PER_GPU", 3),
            autostart=_bool(e("SIGHTLINE_AUTOSTART"), True),
            seed_demo_camera=_bool(e("SIGHTLINE_SEED_DEMO_CAMERA"), True),
            demo_video=_path(e("SIGHTLINE_DEMO_VIDEO"), REPO / "data" / "demo" / "sightline_demo.mp4"),
            pipeline_config=e("SIGHTLINE_PIPELINE_CONFIG") or None,
            push_enabled=_bool(e("SIGHTLINE_PUSH"), True),
            expo_access_token=e("EXPO_ACCESS_TOKEN") or None,
            push_cooldown_s=_float("SIGHTLINE_PUSH_COOLDOWN_S", 60.0),
            login_per_minute=_int("SIGHTLINE_LOGIN_PER_MINUTE", 5),
            api_per_minute=_int("SIGHTLINE_API_PER_MINUTE", 600),
            preview_quality=_int("SIGHTLINE_PREVIEW_QUALITY", 60),
            mjpeg_max_seconds=_int("SIGHTLINE_MJPEG_MAX_SECONDS", 300),
        )
