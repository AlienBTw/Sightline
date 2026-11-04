"""Backend SQLite (cameras, refresh tokens, push devices, uploads, event actions).

The events themselves live in the pipeline's index (`sightline_events.db` + event folders);
this database only holds what the backend owns.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('rtsp', 'stream', 'virtual')),
  uri TEXT NOT NULL,
  loop INTEGER NOT NULL DEFAULT 0,
  enabled INTEGER NOT NULL DEFAULT 0,
  max_runtime_s REAL,
  created_by TEXT,
  created_at REAL NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
  jti TEXT PRIMARY KEY,
  sub TEXT NOT NULL,
  family TEXT NOT NULL,
  exp INTEGER NOT NULL,
  revoked INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rt_family ON refresh_tokens(family);
CREATE TABLE IF NOT EXISTS devices (
  token TEXT PRIMARY KEY,
  sub TEXT NOT NULL,
  platform TEXT,
  created_at REAL NOT NULL,
  last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY,
  sub TEXT NOT NULL,
  filename TEXT NOT NULL,
  ext TEXT NOT NULL,
  size INTEGER NOT NULL,
  received INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS event_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  status TEXT NOT NULL,
  sub TEXT NOT NULL,
  at REAL NOT NULL
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._c = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._c.row_factory = sqlite3.Row
        with self._lock:
            self._c.executescript(SCHEMA)

    def q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._c.execute(sql, args)]

    def one(self, sql: str, args=()) -> dict | None:
        r = self.q(sql, args)
        return r[0] if r else None

    def x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._c.execute(sql, args)
            self._c.commit()
            return cur.rowcount

    def ping(self) -> bool:
        try:
            self.q("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    # ---- cameras
    def cameras(self) -> list[dict]:
        return [_cam(r) for r in self.q("SELECT * FROM cameras ORDER BY created_at")]

    def camera(self, cid: str) -> dict | None:
        r = self.one("SELECT * FROM cameras WHERE id = ?", (cid,))
        return _cam(r) if r else None

    def add_camera(self, cam: dict):
        self.x("INSERT INTO cameras (id, name, kind, uri, loop, enabled, max_runtime_s, created_by, created_at, meta)"
               " VALUES (?,?,?,?,?,?,?,?,?,?)",
               (cam["id"], cam["name"], cam["kind"], cam["uri"], int(cam.get("loop", 0)), int(cam.get("enabled", 0)),
                cam.get("max_runtime_s"), cam.get("created_by"), time.time(), json.dumps(cam.get("meta") or {})))

    def update_camera(self, cid: str, **kw):
        cols = {k: v for k, v in kw.items() if k in ("name", "uri", "loop", "enabled", "max_runtime_s", "meta")}
        if "meta" in cols:
            cols["meta"] = json.dumps(cols["meta"])
        if cols:
            self.x(f"UPDATE cameras SET {', '.join(k + ' = ?' for k in cols)} WHERE id = ?", (*cols.values(), cid))

    def delete_camera(self, cid: str):
        self.x("DELETE FROM cameras WHERE id = ?", (cid,))

    # ---- refresh tokens
    def add_refresh(self, c: dict):
        self.x("INSERT INTO refresh_tokens (jti, sub, family, exp, created_at) VALUES (?,?,?,?,?)",
               (c["jti"], c["sub"], c["fam"], c["exp"], time.time()))
        self.x("DELETE FROM refresh_tokens WHERE exp < ?", (int(time.time()) - 86400,))

    def refresh_row(self, jti: str) -> dict | None:
        return self.one("SELECT * FROM refresh_tokens WHERE jti = ?", (jti,))

    def revoke_refresh(self, jti: str) -> bool:
        """Atomically revoke one token; False if it was already revoked (reuse)."""
        return self.x("UPDATE refresh_tokens SET revoked = 1 WHERE jti = ? AND revoked = 0", (jti,)) == 1

    def revoke_family(self, family: str):
        self.x("UPDATE refresh_tokens SET revoked = 1 WHERE family = ?", (family,))

    # ---- devices
    def add_device(self, token: str, sub: str, platform: str | None, cap: int = 10):
        now = time.time()
        self.x("INSERT INTO devices (token, sub, platform, created_at, last_seen) VALUES (?,?,?,?,?) "
               "ON CONFLICT(token) DO UPDATE SET sub = excluded.sub, platform = excluded.platform, last_seen = excluded.last_seen",
               (token, sub, platform, now, now))
        extra = self.q("SELECT token FROM devices WHERE sub = ? ORDER BY last_seen DESC LIMIT -1 OFFSET ?", (sub, cap))
        for r in extra:
            self.x("DELETE FROM devices WHERE token = ?", (r["token"],))

    def devices(self, sub: str | None = None) -> list[dict]:
        if sub:
            return self.q("SELECT * FROM devices WHERE sub = ? ORDER BY last_seen DESC", (sub,))
        return self.q("SELECT * FROM devices ORDER BY last_seen DESC")

    def delete_device(self, token: str, sub: str | None = None) -> int:
        if sub:
            return self.x("DELETE FROM devices WHERE token = ? AND sub = ?", (token, sub))
        return self.x("DELETE FROM devices WHERE token = ?", (token,))


def _cam(r: dict) -> dict:
    r = dict(r)
    r["loop"], r["enabled"] = bool(r["loop"]), bool(r["enabled"])
    r["meta"] = json.loads(r.get("meta") or "{}")
    return r
