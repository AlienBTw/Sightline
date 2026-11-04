"""Event output: ring buffer of recent frames, MP4 clip writer, event folders + SQLite index.

Layout under data_dir (gitignored):
  events/<event_id>/clip.mp4           raw flagged clip (pre_s before .. post_s after)
  events/<event_id>/clip_circles.mp4   same clip with circles burned in
  events/<event_id>/thumb.jpg          peak frame with circles
  events/<event_id>/event.json         event record (camera, times, factors, scores, circles, escalation)
  sightline_events.db                  SQLite index (one row per event)
"""
from __future__ import annotations

import collections
import json
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path

import numpy as np


class RingBuffer:
    """Keeps the last `seconds` of (t, frame) pairs."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.buf: collections.deque = collections.deque()

    def push(self, t: float, img: np.ndarray):
        self.buf.append((t, img))
        while self.buf and t - self.buf[0][0] > self.seconds:
            self.buf.popleft()

    def since(self, t0: float) -> list:
        return [(t, im) for t, im in self.buf if t >= t0]


def resize_w(img: np.ndarray, width: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    if w <= width:
        return img
    nh = int(round(h * width / w)) // 2 * 2
    return cv2.resize(img, (width, nh), interpolation=cv2.INTER_AREA)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class ClipWriter:
    """H.264/yuv420p MP4 (+faststart, plays in browsers and Android) through an ffmpeg pipe;
    falls back to OpenCV mp4v if no ffmpeg binary is available."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], crf: int = 26, prefer_ffmpeg: bool = True):
        self.path, self.fps, self.size = Path(path), float(fps), size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = self.vw = None
        exe = _ffmpeg_exe() if prefer_ffmpeg else None
        w, h = size
        if exe:
            self.codec = "h264"
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{self.fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE)
        else:
            import cv2
            self.codec = "mp4v"
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def write(self, img: np.ndarray):
        if img.shape[1] != self.size[0] or img.shape[0] != self.size[1]:
            import cv2
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.vw.write(img)

    def close(self) -> str:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        elif self.vw is not None:
            self.vw.release()
        return self.codec


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  camera_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  stream_start_s REAL,
  stream_end_s REAL,
  risk_max REAL,
  top_factor TEXT,
  factors TEXT,
  status TEXT DEFAULT 'open',
  escalated INTEGER DEFAULT 0,
  dir TEXT NOT NULL,
  created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, started_at);
"""


class EventStore:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sightline_events.db"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def event_dir(self, event_id: str) -> Path:
        d = self.root / "events" / event_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def next_id(self, camera_id: str, stamp: str) -> str:
        """evt_<camera>_<YYYYmmddTHHMMSS>_<n> - unique and sortable."""
        base = f"evt_{camera_id}_{stamp}"
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM events WHERE event_id LIKE ?", (base + "%",)).fetchone()[0]
        return f"{base}_{n + 1:02d}"

    def save(self, rec: dict) -> Path:
        d = self.event_dir(rec["event_id"])
        with open(d / "event.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        esc = rec.get("escalation") or {}
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO events (event_id, camera_id, started_at, ended_at, stream_start_s, stream_end_s,"
                " risk_max, top_factor, factors, status, escalated, dir) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["event_id"], rec["camera_id"], rec["started_at"], rec.get("ended_at"), rec.get("stream_start_s"),
                 rec.get("stream_end_s"), rec.get("risk_max"), rec.get("top_factor"), ",".join(rec.get("factors", [])),
                 rec.get("status", "open"), int(bool(esc.get("calls"))), str(d)))
        return d

    def list(self, camera_id: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM events", []
        if camera_id:
            q, args = q + " WHERE camera_id = ?", [camera_id]
        q += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(q, args)]

    def get(self, event_id: str) -> dict | None:
        p = self.root / "events" / event_id / "event.json"
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    def set_status(self, event_id: str, status: str):
        with self._lock, self._conn() as c:
            c.execute("UPDATE events SET status = ? WHERE event_id = ?", (status, event_id))
        rec = self.get(event_id)
        if rec:
            rec["status"] = status
            with open(self.root / "events" / event_id / "event.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
