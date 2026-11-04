"""One ingest path for every camera kind: RTSP / HTTP stream, video file, uploaded video.

Decoding is OpenCV's FFmpeg backend. Files are paced to real time by default so an uploaded
video behaves like a live camera (`realtime=False` processes as fast as possible, for eval).
Live streams reconnect with exponential backoff. A reader thread feeds a bounded queue; for
live/real-time sources the oldest frame is dropped when the consumer falls behind (like a
real camera), for offline sources the reader blocks so no frame is lost.

Real-camera behaviour (tested against MediaMTX + ffmpeg, see tools/camera_sim/):
  * RTSP over TCP by default (works through NAT/ngrok/Wi-Fi); `rtsp_transport="udp"` or "auto".
  * open/read timeouts (a dead camera or a stalled stream never blocks for long).
  * when an RTSP URL does not open, a tiny RTSP DESCRIBE probe says why (auth, wrong path,
    unreachable) in `last_error`; wrong credentials back off for minutes so the camera's
    account lock-out (Hikvision/Dahua) is not triggered.
  * a resolution change mid-session (camera reconfigured, sub/main stream swap) is scaled back
    to the session's first resolution so clip writers and the model window stay consistent.
  * `cancel` (e.g. the camera worker's stop event) ends the iteration even while reconnecting.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "srt://")
AUTH_BACKOFF_S = 120.0          # after a 401: wait this long before trying the same credentials again


def source_kind(uri: str) -> str:
    u = str(uri).lower()
    if u.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if u.startswith(LIVE_SCHEMES):
        return "stream"
    return "file"


def redact(uri: str) -> str:
    """Hide credentials in rtsp://user:pass@host URLs for logs and event records.
    Uses the LAST '@' of the authority, so passwords containing '@' are hidden completely."""
    if "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    cut = len(rest)
    for ch in "/?#":
        i = rest.find(ch)
        if i != -1:
            cut = min(cut, i)
    auth, tail = rest[:cut], rest[cut:]
    if "@" not in auth:
        return uri
    return f"{scheme}://***@{auth.rsplit('@', 1)[1]}{tail}"


@dataclass
class Frame:
    index: int          # frame counter since the source opened (keeps counting across reconnects/loops)
    t: float            # stream time in seconds (monotonic for the whole session)
    image: np.ndarray   # BGR uint8
    wall: float = 0.0   # time.time() when the frame was delivered


class Pacer:
    """Releases frames at their presentation time (scaled by `speed`), like a live camera.
    If the consumer is late it does not sleep (catches up) unless `max_lag_s` is exceeded,
    in which case the clock is re-anchored instead of bursting."""

    def __init__(self, speed: float = 1.0, max_lag_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.speed, self.max_lag_s, self.clock, self.sleep = speed, max_lag_s, clock, sleep
        self.t0_wall = None
        self.t0_media = 0.0

    def wait(self, media_t: float) -> float:
        """Block until media_t is due; returns the delay slept (negative = late)."""
        now = self.clock()
        if self.t0_wall is None:
            self.t0_wall, self.t0_media = now, media_t
            return 0.0
        due = self.t0_wall + (media_t - self.t0_media) / self.speed
        d = due - now
        if d > 0:
            self.sleep(d)
        elif -d > self.max_lag_s:
            self.t0_wall, self.t0_media = now, media_t
        return d


class Backoff:
    def __init__(self, base: float = 0.5, cap: float = 15.0):
        self.base, self.cap, self.n = base, cap, 0

    def next(self) -> float:
        d = min(self.cap, self.base * (2 ** self.n))
        self.n += 1
        return d

    def reset(self):
        self.n = 0


class VideoSource:
    """Iterate frames from any camera kind through one code path.

    uri:            rtsp://..., http(s)://... (MJPEG/HLS), or a file path (upload = file)
    realtime:       pace files at 1x (default True for files, ignored for live streams)
    loop:           restart a file at EOF (a virtual camera that never ends)
    reconnect:      re-open live streams on failure, with backoff
    rtsp_transport: "tcp" (default), "udp" or "auto" (FFmpeg tries UDP, then TCP)
    cancel:         optional external threading.Event; when set, iteration ends promptly
    state:          "connecting" | "streaming" | "reconnecting" | "stopped" | "failed"
    last_error:     latest human-readable reason (kept while retrying; `error` is set only on give-up)
    """

    def __init__(self, uri: str, realtime: bool | None = None, loop: bool = False, speed: float = 1.0,
                 reconnect: bool = True, max_reconnects: int = -1, open_fn=None, queue_size: int = 64,
                 rtsp_transport: str = "tcp", open_timeout_s: float = 8.0, read_timeout_s: float = 8.0,
                 cancel: threading.Event | None = None, diagnose_fn=None, auth_backoff_s: float = AUTH_BACKOFF_S):
        self.uri = str(uri)
        self.kind = source_kind(self.uri)
        self.live = self.kind != "file"
        self.realtime = (not self.live) if realtime is None else (realtime and not self.live)
        self.loop, self.speed = loop, speed
        self.reconnect, self.max_reconnects = reconnect, max_reconnects
        if rtsp_transport not in ("tcp", "udp", "auto"):
            raise ValueError("rtsp_transport must be tcp, udp or auto")
        self.rtsp_transport = rtsp_transport
        self.open_timeout_s, self.read_timeout_s = open_timeout_s, read_timeout_s
        if open_fn is None:
            self.open_fn = lambda u: _cv2_open(u, rtsp_transport, open_timeout_s, read_timeout_s)
            if diagnose_fn is None and self.kind == "rtsp":
                from .rtsp_probe import diagnose
                diagnose_fn = lambda u: diagnose(u, timeout=min(4.0, open_timeout_s))
        else:
            self.open_fn = open_fn
        self.diagnose_fn = diagnose_fn
        self.auth_backoff_s = auth_backoff_s
        self.fps = 0.0
        self.size = (0, 0)
        self.stats = {"frames": 0, "dropped": 0, "reconnects": 0, "decode_errors": 0, "resized_frames": 0,
                      "resolution_changes": 0}
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._cancel = cancel
        self._thread = None
        self.error: str | None = None
        self.last_error: str | None = None
        self.state = "connecting"

    def _stopped(self) -> bool:
        return self._stop.is_set() or (self._cancel is not None and self._cancel.is_set())

    def _wait(self, d: float):
        """Sleep `d` seconds in short slices so stop()/cancel interrupt a long backoff."""
        n = max(1, math.ceil(d / 0.2))
        for _ in range(n):
            if self._stopped():
                return
            time.sleep(d / n)

    def _open_failed_reason(self) -> tuple[str, bool]:
        """(message, is_auth_failure) for a live URL that did not open."""
        if self.diagnose_fn is None:
            return f"cannot open source {redact(self.uri)}", False
        try:
            d = self.diagnose_fn(self.uri)
        except Exception as e:     # diagnostics must never break ingest
            return f"cannot open source {redact(self.uri)} ({e!r})", False
        st = d.get("status")
        if st == "ok":
            return f"stream opened by RTSP but could not be decoded ({redact(self.uri)})", False
        return f"{d.get('detail') or st} [{st}] ({redact(self.uri)})", st == "auth"

    # ---- raw generator (no thread) -------------------------------------------------
    def _raw(self) -> Iterator[Frame]:
        idx, t_off, backoff = 0, 0.0, Backoff()
        pacer = Pacer(self.speed) if self.realtime else None
        last_t = 0.0
        out_size = None                       # (w, h) of the first frame of the session
        last_in = None                        # camera resolution of the previous frame
        while not self._stopped():
            cap = self.open_fn(self.uri)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                msg, is_auth = self._open_failed_reason() if self.live else (f"cannot open source {redact(self.uri)}", False)
                self.last_error = msg
                if self.live and self.reconnect and (self.max_reconnects < 0 or self.stats["reconnects"] < self.max_reconnects):
                    self.state = "reconnecting"
                    self.stats["reconnects"] += 1
                    d = backoff.next()
                    self._wait(max(d, self.auth_backoff_s) if is_auth else d)
                    continue
                self.error = msg
                self.state = "failed"
                return
            fps = cap.get(5) or 0.0           # CAP_PROP_FPS
            self.fps = fps if 1.0 <= fps <= 120.0 else 25.0
            self.size = (int(cap.get(3)), int(cap.get(4)))
            self.state = "streaming"
            n_this, t_live0 = 0, time.monotonic()
            while not self._stopped():
                ok, img = cap.read()
                if not ok or img is None:
                    break
                backoff.reset()
                h, w = img.shape[:2]
                if out_size is None:
                    out_size = (w, h)
                if (w, h) != last_in:
                    if last_in is not None:
                        self.stats["resolution_changes"] += 1
                    last_in = (w, h)
                    self.size = (w, h)
                if (w, h) != out_size:
                    import cv2
                    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
                    self.stats["resized_frames"] += 1
                if self.live:
                    t = t_off + (time.monotonic() - t_live0)
                else:
                    t = t_off + n_this / self.fps
                n_this += 1
                if pacer:
                    pacer.wait(t)
                last_t = t
                yield Frame(idx, t, img, time.time())
                idx += 1
            cap.release()
            t_off = last_t + 1.0 / (self.fps or 25.0)
            if self._stopped():
                break
            if not self.live:
                if self.loop and n_this > 0:
                    continue
                return
            self.last_error = "stream interrupted (no frame within the read timeout or the camera closed it)"
            if not self.reconnect or (0 <= self.max_reconnects <= self.stats["reconnects"]):
                self.error = "stream ended"
                self.state = "failed"
                return
            self.state = "reconnecting"
            self.stats["reconnects"] += 1
            self._wait(backoff.next())
        self.state = "stopped"

    # ---- threaded iteration ---------------------------------------------------------
    def _reader(self):
        drop = self.live or self.realtime
        try:
            for fr in self._raw():
                self.stats["frames"] += 1
                if drop:
                    while True:
                        try:
                            self._q.put_nowait(fr)
                            break
                        except queue.Full:
                            try:
                                self._q.get_nowait()
                                self.stats["dropped"] += 1
                            except queue.Empty:
                                pass
                else:
                    while not self._stopped():
                        try:
                            self._q.put(fr, timeout=0.2)
                            break
                        except queue.Full:
                            continue
        except Exception as e:  # never kill the consumer silently
            self.error = repr(e)
            self.state = "failed"
        finally:
            while True:                      # the end marker must get through even if the queue is full
                try:
                    self._q.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass

    def __iter__(self) -> Iterator[Frame]:
        self._thread = threading.Thread(target=self._reader, daemon=True, name="sightline-ingest")
        self._thread.start()
        while True:
            try:
                fr = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._stopped():          # e.g. camera stopped while the source is reconnecting
                    self._stop.set()
                    return
                continue
            if fr is None:
                return
            yield fr

    def stop(self):
        self._stop.set()


_OPEN_LOCK = threading.Lock()


def capture_options(uri: str, transport: str = "tcp") -> str:
    """OPENCV_FFMPEG_CAPTURE_OPTIONS for one open (FFmpeg demuxer options, key;value|key;value)."""
    if source_kind(uri) != "rtsp":
        return ""
    opts = []
    if transport in ("tcp", "udp"):
        opts.append(f"rtsp_transport;{transport}")
    # UDP: reorder buffer for late/lost RTP packets; both: keep latency low
    opts += ["fflags;nobuffer", "max_delay;500000"]
    if transport != "tcp":
        opts.append("reorder_queue_size;64")
    return "|".join(opts)


def _cv2_open(uri: str, transport: str = "tcp", open_timeout_s: float = 8.0, read_timeout_s: float = 8.0):
    """Open with per-camera FFmpeg options. OpenCV reads them from an env var at open time, so the
    variable is set under a lock (cameras open from several threads) and restored afterwards."""
    import cv2
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000),
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_s * 1000)]
    opts = capture_options(uri, transport)
    with _OPEN_LOCK:
        old = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        try:
            if opts:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG, params)
        finally:
            if opts:
                if old is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old
    return cap
