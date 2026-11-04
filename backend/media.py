"""File responses with HTTP Range support (Android ExoPlayer / expo-video seek with byte ranges)."""
from __future__ import annotations

import os
import re
import time
from email.utils import formatdate
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

CHUNK = 256 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, size: int):
    """-> None (no/ignored range: send 200), (start, end) inclusive, or 'unsatisfiable'."""
    if not header:
        return None
    header = header.strip()
    if "," in header:                       # multipart ranges: not supported, serve the whole file
        return None
    m = _RANGE.match(header)
    if not m:
        return None
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":                             # suffix: last N bytes
        n = int(b)
        if n == 0:
            return "unsatisfiable"
        return (max(0, size - n), size - 1)
    start = int(a)
    end = int(b) if b else size - 1
    if start >= size or end < start:
        return "unsatisfiable"
    return (start, min(end, size - 1))


def _iter(path: Path, start: int, length: int):
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left > 0:
            b = f.read(min(CHUNK, left))
            if not b:
                break
            left -= len(b)
            yield b


def ready(path: Path, settle_s: float = 2.0) -> bool:
    """A clip is ready when it exists, is non-empty and has not been written to for settle_s
    (clip_circles.mp4 is rendered in a background thread after the event is saved)."""
    try:
        st = path.stat()
    except OSError:
        return False
    return st.st_size > 0 and time.time() - st.st_mtime >= settle_s


def file_response(request: Request, path: Path, media_type: str, cache: str = "private, max-age=600") -> Response:
    st = os.stat(path)
    size = st.st_size
    etag = f'"{st.st_mtime_ns:x}-{size:x}"'
    headers = {"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": cache,
               "Last-Modified": formatdate(st.st_mtime, usegmt=True)}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    rng = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if if_range and if_range != etag:
        rng = None
    r = parse_range(rng, size)
    head = request.method == "HEAD"
    if r == "unsatisfiable":
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    if r is None:
        headers["Content-Length"] = str(size)
        if head:
            return Response(status_code=200, headers=headers, media_type=media_type)
        return StreamingResponse(_iter(path, 0, size), status_code=200, headers=headers, media_type=media_type)
    start, end = r
    length = end - start + 1
    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
    if head:
        return Response(status_code=206, headers=headers, media_type=media_type)
    return StreamingResponse(_iter(path, start, length), status_code=206, headers=headers, media_type=media_type)
