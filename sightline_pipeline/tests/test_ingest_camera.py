"""Real-camera behaviour of the ingest path: credentials, RTSP diagnostics, cancel, resolution change."""
import socket
import sys
import threading
import time

import numpy as np
import pytest

from sightline_pipeline.ingest import VideoSource, capture_options, redact
from sightline_pipeline.rtsp_probe import diagnose, split_credentials


class Cap:
    def __init__(self, sizes=(), fail_open=False):
        self.sizes, self.ok = list(sizes), not fail_open
        self.wh = self.sizes[0] if self.sizes else (64, 48)

    def isOpened(self):
        return self.ok

    def get(self, prop):
        return {5: 25.0, 3: self.wh[0], 4: self.wh[1]}.get(prop, 0)   # like OpenCV: the real size

    def read(self):
        if not self.sizes:
            return False, None
        w, h = self.sizes.pop(0)
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        pass


def test_redact_hides_passwords_with_at_sign_and_keeps_path():
    assert redact("rtsp://admin:p@ss@10.0.0.9:554/Streams/101") == "rtsp://***@10.0.0.9:554/Streams/101"
    assert redact("rtsp://cam:554/a@b") == "rtsp://cam:554/a@b"           # '@' in the path is not a credential
    assert redact("http://u:p@h/x?y=1") == "http://***@h/x?y=1"


def test_split_credentials_decodes_percent_escapes():
    clean, u, p = split_credentials("rtsp://admin:p%40ss@10.0.0.9:554/Streams/101")
    assert (clean, u, p) == ("rtsp://10.0.0.9:554/Streams/101", "admin", "p@ss")


def test_capture_options_per_transport():
    assert "rtsp_transport;tcp" in capture_options("rtsp://h/x", "tcp")
    assert "rtsp_transport;udp" in capture_options("rtsp://h/x", "udp")
    assert "rtsp_transport" not in capture_options("rtsp://h/x", "auto")
    assert capture_options("http://h/x.mjpg") == ""
    with pytest.raises(ValueError):
        VideoSource("rtsp://h/x", rtsp_transport="quic")


def test_auth_failure_is_reported_and_backs_off_long(monkeypatch):
    import sightline_pipeline.ingest as ing
    waits = []
    monkeypatch.setattr(VideoSource, "_wait", lambda self, d: waits.append(d))
    src = VideoSource("rtsp://admin:bad@cam/Streams/101", open_fn=lambda u: Cap(fail_open=True), max_reconnects=2,
                      diagnose_fn=lambda u: {"status": "auth", "detail": "authentication failed"}, auth_backoff_s=120)
    assert list(src) == []
    assert waits == [120, 120]                         # minutes, not seconds: no account lock-out
    assert "authentication failed" in src.error and "bad" not in src.error and src.state == "failed"


def test_unreachable_reason_kept_while_retrying(monkeypatch):
    monkeypatch.setattr(VideoSource, "_wait", lambda self, d: None)
    src = VideoSource("rtsp://10.255.0.1/live", open_fn=lambda u: Cap(fail_open=True), max_reconnects=1,
                      diagnose_fn=lambda u: {"status": "unreachable", "detail": "cannot reach 10.255.0.1:554"})
    list(src)
    assert "unreachable" in src.last_error and "unreachable" in src.error


def test_cancel_ends_iteration_while_reconnecting():
    """A camera that never opens must still be stoppable (the worker thread used to hang in queue.get)."""
    cancel = threading.Event()
    src = VideoSource("rtsp://cam/x", open_fn=lambda u: Cap(fail_open=True), cancel=cancel,
                      diagnose_fn=lambda u: {"status": "refused", "detail": "refused"})
    done = threading.Event()

    def consume():
        list(src)
        done.set()
    th = threading.Thread(target=consume, daemon=True)
    th.start()
    time.sleep(0.5)
    assert not done.is_set() and src.state == "reconnecting"
    cancel.set()
    assert done.wait(3.0)


def test_resolution_change_is_scaled_to_session_size(monkeypatch):
    monkeypatch.setattr(VideoSource, "_wait", lambda self, d: None)
    caps = [Cap([(64, 48)] * 3), Cap([(32, 24)] * 2)]   # reconnect comes back as the sub-stream size
    src = VideoSource("rtsp://cam/x", open_fn=lambda u: caps.pop(0) if caps else Cap(fail_open=True),
                      max_reconnects=2, diagnose_fn=lambda u: {"status": "refused"})
    fr = list(src)
    assert len(fr) == 5 and all(f.image.shape == (48, 64, 3) for f in fr)
    assert src.stats["resolution_changes"] == 1 and src.stats["resized_frames"] == 2


def _fake_rtsp_server(responses):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    seen = []

    def run():
        conn, _ = srv.accept()
        for resp in responses:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(4096)
            seen.append(data.decode())
            conn.sendall(resp.encode())
        conn.close()
        srv.close()
    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1], seen


DIGEST_401 = 'RTSP/1.0 401 Unauthorized\r\nCSeq: {}\r\nWWW-Authenticate: Digest realm="IPCAM", nonce="abc123"\r\n\r\n'


def test_diagnose_digest_auth_ok_and_wrong_password():
    port, seen = _fake_rtsp_server([DIGEST_401.format(1), "RTSP/1.0 200 OK\r\nCSeq: 2\r\n\r\n"])
    r = diagnose(f"rtsp://admin:good@127.0.0.1:{port}/Streams/101", timeout=2)
    assert r["status"] == "ok"
    assert 'Digest username="admin", realm="IPCAM", nonce="abc123"' in seen[1] and "good" not in seen[1]
    port, _ = _fake_rtsp_server([DIGEST_401.format(1), DIGEST_401.format(2)])
    assert diagnose(f"rtsp://admin:bad@127.0.0.1:{port}/Streams/101", timeout=2)["status"] == "auth"


def test_diagnose_not_found_and_refused():
    port, _ = _fake_rtsp_server(["RTSP/1.0 404 Not Found\r\nCSeq: 1\r\n\r\n"])
    assert diagnose(f"rtsp://127.0.0.1:{port}/nope", timeout=2)["status"] == "not_found"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    status = diagnose(f"rtsp://127.0.0.1:{free}/x", timeout=2)["status"]
    # Windows retries a refused TCP connect until the timeout fires, so it can
    # surface as "timeout" there; elsewhere it must be reported as "refused".
    allowed = {"refused", "timeout"} if sys.platform == "win32" else {"refused"}
    assert status in allowed
