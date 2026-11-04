"""Integration: real RTSP server (MediaMTX) + real ffmpeg camera -> Sightline ingest.
Skipped unless the simulator binaries exist ($CAMSIM_BIN, default <repo>/work/camsim/bin; fetch them
with tools/camera_sim/start_sim.sh). Run:  pytest -m integration sightline_pipeline/tests"""
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from sightline_pipeline.ingest import VideoSource

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
BIN = Path(os.environ.get("CAMSIM_BIN", WORK / "camsim" / "bin"))
CFG = REPO / "tools" / "camera_sim" / "mediamtx.yml"
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not ((BIN / "mediamtx").exists() and (BIN / "ffmpeg").exists() and CFG.exists()),
                                 reason="camera simulator binaries not installed")]
PORT = 18554


def _wait_port(port, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.fixture(scope="module")
def camera(tmp_path_factory):
    d = tmp_path_factory.mktemp("camsim")
    cfg = CFG.read_text().replace(":8554", f":{PORT}").replace(":8050", ":18050").replace(":8051", ":18051") \
        .replace("api: yes", "api: no")
    (d / "mediamtx.yml").write_text(cfg)
    srv = subprocess.Popen([str(BIN / "mediamtx"), str(d / "mediamtx.yml")], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    if not _wait_port(PORT) or srv.poll() is not None:
        srv.kill()
        pytest.skip("mediamtx could not start (port in use?)")
    pub = subprocess.Popen([str(BIN / "ffmpeg"), "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=15", "-c:v", "libx264", "-preset", "ultrafast",
                            "-tune", "zerolatency", "-g", "15", "-pix_fmt", "yuv420p", "-f", "rtsp",
                            f"rtsp://publisher:pub-local-only@127.0.0.1:{PORT}/Streams/101"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    yield f"rtsp://admin:Sightline2026@127.0.0.1:{PORT}/Streams/101"
    for p in (pub, srv):
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def _read(uri, seconds=3.0, **kw):
    import threading
    stop = threading.Event()
    src = VideoSource(uri, cancel=stop, max_reconnects=3, **kw)
    timer = threading.Timer(seconds + 15, stop.set)          # never hang the test run
    timer.start()
    frames, t = [], time.time()
    for fr in src:
        frames.append(fr)
        if time.time() - t > seconds:
            src.stop()
            break
    timer.cancel()
    return src, frames


@pytest.mark.parametrize("transport", ["tcp", "udp"])
def test_reads_rtsp_with_credentials(camera, transport):
    src, frames = _read(camera, seconds=5.0, rtsp_transport=transport)
    assert len(frames) >= 10 and frames[-1].image.shape == (360, 640, 3) and src.size == (640, 360)


def test_wrong_password_is_a_clear_error(camera):
    src = VideoSource(camera.replace("Sightline2026", "nope"), max_reconnects=0)
    assert list(src) == []
    assert "authentication failed" in src.error and "nope" not in src.error


def test_wrong_path_is_reported(camera):
    src = VideoSource(camera.replace("101", "999"), max_reconnects=0)
    list(src)
    assert "not_found" in src.error
