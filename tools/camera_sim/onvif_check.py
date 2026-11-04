"""ONVIF end-to-end check against start_onvif.sh: discover -> profiles -> read frames from the stream URI."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend import onvif  # noqa: E402
from sightline_pipeline.ingest import VideoSource  # noqa: E402

USER, PW = "admin", "Sightline2026"
rep = {}
t = time.time()
devs = onvif.discover(3.0)
rep["multicast_discovery"] = {"devices": devs, "seconds": round(time.time() - t, 2)}
if not devs:
    import subprocess
    ip = subprocess.run(["bash", "-c", "ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1"],
                        capture_output=True, text=True).stdout.strip()
    devs = onvif.discover(3.0, hosts=[ip], multicast=False)
    rep["unicast_discovery"] = {"devices": devs}
if devs:
    x = devs[0]["xaddrs"][0]
    info = onvif.profiles(x, USER, PW)
    rep["profiles"] = info
    for p in info["profiles"]:
        src = VideoSource(onvif.with_credentials(p["uri"], USER, PW), max_reconnects=1)
        n, t0 = 0, time.time()
        for fr in src:
            n += 1
            if time.time() - t0 > 4:
                src.stop()
                break
        p["frames_4s"], p["size"], p["error"] = n, src.size, src.error
print(json.dumps(rep, indent=1, default=str))
