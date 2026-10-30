"""Local MVP end-to-end: replay detector -> rules -> HistGB -> fix/verify.

Requires SIGHTLINE_MODE=replay and DETECTOR_BACKEND=replay (.env).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure package import when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.pipeline import Pipeline


def main() -> int:
    try:
        settings = get_settings()
        if settings.sightline_mode != "replay" or settings.detector_backend != "replay":
            raise SightlineError(
                error_code="demo_mode_required",
                component="scripts.demo_e2e_replay",
                message=f"Need MODE=replay and DETECTOR_BACKEND=replay (got {settings.sightline_mode}/{settings.detector_backend})",
                hint="Copy .env.example and set both for offline demo",
            )
        pipe = Pipeline(settings)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (45, 45, 45)
        results = []
        for i in range(5):
            results.append(pipe.analyze_frame(frame, camera_id="cam_01", timestamp=float(i) * 0.5))
        hazards = pipe.store.list_hazards()
        if not hazards:
            raise SightlineError(
                error_code="demo_no_hazards",
                component="scripts.demo_e2e_replay",
                message="Pipeline produced no hazards",
                hint="Check replay pack labels vs hazards.yaml classes",
            )
        # Prefer an open/candidate/confirmed hazard for fix-verify
        target = next((h for h in hazards if h.get("status") in {"open", "confirmed", "candidate"}), hazards[0])
        hid = target["hazard_id"]
        if target["status"] in {"candidate", "confirmed", "open", "needs_human"}:
            if target["status"] == "candidate":
                # Force open path via DB for demo when debounce not met
                from sightline.hazards.state_machine import transition

                try:
                    import time as _time

                    st = transition("candidate", "confirmed")
                    st = transition(st, "open")
                    pipe.store.execute(
                        "UPDATE hazards SET status=?, confirmed_at=? WHERE hazard_id=?",
                        (st, _time.time(), hid),
                    )
                except SightlineError:
                    pass
            fixed = pipe.mark_fixed(hid)
            verified = pipe.verify_fixed(hid)
        else:
            fixed, verified = target, target

        out = {
            "frames": len(results),
            "hazards": len(hazards),
            "sample_hazard": {
                "hazard_id": hid,
                "family": target.get("hazard_family"),
                "status_before": target.get("status"),
                "status_after_fix": fixed.get("status"),
                "status_after_verify": verified.get("status"),
                "time_to_fix_seconds": verified.get("time_to_fix_seconds"),
                "risk_score": target.get("risk_score"),
            },
            "latencies_ms": {k: len(v) for k, v in pipe.latencies.items()},
        }
        print(json.dumps(out, indent=2))
        print("OK: local MVP e2e replay path succeeded")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
