"""Expo push notifications (Android via FCM through Expo's push service).

Payload carries only event id, factor and risk; the app fetches the details over the authed API.
EXPO_ACCESS_TOKEN is optional (only needed if "enhanced push security" is on for the Expo project).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from sightline_pipeline.config import FACTOR_LABELS

log = logging.getLogger("sightline.push")
TOKEN_RE = re.compile(r"^Expo(nent)?PushToken\[[A-Za-z0-9_\-]{10,200}\]$")


def valid_token(t: str) -> bool:
    return bool(TOKEN_RE.match(t or ""))


class ExpoPush:
    def __init__(self, db, url: str, access_token: str | None = None, enabled: bool = True, cooldown_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        self.db, self.url, self.enabled, self.cooldown_s = db, url, enabled, cooldown_s
        headers = {"Accept": "application/json", "Accept-Encoding": "gzip, deflate", "Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self.client = httpx.Client(headers=headers, timeout=15.0, transport=transport)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="push")
        self._last: dict = {}
        self._lock = threading.Lock()
        self.stats = {"sent": 0, "errors": 0, "suppressed": 0, "last_error": None}

    @staticmethod
    def message(token: str, rec: dict) -> dict:
        f = rec.get("top_factor") or (rec.get("factors") or ["?"])[0]
        risk = float(rec.get("risk_max") or 0.0)
        return {"to": token, "title": f"Hazard: {FACTOR_LABELS.get(f, f)}",
                "body": f"Risk {round(risk * 100)}%. Tap to review the flagged clip.",
                "data": {"event_id": rec["event_id"], "factor": f, "risk": round(risk, 3)},
                "sound": "default", "priority": "high", "channelId": "alerts", "ttl": 3600}

    def notify(self, rec: dict):
        """Called from a pipeline thread; sends in the background."""
        if not self.enabled:
            return None
        key = (rec.get("camera_id"), rec.get("top_factor"))
        now = time.time()
        with self._lock:
            if now - self._last.get(key, 0.0) < self.cooldown_s:
                self.stats["suppressed"] += 1
                return None
            self._last[key] = now
        return self.pool.submit(self._send, rec)

    def _send(self, rec: dict):
        tokens = [d["token"] for d in self.db.devices()]
        out = []
        for i in range(0, len(tokens), 100):
            batch = tokens[i:i + 100]
            try:
                r = self.client.post(self.url, json=[self.message(t, rec) for t in batch])
                r.raise_for_status()
                tickets = r.json().get("data", [])
                for t, tk in zip(batch, tickets):
                    if tk.get("status") == "error":
                        self.stats["errors"] += 1
                        err = (tk.get("details") or {}).get("error")
                        self.stats["last_error"] = err or tk.get("message")
                        if err == "DeviceNotRegistered":
                            self.db.delete_device(t)
                    else:
                        self.stats["sent"] += 1
                out.extend(tickets)
            except Exception as e:
                self.stats["errors"] += 1
                self.stats["last_error"] = type(e).__name__
                log.warning("expo push failed: %s", type(e).__name__)
        return out

    def close(self):
        self.pool.shutdown(wait=False)
        self.client.close()
