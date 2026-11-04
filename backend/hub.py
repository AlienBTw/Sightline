"""Fan-out of real-time messages (new events, camera status, live alert state) to WebSocket clients.

Pipeline workers run in threads; `publish` is thread-safe and hands messages to the asyncio loop.
"""
from __future__ import annotations

import asyncio
import json
import threading


class Hub:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: dict[int, tuple[str, asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._n = 0

    def bind(self, loop):
        self.loop = loop

    def count(self, sub: str | None = None) -> int:
        with self._lock:
            return sum(1 for s, _ in self._clients.values() if sub is None or s == sub)

    def add(self, sub: str) -> tuple[int, asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._n += 1
            self._clients[self._n] = (sub, q)
            return self._n, q

    def remove(self, cid: int):
        with self._lock:
            self._clients.pop(cid, None)

    def _put_all(self, msg: str):
        with self._lock:
            qs = [q for _, q in self._clients.values()]
        for q in qs:
            if q.full():                      # slow client: drop its oldest message
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    def publish(self, message: dict):
        msg = json.dumps(message, default=str)
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._put_all(msg)
        else:
            loop.call_soon_threadsafe(self._put_all, msg)
