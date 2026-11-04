"""Password hashing (stdlib scrypt), JWTs (access / refresh / media / ws ticket) and rate limiting."""
from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import secrets
import threading
import time
import uuid

import jwt

ISSUER, AUDIENCE, ALG = "sightline", "sightline-api", "HS256"
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(password: str, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    """'scrypt:<n>:<r>:<p>:<salt>:<hash>' - no '$' so it is safe in .env files and shells."""
    if not password:
        raise ValueError("empty password")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=256 * 1024 * 1024, dklen=32)
    return f"scrypt:{n}:{r}:{p}:{_b64(salt)}:{_b64(dk)}"


_DUMMY = None


def verify_password(password: str, encoded: str | None) -> bool:
    """Constant-time check. An unknown user is checked against a dummy hash (same cost)."""
    global _DUMMY
    if encoded is None:
        _DUMMY = _DUMMY or hash_password(secrets.token_urlsafe(12))
        encoded, ok_user = _DUMMY, False
    else:
        ok_user = True
    try:
        algo, n, r, p, salt, want = encoded.split(":")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt((password or "").encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p),
                            maxmem=256 * 1024 * 1024, dklen=len(_unb64(want)))
        return hmac.compare_digest(dk, _unb64(want)) and ok_user
    except (ValueError, TypeError):
        return False


class Tokens:
    def __init__(self, secret: str, access_ttl: int, refresh_ttl: int, media_ttl: int):
        self.secret, self.access_ttl, self.refresh_ttl, self.media_ttl = secret, access_ttl, refresh_ttl, media_ttl

    def _encode(self, claims: dict, ttl: int) -> tuple[str, dict]:
        now = int(time.time())
        c = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "nbf": now, "exp": now + ttl, "jti": uuid.uuid4().hex, **claims}
        return jwt.encode(c, self.secret, algorithm=ALG), c

    def decode(self, token: str, typ: str) -> dict:
        c = jwt.decode(token, self.secret, algorithms=[ALG], audience=AUDIENCE, issuer=ISSUER,
                       options={"require": ["exp", "iat", "sub", "typ", "jti"]})
        if c.get("typ") != typ:
            raise jwt.InvalidTokenError("wrong token type")
        return c

    def access(self, sub: str, role: str) -> tuple[str, dict]:
        return self._encode({"sub": sub, "role": role, "typ": "access"}, self.access_ttl)

    def refresh(self, sub: str, role: str, family: str | None = None) -> tuple[str, dict]:
        return self._encode({"sub": sub, "role": role, "typ": "refresh", "fam": family or uuid.uuid4().hex},
                            self.refresh_ttl)

    def media(self, sub: str, scope: str) -> tuple[str, dict]:
        """Short-lived signed-URL token, valid only for URL paths under `scope`."""
        return self._encode({"sub": sub, "typ": "media", "scp": scope}, self.media_ttl)

    def ws_ticket(self, sub: str, role: str) -> tuple[str, dict]:
        return self._encode({"sub": sub, "role": role, "typ": "ws"}, 60)


class RateLimiter:
    """In-memory sliding-window limiter: allow(key, limit, window_s)."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._hits: dict = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_s: float) -> tuple[bool, float]:
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(0.0, window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 50_000:            # bound memory under abuse
                for k in [k for k, v in self._hits.items() if not v][:10_000]:
                    del self._hits[k]
            return True, 0.0

    def reset(self, key: str):
        with self._lock:
            self._hits.pop(key, None)
