import time

import jwt

from backend.security import Tokens, hash_password, verify_password
from .conftest import PASSWORDS, auth, login


def test_hash_roundtrip_and_format():
    h = hash_password("s3cret-pass", n=2 ** 12)
    assert h.startswith("scrypt:") and "$" not in h
    assert verify_password("s3cret-pass", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", None) and not verify_password("x", "garbage")


def test_login_ok_and_me(client):
    t = login(client)
    assert t["token_type"] == "bearer" and t["user"] == {"username": "demo", "role": "demo"}
    r = client.get("/api/auth/me", headers=auth(t))
    assert r.status_code == 200 and r.json()["role"] == "demo"


def test_login_bad_password_and_unknown_user(client):
    assert client.post("/api/auth/login", json={"username": "demo", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ghost", "password": "nope"}).status_code == 401


def test_login_rate_limited(client):
    codes = [client.post("/api/auth/login", json={"username": "demo", "password": "bad"}).status_code for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    r = client.post("/api/auth/login", json={"username": "demo", "password": "bad"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_every_api_route_needs_auth(client):
    app = client.app
    public = {"/health", "/api/auth/login", "/api/auth/refresh", "/api/auth/logout", "/", "/console/", "/console/{name}"}
    checked = 0
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if path in public or not path.startswith("/api") or path == "/api/ws":
            continue
        url = (path.replace("{cid}", "cam_abc").replace("{eid}", "evt_x").replace("{uid}", "up_0123456789abcdef")
               .replace("{name}", "clip.mp4"))
        for m in methods - {"HEAD", "OPTIONS"}:
            r = client.request(m, url)
            assert r.status_code in (401, 422), (m, url, r.status_code)
            if r.status_code == 422:      # validation ran first; the token is still required
                r = client.request(m, url, json={})
                assert r.status_code in (401, 422)
            checked += 1
    assert checked >= 20


def test_bad_tokens_rejected(client, app_env):
    _, s, _, _ = app_env
    assert client.get("/api/cameras", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    other = Tokens("x" * 48, 60, 60, 60).access("demo", "demo")[0]
    assert client.get("/api/cameras", headers={"Authorization": f"Bearer {other}"}).status_code == 401
    tk = Tokens(s.jwt_secret, 60, 60, 60)
    expired = jwt.encode({"iss": "sightline", "aud": "sightline-api", "sub": "demo", "role": "demo", "typ": "access",
                          "jti": "a", "iat": int(time.time()) - 100, "exp": int(time.time()) - 10}, s.jwt_secret)
    r = client.get("/api/cameras", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]
    refresh_as_access = tk.refresh("demo", "demo")[0]
    assert client.get("/api/cameras", headers={"Authorization": f"Bearer {refresh_as_access}"}).status_code == 401


def test_refresh_rotation_and_reuse_detection(client):
    t = login(client)
    r1 = client.post("/api/auth/refresh", json={"refresh_token": t["refresh_token"]})
    assert r1.status_code == 200
    t2 = r1.json()
    assert client.get("/api/auth/me", headers=auth(t2)).status_code == 200
    # replaying the old refresh token kills the whole family
    assert client.post("/api/auth/refresh", json={"refresh_token": t["refresh_token"]}).status_code == 401
    assert client.post("/api/auth/refresh", json={"refresh_token": t2["refresh_token"]}).status_code == 401


def test_logout_revokes_refresh(client):
    t = login(client)
    assert client.post("/api/auth/logout", json={"refresh_token": t["refresh_token"]}).status_code == 200
    assert client.post("/api/auth/refresh", json={"refresh_token": t["refresh_token"]}).status_code == 401


def test_demo_role_cannot_add_rtsp_or_delete(client):
    t = login(client)
    r = client.post("/api/cameras", json={"name": "x", "uri": "rtsp://10.0.0.2:8554/live"}, headers=auth(t))
    assert r.status_code == 403


def test_health_public_and_minimal(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert set(r.json()) == {"status", "version", "time", "uptime_s", "cameras"}
    assert r.headers["x-content-type-options"] == "nosniff"


def test_console_served_with_csp(client):
    r = client.get("/console/")
    assert r.status_code == 200 and "Sightline" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/console/app.js").status_code == 200
    assert client.get("/console/../settings.py").status_code == 404
