"""Tiny RTSP diagnostics (no ffmpeg): tell *why* a camera URL does not open.

Real cameras fail in a few distinct ways and they need different handling:
  * auth      - wrong user/password (401). Retrying fast can LOCK the account on Hikvision/Dahua
                NVRs (typically 5-7 failures), so callers back off for minutes, not seconds.
  * not_found - wrong stream path (404/454), e.g. /Streams/101 vs /cam/realmonitor.
  * unreachable / refused / timeout - host down, wrong IP/port, firewall.
  * ok        - DESCRIBE succeeded: the URL is fine, the failure is in the decoder/stream.
Speaks just enough RTSP/1.0 (OPTIONS-less DESCRIBE with Basic or Digest auth) to classify.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import socket
from urllib.parse import unquote, urlsplit

USER_AGENT = "Sightline/0.1"


def split_credentials(uri: str) -> tuple[str, str | None, str | None]:
    """rtsp://user:p%40ss@host:554/path -> ("rtsp://host:554/path", "user", "p@ss")."""
    u = urlsplit(uri)
    host = u.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{u.port}" if u.port else "")
    clean = u._replace(netloc=netloc).geturl()
    user = unquote(u.username) if u.username is not None else None
    pw = unquote(u.password) if u.password is not None else None
    return clean, user, pw


def _digest(user, pw, method, uri, chal):
    realm, nonce = chal.get("realm", ""), chal.get("nonce", "")
    ha1 = hashlib.md5(f"{user}:{realm}:{pw}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    qop = chal.get("qop")
    if qop and "auth" in [q.strip() for q in qop.split(",")]:
        nc, cnonce = "00000001", os.urandom(8).hex()
        resp = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
        return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", '
                f'response="{resp}", qop=auth, nc={nc}, cnonce="{cnonce}"')
    resp = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{resp}"'


def _challenges(headers: list[str]) -> dict:
    out = {}
    for h in headers:
        m = re.match(r"(?i)www-authenticate:\s*(\w+)\s*(.*)", h)
        if m:
            out[m.group(1).lower()] = dict(re.findall(r'(\w+)="?([^",]*)"?', m.group(2)))
    return out


def _request(sock, clean, cseq, auth=None) -> tuple[int, list[str]]:
    lines = [f"DESCRIBE {clean} RTSP/1.0", f"CSeq: {cseq}", f"User-Agent: {USER_AGENT}", "Accept: application/sdp"]
    if auth:
        lines.append(f"Authorization: {auth}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    m = re.match(r"RTSP/1\.\d (\d{3})", head[0] if head else "")
    if not m:
        raise ValueError("not an RTSP server")
    return int(m.group(1)), head[1:]


def diagnose(uri: str, timeout: float = 4.0) -> dict:
    """Returns {"status": ok|auth|not_found|refused|unreachable|timeout|protocol|unsupported, "code", "detail"}."""
    u = urlsplit(uri)
    if u.scheme not in ("rtsp",):
        return {"status": "unsupported", "code": None, "detail": f"{u.scheme}:// is not probed"}
    clean, user, pw = split_credentials(uri)
    host, port = u.hostname, u.port or 554
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except ConnectionRefusedError:
        return {"status": "refused", "code": None, "detail": f"connection refused by {host}:{port} (wrong port or service down)"}
    except socket.timeout:
        return {"status": "timeout", "code": None, "detail": f"no answer from {host}:{port} within {timeout:g}s"}
    except OSError as e:
        return {"status": "unreachable", "code": None, "detail": f"cannot reach {host}:{port} ({e.strerror or e})"}
    try:
        sock.settimeout(timeout)
        code, hdrs = _request(sock, clean, 1)
        if code == 401:
            ch = _challenges(hdrs)
            if user is None:
                return {"status": "auth", "code": 401, "detail": "camera requires a username and password"}
            if "digest" in ch:
                auth = _digest(user, pw or "", "DESCRIBE", clean, ch["digest"])
            else:
                auth = "Basic " + base64.b64encode(f"{user}:{pw or ''}".encode()).decode()
            code, hdrs = _request(sock, clean, 2, auth)
        if code == 200:
            return {"status": "ok", "code": 200, "detail": "stream description received"}
        if code in (401, 403):
            return {"status": "auth", "code": code, "detail": "authentication failed (wrong username or password)"}
        if code in (404, 454):
            return {"status": "not_found", "code": code, "detail": "stream path not found (wrong path, or the camera/channel is offline)"}
        return {"status": "protocol", "code": code, "detail": f"camera answered RTSP {code}"}
    except socket.timeout:
        return {"status": "timeout", "code": None, "detail": "camera accepted the connection but did not answer RTSP"}
    except (OSError, ValueError) as e:
        return {"status": "protocol", "code": None, "detail": f"RTSP handshake failed ({e})"}
    finally:
        sock.close()
