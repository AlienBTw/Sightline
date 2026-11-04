"""Minimal ONVIF client (Profile S): WS-Discovery + GetProfiles + GetStreamUri, no WSDL/zeep.

Lets an admin add a camera by ONVIF instead of typing a brand-specific RTSP URL:
  discover()                      -> devices answering a WS-Discovery Probe (multicast and/or unicast)
  profiles(xaddr, user, password) -> media profiles with their RTSP stream URIs
Speaks plain SOAP 1.2 over httpx with a WS-Security UsernameToken (PasswordDigest), which is what
Hikvision, Dahua, Reolink, Axis and most NVRs accept. Optional: nothing else in Sightline needs it.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import os
import re
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlsplit

import httpx

WSD_ADDR = ("239.255.255.250", 3702)
MAX_XML = 512 * 1024

PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{mid}</w:MessageID><w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"""


class OnvifError(RuntimeError):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [el for el in root.iter() if _local(el.tag) == name]


def _text(root, name, default=None):
    for el in root.iter():
        if _local(el.tag) == name and el.text:
            return el.text.strip()
    return default


def _parse(data: bytes):
    if len(data) > MAX_XML or b"<!ENTITY" in data:
        raise OnvifError("refusing oversized or entity-bearing XML")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise OnvifError(f"bad XML from device: {e}") from None


def parse_probe_match(data: bytes) -> dict | None:
    root = _parse(data)
    if not _find_all(root, "ProbeMatch"):
        return None
    xaddrs = (_text(root, "XAddrs") or "").split()
    scopes = (_text(root, "Scopes") or "").split()

    def scope(kind):
        for s in scopes:
            m = re.match(rf"onvif://www\.onvif\.org/{kind}/(.+)", s)
            if m:
                return m.group(1).replace("%20", " ")
        return None
    return {"xaddrs": xaddrs, "endpoint": _text(root, "Address"), "name": scope("name"),
            "hardware": scope("hardware"), "scopes": scopes}


def discover(timeout: float = 3.0, hosts: list[str] | None = None, multicast: bool = True) -> list[dict]:
    """WS-Discovery Probe. `hosts` adds unicast probes (useful across NAT/WSL where multicast is lost)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    sock.settimeout(0.3)
    msg = PROBE.format(mid=uuid.uuid4()).encode()
    targets = ([WSD_ADDR] if multicast else []) + [(h, 3702) for h in (hosts or [])]
    found: dict[str, dict] = {}
    try:
        sock.bind(("", 0))
        for t in targets:
            try:
                sock.sendto(msg, t)
            except OSError:
                pass
        end = time.time() + timeout
        while time.time() < end:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                m = parse_probe_match(data)
            except OnvifError:
                continue
            if m and m["xaddrs"]:
                m["from"] = addr[0]
                found.setdefault(m["endpoint"] or m["xaddrs"][0], m)
    finally:
        sock.close()
    return list(found.values())


def security_header(user: str | None, password: str | None, now: dt.datetime | None = None, nonce: bytes | None = None) -> str:
    if not user:
        return ""
    nonce = nonce or os.urandom(16)
    created = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + (password or "").encode()).digest()).decode()
    return ('<s:Header><Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
            '<UsernameToken><Username>{u}</Username><Password Type="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{d}</Password><Nonce EncodingType="http://docs.'
            'oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{n}</Nonce>'
            '<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{c}</Created>'
            '</UsernameToken></Security></s:Header>').format(u=_xml_escape(user), d=digest,
                                                              n=base64.b64encode(nonce).decode(), c=created)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _soap(client: httpx.Client, url: str, body: str, user, password):
    env = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
           'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
           'xmlns:tt="http://www.onvif.org/ver10/schema">' + security_header(user, password) +
           f"<s:Body>{body}</s:Body></s:Envelope>")
    try:
        r = client.post(url, content=env.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    except httpx.HTTPError as e:
        raise OnvifError(f"cannot reach ONVIF service ({type(e).__name__})") from None
    root = _parse(r.content) if r.content else None
    if r.status_code in (401, 403) or b"NotAuthorized" in r.content:
        raise OnvifError("ONVIF authentication failed (wrong username or password)")
    if r.status_code >= 400 or root is None:
        raise OnvifError(f"ONVIF call failed (HTTP {r.status_code})")
    return root


def _check_xaddr(xaddr: str) -> str:
    u = urlsplit(xaddr)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise OnvifError("xaddr must be an http(s) ONVIF device service URL")
    return xaddr


def profiles(xaddr: str, user: str | None = None, password: str | None = None, timeout: float = 5.0) -> dict:
    """Device info + media profiles with RTSP URIs (URIs as the camera reports them, without credentials)."""
    _check_xaddr(xaddr)
    with httpx.Client(timeout=timeout, follow_redirects=False) as c:
        root = _soap(c, xaddr, '<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>', user, password)
        media = None
        for el in root.iter():
            if _local(el.tag) == "Media":
                media = _text(el, "XAddr")
                if media:
                    break
        media = media or xaddr
        try:
            info = _soap(c, xaddr, "<tds:GetDeviceInformation/>", user, password)
            device = {k: _text(info, k) for k in ("Manufacturer", "Model", "FirmwareVersion")}
        except OnvifError:
            device = {}
        pr = _soap(c, media, "<trt:GetProfiles/>", user, password)
        out = []
        for p in _find_all(pr, "Profiles"):
            tok = p.get("token")
            if not tok:
                continue
            enc = None
            for el in p.iter():
                if _local(el.tag) == "VideoEncoderConfiguration":
                    enc = {"encoding": _text(el, "Encoding"), "width": _int(_text(el, "Width")),
                           "height": _int(_text(el, "Height")), "fps": _int(_text(el, "FrameRateLimit"))}
                    break
            su = _soap(c, media, '<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport>'
                                 f'<tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup><trt:ProfileToken>{_xml_escape(tok)}'
                                 '</trt:ProfileToken></trt:GetStreamUri>', user, password)
            out.append({"token": tok, "name": _text(p, "Name") or tok, "video": enc, "uri": _text(su, "Uri")})
    return {"xaddr": xaddr, "media_xaddr": media, "device": device, "profiles": out}


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def with_credentials(uri: str, user: str | None, password: str | None) -> str:
    """rtsp://host/path + creds -> rtsp://user:pass@host/path (percent-encoded; replaces any existing creds)."""
    if not user:
        return uri
    u = urlsplit(uri)
    host = u.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{quote(user, safe='')}:{quote(password or '', safe='')}@{host}" + (f":{u.port}" if u.port else "")
    return u._replace(netloc=netloc).geturl()
