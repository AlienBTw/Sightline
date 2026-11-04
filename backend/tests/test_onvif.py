"""ONVIF client (no network): probe-match parsing, WS-Security digest, profiles via a mock device, API."""
import base64
import datetime as dt
import hashlib

import httpx
import pytest

from backend import onvif

from .conftest import auth, login

MATCH = b"""<?xml version="1.0"?><e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing">
<e:Body><d:ProbeMatches><d:ProbeMatch><w:EndpointReference><w:Address>urn:uuid:1234</w:Address></w:EndpointReference>
<d:Scopes>onvif://www.onvif.org/name/Sim%20Cam onvif://www.onvif.org/hardware/Virtual</d:Scopes>
<d:XAddrs>http://10.0.0.5:8081/onvif/device_service</d:XAddrs></d:ProbeMatch></d:ProbeMatches></e:Body></e:Envelope>"""


def test_parse_probe_match():
    m = onvif.parse_probe_match(MATCH)
    assert m["xaddrs"] == ["http://10.0.0.5:8081/onvif/device_service"]
    assert m["name"] == "Sim Cam" and m["hardware"] == "Virtual" and m["endpoint"] == "urn:uuid:1234"
    with pytest.raises(onvif.OnvifError):
        onvif.parse_probe_match(b'<!DOCTYPE x [<!ENTITY a "b">]><x/>')


def test_password_digest_matches_spec():
    now, nonce = dt.datetime(2026, 11, 1, 1, 0, tzinfo=dt.timezone.utc), b"0123456789abcdef"
    h = onvif.security_header("admin", "pw", now=now, nonce=nonce)
    want = base64.b64encode(hashlib.sha1(nonce + b"2026-11-01T01:00:00.000Z" + b"pw").digest()).decode()
    assert want in h and "pw<" not in h and "<Username>admin</Username>" in h


def test_with_credentials_encodes():
    assert onvif.with_credentials("rtsp://10.0.0.5:554/Streams/101", "ad min", "p@ss:w") == \
        "rtsp://ad%20min:p%40ss%3Aw@10.0.0.5:554/Streams/101"
    assert onvif.with_credentials("rtsp://h/x", None, None) == "rtsp://h/x"


def _device(req: httpx.Request):
    body = req.content.decode()
    if "PasswordDigest" not in body:
        return httpx.Response(400, content=b"<x/>")
    if "GetCapabilities" in body:
        x = "<tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media><tt:XAddr>http://cam/onvif/media_service</tt:XAddr></tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse>"
    elif "GetDeviceInformation" in body:
        x = "<tds:GetDeviceInformationResponse><tds:Manufacturer>Sim</tds:Manufacturer><tds:Model>CamX</tds:Model></tds:GetDeviceInformationResponse>"
    elif "GetProfiles" in body:
        x = ('<trt:GetProfilesResponse><trt:Profiles token="main"><tt:Name>MainStream</tt:Name><tt:VideoEncoderConfiguration>'
             '<tt:Encoding>H264</tt:Encoding><tt:Resolution><tt:Width>1920</tt:Width><tt:Height>1080</tt:Height></tt:Resolution>'
             '<tt:RateControl><tt:FrameRateLimit>25</tt:FrameRateLimit></tt:RateControl></tt:VideoEncoderConfiguration></trt:Profiles>'
             '<trt:Profiles token="sub"><tt:Name>SubStream</tt:Name></trt:Profiles></trt:GetProfilesResponse>')
    elif "GetStreamUri" in body:
        path = "101" if ">main<" in body else "102"
        x = f"<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>rtsp://cam:554/Streams/{path}</tt:Uri></trt:MediaUri></trt:GetStreamUriResponse>"
    else:
        return httpx.Response(500, content=b"<x/>")
    env = ('<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
           'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>'
           f"{x}</s:Body></s:Envelope>")
    return httpx.Response(200, content=env.encode())


@pytest.fixture
def mock_device(monkeypatch):
    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(_device)
        return real(*a, **kw)
    monkeypatch.setattr(onvif.httpx, "Client", client)


def test_profiles_from_mock_device(mock_device):
    info = onvif.profiles("http://cam/onvif/device_service", "admin", "pw")
    assert info["device"]["Model"] == "CamX" and info["media_xaddr"] == "http://cam/onvif/media_service"
    assert [(p["token"], p["uri"]) for p in info["profiles"]] == [("main", "rtsp://cam:554/Streams/101"),
                                                                  ("sub", "rtsp://cam:554/Streams/102")]
    assert info["profiles"][0]["video"] == {"encoding": "H264", "width": 1920, "height": 1080, "fps": 25}
    with pytest.raises(onvif.OnvifError):
        onvif.profiles("file:///etc/passwd")


def test_api_add_camera_by_onvif(client, mock_device):
    h = auth(login(client, "admin"))
    r = client.post("/api/onvif/add", headers=h, json={"xaddr": "http://cam/onvif/device_service", "username": "admin",
                                                        "password": "p@ss", "profile_token": "sub", "start": False})
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["source"] == "rtsp://***@cam:554/Streams/102" and j["onvif"]["profile"] == "sub" and "p@ss" not in r.text
    assert j["name"] == "CamX SubStream"
    demo = auth(login(client, "demo"))
    assert client.post("/api/onvif/discover", headers=demo, json={}).status_code == 403
