from fastapi.testclient import TestClient

from sightline.api.main import app


def test_health_shape():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body
    assert "mode" in body or "error" in body
