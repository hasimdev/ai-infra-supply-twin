from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_offline():
    r = client.post("/analyze", json={"runs": 2000, "backend": "numpy"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_eal"] > body["total_eal_before_news"] > 0
    assert len(body["risks"]) == len(body["network"]["nodes"])
    assert {"critical", "watch", "ok"} >= {x["flag"] for x in body["risks"]}


def test_analyze_rejects_bad_input():
    assert client.post("/analyze", json={"runs": 10}).status_code == 422


def test_dashboard_served():
    r = client.get("/")
    assert r.status_code == 200 and "AI Infra Supply Twin" in r.text
