from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health(): assert client.get("/health").json()["status"]=="ok"
def test_model_status_shape(): assert client.get("/api/model/status").json()["expected_shape"]==[32,258]
def test_emergency_keyword_flag():
    r=client.post("/api/emergency/check",json={"text":"I cannot breathe"}).json()
    assert r["attention_required"]
