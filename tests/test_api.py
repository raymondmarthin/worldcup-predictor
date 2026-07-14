"""test_api.py - FastAPI smoke tests."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app

# Use context manager so FastAPI's startup event (model loading) actually runs.
client = TestClient(app)
client.__enter__()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert "features_used" in body
    assert isinstance(body["features_used"], list)


def test_predict_valid_teams():
    r = client.get("/predict", params={"team_a": "France", "team_b": "Brazil"})
    assert r.status_code == 200
    body = r.json()
    assert body["team_a"] == "France"
    assert body["team_b"] == "Brazil"
    assert abs(body["win_a"] + body["win_b"] - 1.0) < 1e-3
    assert 0.0 <= body["win_a"] <= 1.0
    assert 0.0 <= body["win_b"] <= 1.0
    assert body["confidence"] in {"low", "medium", "high"}


def test_predict_unknown_team():
    r = client.get("/predict", params={"team_a": "Narnia", "team_b": "Brazil"})
    assert r.status_code == 404


def test_predict_same_team():
    r = client.get("/predict", params={"team_a": "France", "team_b": "France"})
    assert r.status_code == 400


def test_bracket_returns_predictions():
    r = client.get("/bracket")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    for pred in body["predictions"]:
        assert abs(pred["win_a"] + pred["win_b"] - 1.0) < 1e-3
