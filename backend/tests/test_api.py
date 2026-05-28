"""
Basic API tests for CryptoPredict backend.
Run with: pytest tests/ -v
"""
import pytest
from httpx import AsyncClient, ASGITransport

from main import app, state


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "model_loaded" in data
    assert "models_loaded" in data
    assert "feedback_count" in data


@pytest.mark.anyio
async def test_metrics(client):
    r = await client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data
    assert "accuracy_percentage" in data


@pytest.mark.anyio
async def test_coins(client):
    r = await client.get("/api/coins")
    assert r.status_code == 200
    coins = r.json()["supported_coins"]
    assert any(c["id"] == "bitcoin" for c in coins)


@pytest.mark.anyio
async def test_feedback_history(client):
    r = await client.get("/api/feedback/history")
    assert r.status_code == 200
    assert "feedback" in r.json()


@pytest.mark.anyio
async def test_predict_no_model(client):
    """Should return 503 when model not loaded."""
    original = state.models.pop("bitcoin", None)
    r = await client.post("/api/predict", json={"coin_id": "bitcoin", "days": 7})
    assert r.status_code == 503
    if original is not None:
        state.models["bitcoin"] = original


@pytest.mark.anyio
async def test_feedback_submission(client):
    payload = {
        "prediction_id": "test_123",
        "actual_price": 65000.0,
        "predicted_price": 64500.0,
        "date": "2026-01-01",
        "is_accurate": True,
        "user_comment": "Good prediction",
    }
    r = await client.post("/api/feedback", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["feedback_count"] >= 1
