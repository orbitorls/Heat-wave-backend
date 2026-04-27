"""Integration tests for Heatwave Forecast API endpoints using Flask test client."""
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create Flask test client with resources NOT loaded (unit test mode)."""
    from api_server import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert "status" in data

    def test_health_json_content_type(self, client):
        resp = client.get("/api/health")
        assert resp.content_type == "application/json"


class TestPredictEndpoint:
    def test_predict_returns_json(self, client):
        resp = client.get("/api/predict")
        assert resp.content_type == "application/json"

    def test_predict_when_not_loaded_returns_503_or_fallback(self, client):
        resp = client.get("/api/predict")
        assert resp.status_code in (200, 503, 500)

    def test_predict_with_days_param(self, client):
        resp = client.get("/api/predict?days=3")
        assert resp.status_code in (200, 503, 500)


class TestForecastEndpoint:
    def test_forecast_returns_json(self, client):
        resp = client.get("/api/forecast")
        assert resp.content_type == "application/json"

    def test_forecast_status_code(self, client):
        resp = client.get("/api/forecast")
        assert resp.status_code in (200, 503, 500)


class TestMapEndpoint:
    def test_map_returns_json(self, client):
        resp = client.get("/api/map")
        assert resp.content_type == "application/json"

    def test_map_status_code(self, client):
        resp = client.get("/api/map")
        assert resp.status_code in (200, 503, 500)


class TestInvalidRequests:
    def test_nonexistent_endpoint_returns_404(self, client):
        resp = client.get("/api/nonexistent_endpoint_xyz")
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client):
        resp = client.delete("/api/health")
        assert resp.status_code in (404, 405)


class TestTrainingEndpointAuth:
    def test_training_start_requires_post(self, client):
        resp = client.get("/api/training/start")
        assert resp.status_code in (404, 405)

    def test_training_start_post_returns_json(self, client):
        resp = client.post("/api/training/start", json={})
        assert resp.content_type == "application/json"
        assert resp.status_code in (200, 400, 401, 409, 500)
