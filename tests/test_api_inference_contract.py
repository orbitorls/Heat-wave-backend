import numpy as np
import types
import sys

import api_server


class _FakeClassifier:
    n_features_in_ = 8

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]], dtype=np.float32)


class _FakeForecastModel:
    def __call__(self, x_tensor, future_seq=1):
        batch_size, _seq_len, channels, height, width = x_tensor.shape
        out = np.zeros((batch_size, future_seq, channels, height, width), dtype=np.float32)
        out[:, :, 1, :, :] = 40.0
        import torch

        return torch.from_numpy(out)


def _base_sequence():
    seq = np.zeros((1, 2, 2, 2), dtype=np.float32)
    seq[-1, 1, :, :] = 23.0
    return np.array([seq], dtype=np.float32)


def test_classifier_sequence_uses_observed_persistence_without_temperature_model(monkeypatch):
    monkeypatch.setattr(api_server, "resources_ready", lambda: True)
    monkeypatch.setattr(api_server, "model", _FakeClassifier())
    monkeypatch.setattr(api_server, "runtime_model_type", "balanced_random_forest")
    monkeypatch.setattr(api_server, "X_test", _base_sequence())
    monkeypatch.setattr(api_server, "temp_mean_scalar", 0.0)
    monkeypatch.setattr(api_server, "temp_std_scalar", 1.0)
    monkeypatch.setattr(api_server, "temperature_forecast_context", None)
    monkeypatch.setattr(api_server, "_prediction_cache", {})

    result = api_server.get_prediction_sequence(days=1)[0]

    assert result["forecast_available"] is False
    assert result["temperature_source"] == "observed_persistence"
    np.testing.assert_allclose(result["observed_temperature_grid"], 23.0, atol=1e-3)
    np.testing.assert_allclose(result["temperature_grid"], result["observed_temperature_grid"])


def test_classifier_sequence_prefers_secondary_temperature_forecast_when_available(monkeypatch):
    monkeypatch.setattr(api_server, "resources_ready", lambda: True)
    monkeypatch.setattr(api_server, "model", _FakeClassifier())
    monkeypatch.setattr(api_server, "runtime_model_type", "balanced_random_forest")
    monkeypatch.setattr(api_server, "X_test", _base_sequence())
    monkeypatch.setattr(api_server, "temp_mean_scalar", 0.0)
    monkeypatch.setattr(api_server, "temp_std_scalar", 1.0)
    monkeypatch.setattr(
        api_server,
        "temperature_forecast_context",
        {
            "model": _FakeForecastModel(),
            "X_test": _base_sequence(),
            "temp_mean_scalar": 0.0,
            "temp_std_scalar": 1.0,
            "model_type": "convlstm",
        },
    )
    monkeypatch.setattr(api_server, "_prediction_cache", {})

    result = api_server.get_prediction_sequence(days=1)[0]

    assert result["forecast_available"] is True
    assert result["temperature_source"] == "forecast_model"
    np.testing.assert_allclose(result["observed_temperature_grid"], 23.0, atol=1e-3)
    np.testing.assert_allclose(result["forecast_temperature_grid"], 40.0, atol=1e-3)
    np.testing.assert_allclose(result["temperature_grid"], 40.0, atol=1e-3)


def test_predict_summary_exposes_observed_and_forecast_weather(monkeypatch):
    fake_daily = types.SimpleNamespace(
        daily_model_ready=lambda: False,
        load_daily_model=lambda: False,
        predict_from_daily_weather=lambda weather: None,
        _daily_threshold=0.5,
    )
    monkeypatch.setitem(sys.modules, "api_daily_predict", fake_daily)
    monkeypatch.setattr(api_server, "resources_ready", lambda: True)
    monkeypatch.setattr(
        api_server,
        "get_prediction_sequence",
        lambda days=1, sample_idx=-1: [
            {
                "temperature_grid": np.full((2, 2), 40.0, dtype=np.float32),
                "forecast_temperature_grid": np.full((2, 2), 40.0, dtype=np.float32),
                "observed_temperature_grid": np.full((2, 2), 23.0, dtype=np.float32),
                "probability": 0.8,
                "temperature_source": "forecast_model",
                "forecast_available": True,
            }
        ],
    )
    monkeypatch.setattr(api_server, "get_sample_date", lambda sample_idx=-1, day_offset=0: __import__("datetime").datetime(2026, 4, 12))
    monkeypatch.setattr(api_server, "runtime_model_type", "balanced_random_forest")
    monkeypatch.setattr(api_server, "bbox", {"north": 1, "south": 0, "east": 1, "west": 0})
    monkeypatch.setattr(api_server, "_region_payload_from_grid", lambda temp_grid, model_probability=None: [])

    client = api_server.app.test_client()
    response = client.get("/api/predict")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["forecast_available"] is True
    assert payload["temperature_source"] == "forecast_model"
    assert payload["observed_weather"]["T2M_MAX"] == 23.0
    assert payload["forecast_weather"]["T2M_MAX"] == 40.0
    assert payload["weather"]["T2M_MAX"] == 40.0
