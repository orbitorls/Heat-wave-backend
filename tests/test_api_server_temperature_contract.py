"""Contract tests for temperature behavior in api_server prediction helpers."""

import numpy as np
import pytest
import torch

import api_server


class _FakeConvLSTMModel:
    def __init__(self, predicted_temp_c, channels=3, height=2, width=2):
        self.predicted_temp_c = float(predicted_temp_c)
        self.channels = channels
        self.height = height
        self.width = width

    def __call__(self, x_tensor, future_seq=1):
        batch = int(x_tensor.shape[0])
        out = torch.zeros(
            (batch, future_seq, self.channels, self.height, self.width),
            dtype=torch.float32,
            device=x_tensor.device,
        )
        out[:, :, 1, :, :] = self.predicted_temp_c
        return out


class _FakeClassifierModel:
    def __init__(self, n_features):
        self.n_features_in_ = int(n_features)

    def predict_proba(self, x):
        return np.array([[0.2, 0.8]], dtype=np.float32)


@pytest.fixture(autouse=True)
def _restore_api_globals():
    tracked = [
        "model",
        "X_test",
        "Y_test",
        "lats",
        "lons",
        "mean",
        "std",
        "temp_mean_scalar",
        "temp_std_scalar",
        "runtime_model_type",
        "temperature_forecast_context",
        "_prediction_cache",
    ]
    snapshot = {name: getattr(api_server, name) for name in tracked}
    yield
    for name, value in snapshot.items():
        setattr(api_server, name, value)


def _make_test_sequence(observed_temp_c):
    # Shape: (samples=1, seq_len=3, channels=3, H=2, W=2)
    x_test = np.zeros((1, 3, 3, 2, 2), dtype=np.float32)
    x_test[0, -1, 1, :, :] = float(observed_temp_c)
    return x_test


def _prepare_common_state(observed_temp_c):
    api_server.X_test = _make_test_sequence(observed_temp_c)
    api_server.Y_test = np.zeros((1, 1, 3, 2, 2), dtype=np.float32)
    api_server.lats = np.array([13.0, 14.0], dtype=np.float32)
    api_server.lons = np.array([100.0, 101.0], dtype=np.float32)
    api_server.mean = np.zeros((1, 3, 1, 1), dtype=np.float32)
    api_server.std = np.ones((1, 3, 1, 1), dtype=np.float32)
    api_server.temp_mean_scalar = 0.0
    api_server.temp_std_scalar = 1.0
    api_server._prediction_cache = {}


def _observed_last_frame_temperature():
    observed_norm = api_server.X_test[-1, -1, 1, :, :]
    observed = observed_norm * (api_server.temp_std_scalar + api_server.EPSILON)
    observed = observed + api_server.temp_mean_scalar
    return np.clip(observed, -10.0, 60.0)


def test_observed_temperature_is_separable_from_forecast_temperature():
    _prepare_common_state(observed_temp_c=34.0)
    api_server.runtime_model_type = "convlstm"
    api_server.model = _FakeConvLSTMModel(predicted_temp_c=22.0)

    entry = api_server.get_prediction_sequence(days=1)[0]
    forecast_grid = entry["forecast_temperature_grid"]
    observed_grid = entry["observed_temperature_grid"]

    assert np.allclose(observed_grid, 34.0, atol=1e-3)
    assert np.allclose(forecast_grid, 22.0, atol=1e-3)
    assert np.allclose(entry["temperature_grid"], forecast_grid, atol=1e-3)
    assert entry["temperature_source"] == "forecast_model"
    assert entry["forecast_available"] is True
    assert not np.allclose(forecast_grid, observed_grid)


def test_classifier_backend_must_not_report_persistence_as_forecast_temperature():
    _prepare_common_state(observed_temp_c=34.0)
    api_server.runtime_model_type = "balanced_random_forest"
    feature_count = int(np.prod(api_server.X_test.shape[1:]))
    api_server.model = _FakeClassifierModel(n_features=feature_count)

    entry = api_server.get_prediction_sequence(days=1)[0]
    observed_grid = _observed_last_frame_temperature()

    assert entry["forecast_temperature_grid"] is None
    assert entry["forecast_available"] is False
    assert entry["temperature_source"] == "observed_persistence"
    assert np.allclose(entry["observed_temperature_grid"], observed_grid, atol=1e-3)
    assert np.allclose(entry["temperature_grid"], observed_grid, atol=1e-3)


def test_dedicated_convlstm_temperature_context_is_preferred_for_temperature_output():
    _prepare_common_state(observed_temp_c=34.0)
    api_server.runtime_model_type = "balanced_random_forest"
    feature_count = int(np.prod(api_server.X_test.shape[1:]))
    api_server.model = _FakeClassifierModel(n_features=feature_count)

    api_server.temperature_forecast_context = {
        "model": _FakeConvLSTMModel(predicted_temp_c=41.0),
        "X_test": _make_test_sequence(observed_temp_c=34.0),
        "temp_mean_scalar": 0.0,
        "temp_std_scalar": 1.0,
    }

    entry = api_server.get_prediction_sequence(days=1)[0]
    forecast_grid = entry["forecast_temperature_grid"]
    observed_grid = entry["observed_temperature_grid"]

    assert forecast_grid is not None
    assert np.allclose(forecast_grid, 41.0, atol=1e-3)
    assert np.allclose(entry["temperature_grid"], forecast_grid, atol=1e-3)
    assert entry["temperature_source"] == "forecast_model"
    assert entry["forecast_available"] is True
    assert not np.allclose(entry["temperature_grid"], observed_grid, atol=1e-3)
