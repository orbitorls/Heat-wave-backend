"""Unit tests for src/models/manager.py — ModelManager class."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.models.manager import ModelManager


class _FakeModelsDir:
    def __init__(self, rf_files=None, convlstm_files=None):
        self._rf_files = rf_files or []
        self._convlstm_files = convlstm_files or []

    def glob(self, pattern):
        if pattern == "heatwave_model_checkpoint_v*.pth":
            return self._rf_files
        if pattern == "heatwave_convlstm_v*.pth":
            return self._convlstm_files
        return []


class TestModelManagerInit:
    def test_initial_state(self):
        mm = ModelManager()
        assert mm.model is None
        assert mm.metadata == {}
        assert mm.current_path is None
        assert mm.model_type == "unknown"
        assert mm.normalization_mean is None
        assert mm.normalization_std is None

    def test_device_is_set(self):
        mm = ModelManager()
        assert mm.device in (torch.device("cpu"), torch.device("cuda"))


class TestGetLatestCheckpoint:
    def test_returns_none_when_no_checkpoints(self, monkeypatch):
        from src.core.config import settings
        monkeypatch.setattr(settings, "MODELS_DIR", _FakeModelsDir())
        mm = ModelManager()
        result = mm.get_latest_checkpoint()
        assert result is None

    def test_returns_highest_version(self, monkeypatch):
        from src.core.config import settings
        rf_files = [Path(f"D:/virtual/heatwave_model_checkpoint_v{v}.pth") for v in [1, 3, 2]]
        monkeypatch.setattr(settings, "MODELS_DIR", _FakeModelsDir(rf_files=rf_files))

        mm = ModelManager()
        result = mm.get_latest_checkpoint()
        assert result is not None
        assert "v3" in result.name

    def test_prefers_highest_version_across_types(self, monkeypatch):
        from src.core.config import settings
        monkeypatch.setattr(
            settings,
            "MODELS_DIR",
            _FakeModelsDir(
                rf_files=[Path("D:/virtual/heatwave_model_checkpoint_v2.pth")],
                convlstm_files=[Path("D:/virtual/heatwave_convlstm_v5.pth")],
            ),
        )

        mm = ModelManager()
        result = mm.get_latest_checkpoint()
        assert result is not None
        assert "v5" in result.name

    def test_treats_non_numeric_versions_as_zero(self, monkeypatch):
        from src.core.config import settings
        monkeypatch.setattr(
            settings,
            "MODELS_DIR",
            _FakeModelsDir(
                rf_files=[Path("D:/virtual/heatwave_model_checkpoint_vx.pth")],
                convlstm_files=[Path("D:/virtual/heatwave_convlstm_v2.pth")],
            ),
        )

        mm = ModelManager()
        result = mm.get_latest_checkpoint()
        assert result is not None
        assert result.name == "heatwave_convlstm_v2.pth"


class TestDenormalizeTemperature:
    def _make_manager_with_stats(self):
        mm = ModelManager()
        mm.normalization_mean = np.array([[0, 290.0, 0.5, 0, 0, 0, 0, 0]]).reshape(1, 8, 1, 1)
        mm.normalization_std = np.array([[0.1, 10.0, 0.1, 0, 0, 0, 0, 0]]).reshape(1, 8, 1, 1)
        return mm

    def test_denormalize_returns_celsius(self):
        mm = self._make_manager_with_stats()
        # normalized value 0.0 → raw = 290K → -0.15°C after Kelvin conversion
        norm_grid = np.zeros((2, 2))
        result = mm.denormalize_temperature(norm_grid, channel_idx=1)
        assert result.shape == (2, 2)
        # raw = 0*10 + 290 = 290 K → Kelvin auto-converted to Celsius
        assert result[0, 0] == pytest.approx(290.0 - 273.15, abs=0.1)

    def test_no_stats_returns_input_unchanged(self):
        mm = ModelManager()
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = mm.denormalize_temperature(grid)
        np.testing.assert_array_equal(result, grid)

    def test_output_shape_preserved(self):
        mm = self._make_manager_with_stats()
        for shape in [(4, 6), (10, 8)]:
            grid = np.random.randn(*shape).astype(np.float32)
            result = mm.denormalize_temperature(grid, channel_idx=1)
            assert result.shape == shape


class TestPredictEvent:
    def test_returns_error_when_no_model(self):
        mm = ModelManager()
        result = mm.predict_event(np.zeros((1, 8)))
        assert "error" in result

    def test_sklearn_model_returns_probabilities(self):
        """Test with a mocked sklearn-like model."""

        class FakeSklearnModel:
            def predict_proba(self, X):
                return np.column_stack([1 - X[:, 0], X[:, 0]])

        mm = ModelManager()
        mm.model = FakeSklearnModel()
        features = np.array([[0.8]])
        result = mm.predict_event(features)
        assert "probabilities" in result
        assert len(result["probabilities"]) == 1
        assert 0.0 <= result["probabilities"][0] <= 1.0


class TestPredictTemperature:
    def test_raises_without_model(self, random_numpy_sequence):
        mm = ModelManager()
        with pytest.raises(RuntimeError, match="No model loaded"):
            mm.predict_temperature(random_numpy_sequence)

    def test_raises_with_non_convlstm_model(self, random_numpy_sequence):
        class FakeModel:
            pass

        mm = ModelManager()
        mm.model = FakeModel()
        with pytest.raises(RuntimeError, match="ConvLSTM"):
            mm.predict_temperature(random_numpy_sequence)

    def test_returns_correct_shape(self, random_numpy_sequence):
        from src.models.convlstm import HeatwaveConvLSTM

        B, T, C, H, W = random_numpy_sequence.shape
        model = HeatwaveConvLSTM(
            input_dim=C,
            hidden_dim=[8, 8],
            kernel_size=[(3, 3), (3, 3)],
            num_layers=2,
        ).eval()

        mm = ModelManager()
        mm.model = model
        mm.device = torch.device("cpu")

        result = mm.predict_temperature(random_numpy_sequence, future_seq=2)
        assert result.shape == (2, H, W)

    def test_output_no_nan(self, random_numpy_sequence):
        from src.models.convlstm import HeatwaveConvLSTM

        _, _, C, H, W = random_numpy_sequence.shape
        model = HeatwaveConvLSTM(
            input_dim=C,
            hidden_dim=[8, 8],
            kernel_size=[(3, 3), (3, 3)],
            num_layers=2,
        ).eval()

        mm = ModelManager()
        mm.model = model
        mm.device = torch.device("cpu")

        result = mm.predict_temperature(random_numpy_sequence, future_seq=1)
        assert not np.isnan(result).any()
