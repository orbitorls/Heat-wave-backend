"""Unit tests for src/data/loader.py — data utilities and preprocessing."""

import numpy as np
import pytest
import xarray as xr

from src.data.loader import (
    DataLoader,
    clean_data,
    compute_normalization_stats,
    create_sequences,
    fill_nan_along_time,
    normalize_data,
)


# ────────────────────────────────────────────────────────────────────────────
# fill_nan_along_time
# ────────────────────────────────────────────────────────────────────────────


class TestFillNanAlongTime:
    def test_no_nan_unchanged(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]]])
        result = fill_nan_along_time(arr)
        np.testing.assert_array_almost_equal(result, arr)

    def test_interpolates_interior_nan(self):
        arr = np.zeros((5, 1, 1))
        arr[:, 0, 0] = [1.0, np.nan, 3.0, np.nan, 5.0]
        result = fill_nan_along_time(arr)
        assert not np.isnan(result).any()
        assert result[1, 0, 0] == pytest.approx(2.0, abs=0.1)
        assert result[3, 0, 0] == pytest.approx(4.0, abs=0.1)

    def test_all_nan_column_filled_with_zero(self):
        arr = np.full((4, 2, 2), np.nan)
        result = fill_nan_along_time(arr)
        assert not np.isnan(result).any()
        assert np.all(result == 0.0)

    def test_single_valid_value_broadcast(self):
        arr = np.full((5, 1, 1), np.nan)
        arr[2, 0, 0] = 42.0
        result = fill_nan_along_time(arr)
        assert not np.isnan(result).any()
        assert np.all(result == 42.0)

    def test_preserves_shape(self):
        arr = np.random.randn(10, 4, 6)
        arr[3, 1, 2] = np.nan
        result = fill_nan_along_time(arr)
        assert result.shape == arr.shape


# ────────────────────────────────────────────────────────────────────────────
# compute_normalization_stats
# ────────────────────────────────────────────────────────────────────────────


class TestComputeNormalizationStats:
    def test_output_shape(self, random_data_array):
        C = random_data_array.shape[1]
        mean, std = compute_normalization_stats(random_data_array)
        assert mean.shape == (1, C, 1, 1)
        assert std.shape == (1, C, 1, 1)

    def test_constant_channel_zero_std(self):
        data = np.ones((10, 4, 6, 6), dtype=np.float32)
        data[:, 1] = 5.0
        mean, std = compute_normalization_stats(data)
        assert mean[0, 0, 0, 0] == pytest.approx(1.0)
        assert std[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_known_values(self):
        data = np.zeros((4, 1, 2, 2), dtype=np.float32)
        # Per-channel stats are computed over (time, lat, lon), not just a single pixel.
        data[:, 0, 0, 0] = [0, 1, 2, 3]
        mean, std = compute_normalization_stats(data)
        expected_mean = float(data[:, 0, :, :].mean())
        expected_std = float(data[:, 0, :, :].std())
        assert mean[0, 0, 0, 0] == pytest.approx(expected_mean, abs=1e-6)
        assert std[0, 0, 0, 0] == pytest.approx(expected_std, abs=1e-6)


# ────────────────────────────────────────────────────────────────────────────
# normalize_data
# ────────────────────────────────────────────────────────────────────────────


class TestNormalizeData:
    def test_zero_mean_unit_variance(self, random_data_array):
        mean, std = compute_normalization_stats(random_data_array)
        normalized = normalize_data(random_data_array, mean, std)
        # Per-channel means should be ~0
        ch_means = normalized.mean(axis=(0, 2, 3))
        np.testing.assert_allclose(ch_means, 0.0, atol=1e-4)

    def test_output_shape_preserved(self, random_data_array):
        mean, std = compute_normalization_stats(random_data_array)
        out = normalize_data(random_data_array, mean, std)
        assert out.shape == random_data_array.shape

    def test_no_nan_output(self, random_data_array):
        mean, std = compute_normalization_stats(random_data_array)
        out = normalize_data(random_data_array, mean, std)
        assert not np.isnan(out).any()


# ────────────────────────────────────────────────────────────────────────────
# clean_data
# ────────────────────────────────────────────────────────────────────────────


class TestCleanData:
    def test_removes_nan(self, random_data_array):
        dirty = random_data_array.copy()
        dirty[0, 0, 0, 0] = np.nan
        cleaned, _ = clean_data(dirty)
        assert not np.isnan(cleaned).any()

    def test_removes_inf(self, random_data_array):
        dirty = random_data_array.copy()
        dirty[0, 0, 0, 0] = np.inf
        dirty[1, 1, 1, 1] = -np.inf
        cleaned, _ = clean_data(dirty)
        assert np.isfinite(cleaned).all()

    def test_clips_outliers(self):
        data = np.zeros((10, 1, 4, 4), dtype=np.float32)
        data[:] = 1.0
        data[0, 0, 0, 0] = 1000.0  # extreme outlier
        cleaned, _ = clean_data(data, clip_percentiles=(0.5, 99.5))
        assert cleaned.max() < 1000.0

    def test_output_shape_preserved(self, random_data_array):
        cleaned, _ = clean_data(random_data_array)
        assert cleaned.shape == random_data_array.shape

    def test_returns_clip_bounds(self, random_data_array):
        _, bounds = clean_data(random_data_array)
        lower, upper = bounds
        assert lower is not None
        assert upper is not None

    def test_custom_clip_bounds(self, random_data_array):
        lower = np.full_like(random_data_array[:1, :1, :1, :1], -2.0)
        upper = np.full_like(random_data_array[:1, :1, :1, :1], 2.0)
        cleaned, _ = clean_data(random_data_array, clip_bounds=(lower, upper))
        assert cleaned.min() >= -2.0
        assert cleaned.max() <= 2.0


# ────────────────────────────────────────────────────────────────────────────
# create_sequences
# ────────────────────────────────────────────────────────────────────────────


class TestCreateSequences:
    def test_output_shapes(self, random_data_array):
        seq_len, pred_len = 7, 3
        X, Y = create_sequences(random_data_array, seq_len=seq_len, pred_len=pred_len)
        T = random_data_array.shape[0]
        expected_samples = T - (seq_len + pred_len) + 1
        assert X.shape[0] == expected_samples
        assert Y.shape[0] == expected_samples
        assert X.shape[1] == seq_len
        assert Y.shape[1] == pred_len

    def test_channel_and_spatial_dims(self, random_data_array):
        C, H, W = random_data_array.shape[1:]
        X, Y = create_sequences(random_data_array, seq_len=5, pred_len=2)
        assert X.shape[2:] == (C, H, W)
        assert Y.shape[2:] == (C, H, W)

    def test_empty_when_too_short(self):
        data = np.random.randn(5, 4, 6, 6).astype(np.float32)
        X, Y = create_sequences(data, seq_len=4, pred_len=3)  # 5 - 7 + 1 = -1
        assert X.size == 0
        assert Y.size == 0

    def test_values_are_contiguous_windows(self):
        data = np.arange(20 * 4 * 2 * 2, dtype=np.float32).reshape(20, 4, 2, 2)
        X, Y = create_sequences(data, seq_len=5, pred_len=3)
        # X[0] should be data[0:5], Y[0] should be data[5:8]
        np.testing.assert_array_equal(X[0], data[0:5])
        np.testing.assert_array_equal(Y[0], data[5:8])
        # X[1] should be data[1:6]
        np.testing.assert_array_equal(X[1], data[1:6])

    def test_no_data_leakage_between_x_and_y(self, random_data_array):
        seq_len, pred_len = 5, 3
        X, Y = create_sequences(random_data_array, seq_len=seq_len, pred_len=pred_len)
        # X[i] and Y[i] should not share time steps
        for i in range(min(3, X.shape[0])):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(X[i, -1], Y[i, 0])


def _make_base_coords():
    return {
        "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
        "latitude": np.array([13.0, 14.0], dtype=np.float32),
        "longitude": np.array([100.0, 101.0], dtype=np.float32),
    }


def _make_era5_dataset():
    coords = _make_base_coords()
    t2m = np.full((2, 2, 2), 300.0, dtype=np.float32)
    return xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), t2m)},
        coords=coords,
    )


def _make_nasa_dataset():
    coords = _make_base_coords()
    t2m_c = np.full((2, 2, 2), 30.0, dtype=np.float32)
    rh = np.full((2, 2, 2), 80.0, dtype=np.float32)
    tp = np.full((2, 2, 2), 2.0, dtype=np.float32)
    dew = np.full((2, 2, 2), 20.0, dtype=np.float32)
    return xr.Dataset(
        {
            "T2M": (("time", "latitude", "longitude"), t2m_c),
            "RH2M": (("time", "latitude", "longitude"), rh),
            "PRECTOTCORR": (("time", "latitude", "longitude"), tp),
            "T2MDEW": (("time", "latitude", "longitude"), dew),
        },
        coords=coords,
    )


class TestLoadCombined:
    def test_merges_and_applies_nasa_renames_and_temperature_conversion(self, monkeypatch):
        loader = DataLoader()
        era5_ds = _make_era5_dataset()
        nasa_ds = _make_nasa_dataset()
        monkeypatch.setattr(loader, "load_era5", lambda year=None: era5_ds)
        monkeypatch.setattr(loader, "load_nasa_power", lambda year=None: nasa_ds)

        combined = loader.load_combined(year=2024)

        assert "t2m" in combined
        assert "t2m_nasa" in combined
        assert "humidity" in combined
        assert "tp" in combined
        assert "d2m_nasa" in combined
        np.testing.assert_allclose(combined["t2m_nasa"].values, nasa_ds["T2M"].values + 273.15)
        np.testing.assert_allclose(combined["d2m_nasa"].values, nasa_ds["T2MDEW"].values + 273.15)

    def test_returns_era5_when_nasa_power_is_empty(self, monkeypatch):
        loader = DataLoader()
        era5_ds = _make_era5_dataset()
        monkeypatch.setattr(loader, "load_era5", lambda year=None: era5_ds)
        monkeypatch.setattr(loader, "load_nasa_power", lambda year=None: xr.Dataset())

        combined = loader.load_combined()
        assert combined.identical(era5_ds)

    def test_returns_nasa_power_when_era5_is_empty(self, monkeypatch):
        loader = DataLoader()
        nasa_ds = _make_nasa_dataset()
        monkeypatch.setattr(loader, "load_era5", lambda year=None: xr.Dataset())
        monkeypatch.setattr(loader, "load_nasa_power", lambda year=None: nasa_ds)

        combined = loader.load_combined()
        assert combined.identical(nasa_ds)
