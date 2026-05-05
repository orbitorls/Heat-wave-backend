"""Tests for app.core.calibration — bias calibration on validation residuals.

ทดสอบ 4 ระดับการ calibrate (L1/L2/L3/L4) + selector + serialization.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.calibration import (
    Calibration,
    fit_calibration,
    select_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

STATIONS = ["BKK_01", "CNX_01", "KKN_01"]
HORIZONS = [6, 24, 48]
RNG = np.random.default_rng(42)


def _hour_bucket(h: int) -> int:
    if 0 <= h <= 5:
        return 0
    if 6 <= h <= 11:
        return 1
    if 12 <= h <= 17:
        return 2
    return 3


def _make_grid(n_per_group: int = 60, seed: int = 0):
    """สร้างกริดของ (station, horizon, hour) ขนาด n ตัวอย่างต่อกลุ่ม"""
    rng = np.random.default_rng(seed)
    stations: list[str] = []
    horizons: list[int] = []
    hours: list[int] = []
    for s in STATIONS:
        for h in HORIZONS:
            for _ in range(n_per_group):
                stations.append(s)
                horizons.append(h)
                hours.append(int(rng.integers(0, 24)))
    return (
        np.array(stations),
        np.array(horizons, dtype=int),
        np.array(hours, dtype=int),
    )


# ---------------------------------------------------------------------------
# L1 — mean shift per station
# ---------------------------------------------------------------------------

def test_l1_per_station_mean_shift():
    """+1.5°C bias for one station → calibrated mean residual ≈ 0"""
    stations, horizons, hours = _make_grid(n_per_group=40, seed=1)
    rng = np.random.default_rng(1)
    y_true = 30.0 + rng.normal(0, 0.5, size=stations.shape[0])
    bias_map = {"BKK_01": 1.5, "CNX_01": -2.0, "KKN_01": 0.0}
    y_pred = y_true + np.array([bias_map[s] for s in stations])

    calib = fit_calibration(
        level="L1",
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )
    y_corr = calib.apply(y_pred, stations, horizons, hours)

    for s in STATIONS:
        mask = stations == s
        residual = (y_corr[mask] - y_true[mask]).mean()
        assert abs(residual) < 0.05, f"L1 residual for {s}: {residual:.3f}"


# ---------------------------------------------------------------------------
# L2 — mean shift per (station, horizon)
# ---------------------------------------------------------------------------

def test_l2_per_station_horizon_shift():
    stations, horizons, hours = _make_grid(n_per_group=50, seed=2)
    rng = np.random.default_rng(2)
    y_true = 30.0 + rng.normal(0, 0.5, size=stations.shape[0])

    # bias varies BY horizon within each station
    bias = np.zeros_like(y_true)
    for s in STATIONS:
        for h in HORIZONS:
            mask = (stations == s) & (horizons == h)
            bias[mask] = 0.5 * h * (1 if s == "BKK_01" else -1)
    y_pred = y_true + bias

    calib = fit_calibration(
        level="L2",
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )
    y_corr = calib.apply(y_pred, stations, horizons, hours)

    for s in STATIONS:
        for h in HORIZONS:
            mask = (stations == s) & (horizons == h)
            res = (y_corr[mask] - y_true[mask]).mean()
            assert abs(res) < 0.05, f"L2 residual for {s},h={h}: {res:.3f}"


# ---------------------------------------------------------------------------
# L3 — mean shift per (station, horizon, hour_bucket)
# ---------------------------------------------------------------------------

def test_l3_per_station_horizon_bucket_shift():
    stations, horizons, hours = _make_grid(n_per_group=120, seed=3)
    rng = np.random.default_rng(3)
    y_true = 30.0 + rng.normal(0, 0.3, size=stations.shape[0])

    bucket_bias = {0: -1.5, 1: 0.5, 2: 2.0, 3: -0.3}
    bias = np.array([bucket_bias[_hour_bucket(int(h))] for h in hours])
    y_pred = y_true + bias

    calib = fit_calibration(
        level="L3",
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )
    y_corr = calib.apply(y_pred, stations, horizons, hours)

    for s in STATIONS:
        for h in HORIZONS:
            for b in (0, 1, 2, 3):
                hb = np.array([_hour_bucket(int(hr)) for hr in hours])
                mask = (stations == s) & (horizons == h) & (hb == b)
                if mask.sum() == 0:
                    continue
                res = (y_corr[mask] - y_true[mask]).mean()
                assert abs(res) < 0.1, (
                    f"L3 residual for {s},h={h},bucket={b}: {res:.3f}"
                )


# ---------------------------------------------------------------------------
# L4 — isotonic regression beats L1 on non-linear bias
# ---------------------------------------------------------------------------

def test_l4_isotonic_beats_l1_on_nonlinear_bias():
    """quadratic-ish bias in raw → isotonic should win"""
    stations, horizons, hours = _make_grid(n_per_group=200, seed=4)
    rng = np.random.default_rng(4)

    # use a single station/horizon group for clarity but full data is fine
    y_true = 25.0 + rng.uniform(0, 15, size=stations.shape[0])
    # non-linear bias monotone in y_true: y_pred = y_true - 0.1*(y_true-30)^2 - 0.5
    y_pred = y_true - 0.1 * (y_true - 30.0) ** 2 - 0.5
    # add tiny noise
    y_pred = y_pred + rng.normal(0, 0.1, size=y_true.shape[0])

    # split train/val deterministically
    n = y_true.size
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = n // 2
    tr, va = idx[:cut], idx[cut:]

    calib_l1 = fit_calibration(
        level="L1",
        y_true=y_true[tr],
        y_pred=y_pred[tr],
        station_ids=stations[tr],
        horizons=horizons[tr],
        hours=hours[tr],
    )
    calib_l4 = fit_calibration(
        level="L4",
        y_true=y_true[tr],
        y_pred=y_pred[tr],
        station_ids=stations[tr],
        horizons=horizons[tr],
        hours=hours[tr],
    )

    mae_l1 = np.mean(np.abs(
        calib_l1.apply(y_pred[va], stations[va], horizons[va], hours[va]) - y_true[va]
    ))
    mae_l4 = np.mean(np.abs(
        calib_l4.apply(y_pred[va], stations[va], horizons[va], hours[va]) - y_true[va]
    ))
    assert mae_l4 < mae_l1, f"L4 ({mae_l4:.3f}) should beat L1 ({mae_l1:.3f})"


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def test_selector_picks_lowest_val_mae():
    """Generate data where L2 is the right level; selector must pick L2."""
    stations, horizons, hours = _make_grid(n_per_group=80, seed=5)
    rng = np.random.default_rng(5)
    n = stations.shape[0]

    y_true_all = 30.0 + rng.normal(0, 0.4, size=n)
    bias = np.zeros(n)
    for s in STATIONS:
        for h in HORIZONS:
            mask = (stations == s) & (horizons == h)
            bias[mask] = 0.4 * h + (1.0 if s == "CNX_01" else -0.5)
    y_pred_all = y_true_all + bias

    # train/val split
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = n // 2
    tr, va = idx[:cut], idx[cut:]

    chosen = select_calibration(
        y_true_train=y_true_all[tr],
        y_pred_train=y_pred_all[tr],
        station_ids_train=stations[tr],
        horizons_train=horizons[tr],
        hours_train=hours[tr],
        y_true_val=y_true_all[va],
        y_pred_val=y_pred_all[va],
        station_ids_val=stations[va],
        horizons_val=horizons[va],
        hours_val=hours[va],
    )

    # MAE on val for each candidate
    levels = ("L1", "L2", "L3", "L4")
    maes = {}
    for lvl in levels:
        c = fit_calibration(
            level=lvl,
            y_true=y_true_all[tr],
            y_pred=y_pred_all[tr],
            station_ids=stations[tr],
            horizons=horizons[tr],
            hours=hours[tr],
        )
        pred = c.apply(y_pred_all[va], stations[va], horizons[va], hours[va])
        maes[lvl] = float(np.mean(np.abs(pred - y_true_all[va])))

    best_lvl = min(maes, key=maes.get)
    assert chosen.level == best_lvl, f"selector chose {chosen.level}; best={best_lvl} maes={maes}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4"])
def test_serialization_round_trip(level):
    stations, horizons, hours = _make_grid(n_per_group=80, seed=6)
    rng = np.random.default_rng(6)
    y_true = 30.0 + rng.normal(0, 0.5, size=stations.shape[0])
    y_pred = y_true + rng.normal(0.5, 0.2, size=stations.shape[0])

    calib = fit_calibration(
        level=level,
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )

    payload = calib.to_json()
    import json
    # ensure JSON-serializable (numpy types must be converted)
    json.dumps(payload)

    restored = Calibration.from_json(payload)

    a = calib.apply(y_pred, stations, horizons, hours)
    b = restored.apply(y_pred, stations, horizons, hours)
    np.testing.assert_allclose(a, b, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_l4_small_group_falls_back_to_identity():
    """A (station, horizon) group with <30 samples in L4 must apply identity."""
    # 3 stations × 3 horizons; one combo has only 5 samples
    stations = []
    horizons = []
    hours = []
    rng = np.random.default_rng(7)

    big_n = 80
    small_n = 5
    for s in STATIONS:
        for h in HORIZONS:
            n = small_n if (s == "BKK_01" and h == 6) else big_n
            for _ in range(n):
                stations.append(s)
                horizons.append(h)
                hours.append(int(rng.integers(0, 24)))
    stations = np.array(stations)
    horizons = np.array(horizons, dtype=int)
    hours = np.array(hours, dtype=int)

    y_true = 30.0 + rng.normal(0, 0.4, size=stations.shape[0])
    y_pred = y_true + 1.0  # uniform +1 bias

    calib = fit_calibration(
        level="L4",
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )
    y_corr = calib.apply(y_pred, stations, horizons, hours)

    small_mask = (stations == "BKK_01") & (horizons == 6)
    np.testing.assert_allclose(
        y_corr[small_mask], y_pred[small_mask], atol=1e-10,
        err_msg="small L4 group should fall back to identity",
    )


def test_empty_input_raises():
    empty = np.array([])
    with pytest.raises(ValueError):
        fit_calibration(
            level="L1",
            y_true=empty,
            y_pred=empty,
            station_ids=np.array([], dtype=object),
            horizons=np.array([], dtype=int),
            hours=np.array([], dtype=int),
        )


def test_invalid_level_raises():
    stations, horizons, hours = _make_grid(n_per_group=10, seed=8)
    y_true = np.zeros(stations.shape[0])
    y_pred = np.zeros(stations.shape[0])
    with pytest.raises(ValueError):
        fit_calibration(
            level="L7",  # type: ignore[arg-type]
            y_true=y_true,
            y_pred=y_pred,
            station_ids=stations,
            horizons=horizons,
            hours=hours,
        )


def test_apply_unseen_group_passes_through():
    """If a group is unseen at fit time, apply should not crash — pass-through."""
    stations, horizons, hours = _make_grid(n_per_group=40, seed=9)
    rng = np.random.default_rng(9)
    y_true = 30.0 + rng.normal(0, 0.5, size=stations.shape[0])
    y_pred = y_true + 1.0

    calib = fit_calibration(
        level="L1",
        y_true=y_true,
        y_pred=y_pred,
        station_ids=stations,
        horizons=horizons,
        hours=hours,
    )

    new_st = np.array(["UNSEEN_X"] * 5)
    new_h = np.array([24] * 5, dtype=int)
    new_hr = np.array([12] * 5, dtype=int)
    new_pred = np.array([28.0, 29.0, 30.0, 31.0, 32.0])
    out = calib.apply(new_pred, new_st, new_h, new_hr)
    np.testing.assert_allclose(out, new_pred, atol=1e-10)
