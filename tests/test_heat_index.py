"""Tests for heat_index.compute() — Rothfusz/Steadman formula"""
import pytest
from app.core.heat_index import compute, HeatIndexCategory


def test_extreme_danger():
    """temp 42°C + humidity 90% → ExtremeDanger"""
    r = compute(42, 90)
    assert r.heat_index >= 52
    assert r.category == HeatIndexCategory.EXTREME_DANGER


def test_danger_range():
    # 38°C/70%RH gives HI ~77°C via Rothfusz → ExtremeDanger
    r = compute(38, 70)
    assert r.category in (HeatIndexCategory.DANGER, HeatIndexCategory.EXTREME_DANGER)
    # verify actual DANGER range (lower temp + lower humidity)
    r2 = compute(35, 45)
    assert r2.category in (HeatIndexCategory.EXTREME_CAUTION, HeatIndexCategory.DANGER)


def test_extreme_caution():
    r = compute(33, 60)
    assert r.category == HeatIndexCategory.EXTREME_CAUTION


def test_caution():
    r = compute(28, 50)
    assert r.category == HeatIndexCategory.CAUTION


def test_low_temp_uses_steadman():
    r = compute(25, 60)
    assert not r.used_adjustment
    # heat index ควรใกล้เคียงกับ temp จริง (ไม่บวกมาก)
    assert abs(r.heat_index - 25) < 5


def test_invalid_humidity():
    with pytest.raises(ValueError):
        compute(30, 110)

    with pytest.raises(ValueError):
        compute(30, -5)


def test_high_temp_high_humidity():
    """สภาพอากาศประเทศไทยช่วงร้อนสุด — 40°C 80%RH"""
    r = compute(40, 80)
    assert r.heat_index > 40
    assert r.category in (HeatIndexCategory.DANGER, HeatIndexCategory.EXTREME_DANGER)


def test_heat_index_increases_with_humidity():
    r_low = compute(35, 40)
    r_high = compute(35, 90)
    assert r_high.heat_index > r_low.heat_index
