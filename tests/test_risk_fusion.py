"""Tests for risk-fusion module — fuse_risk()

ครอบคลุม rule-table จาก plan:
- q50/q90 thresholds (Danger / Danger Watch)
- P(Danger) gate
- LOCAL_ANOMALY_P95 reason code
- horizon_type (operational vs awareness)
- uncertainty_flag จาก PI width
- custom thresholds override
"""
from __future__ import annotations

import pytest

from app.core.risk_fusion import (
    DEFAULT_THRESHOLDS,
    RiskOutput,
    fuse_risk,
)


# ---------------------------------------------------------------------------
# Rule-table: q50 / q90 thresholds
# ---------------------------------------------------------------------------

def test_q50_at_or_above_42_is_danger():
    """q50=43, q90=44, h=24 → Danger, operational, reason includes Q50_DANGER_THRESHOLD"""
    out = fuse_risk(
        hi_q50=43.0,
        hi_q90=44.0,
        p_high_watch=0.5,
        p_danger=0.3,
        hi_anomaly_p95_excess=0.0,
        horizon_h=24,
        pi_width=1.5,
    )
    assert out.risk_level == "Danger"
    assert out.horizon_type == "operational"
    assert "Q50_DANGER_THRESHOLD" in out.reason_codes


def test_q90_above_42_with_q50_below_is_danger_watch():
    """q50=38, q90=43, h=24 → Danger Watch (Watch level)"""
    out = fuse_risk(
        hi_q50=38.0,
        hi_q90=43.0,
        p_high_watch=0.3,
        p_danger=0.1,
        hi_anomaly_p95_excess=0.0,
        horizon_h=24,
        pi_width=1.5,
    )
    # "Danger Watch" maps to Watch in our 5-level scale
    assert out.risk_level in ("Watch", "Warning")
    assert "Q90_DANGER_THRESHOLD" in out.reason_codes


# ---------------------------------------------------------------------------
# Rule-table: P(Danger) gate
# ---------------------------------------------------------------------------

def test_p_danger_above_gate_at_long_horizon_is_awareness():
    """q50=35, q90=37, h=72, p_danger=0.4 (> gate=0.20) → Watch/Warning, awareness"""
    out = fuse_risk(
        hi_q50=35.0,
        hi_q90=37.0,
        p_high_watch=0.3,
        p_danger=0.4,
        hi_anomaly_p95_excess=0.0,
        horizon_h=72,
        pi_width=1.5,
    )
    assert out.risk_level in ("Watch", "Warning")
    assert out.horizon_type == "awareness"
    assert "P_DANGER_GATE" in out.reason_codes


# ---------------------------------------------------------------------------
# LOCAL_ANOMALY_P95 reason code
# ---------------------------------------------------------------------------

def test_local_anomaly_added_even_at_lower_risk():
    """hi_anomaly_p95_excess > 0 → reason includes LOCAL_ANOMALY_P95 even at lower levels"""
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=1.5,  # 1.5°C above station-month-hour p95
        horizon_h=12,
        pi_width=1.0,
    )
    assert "LOCAL_ANOMALY_P95" in out.reason_codes
    # Should still be a low / moderate level — anomaly alone shouldn't escalate to Danger
    assert out.risk_level in ("Normal", "Caution", "Watch")


# ---------------------------------------------------------------------------
# Uncertainty flag from PI width
# ---------------------------------------------------------------------------

def test_pi_width_high_yields_high_uncertainty():
    """pi_width=5.0 → uncertainty_flag='high'"""
    out = fuse_risk(
        hi_q50=35.0,
        hi_q90=38.0,
        p_high_watch=0.2,
        p_danger=0.05,
        hi_anomaly_p95_excess=0.0,
        horizon_h=24,
        pi_width=5.0,
    )
    assert out.uncertainty_flag == "high"


def test_pi_width_low_yields_low_uncertainty():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=12,
        pi_width=1.5,
    )
    assert out.uncertainty_flag == "low"


def test_pi_width_medium_yields_medium_uncertainty():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=12,
        pi_width=3.0,
    )
    assert out.uncertainty_flag == "medium"


# ---------------------------------------------------------------------------
# Nominal / Normal case
# ---------------------------------------------------------------------------

def test_all_nominal_inputs_produce_normal():
    """q50=30, q90=33, p_danger=0.05, p_high_watch=0.1, anomaly=0, h=12, pi_width=1.5 → Normal"""
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.05,
        hi_anomaly_p95_excess=0.0,
        horizon_h=12,
        pi_width=1.5,
    )
    assert out.risk_level == "Normal"
    assert out.horizon_type == "operational"
    assert out.uncertainty_flag == "low"
    assert "LOCAL_ANOMALY_P95" not in out.reason_codes
    assert 0.0 <= out.confidence <= 1.0


# ---------------------------------------------------------------------------
# Custom thresholds override
# ---------------------------------------------------------------------------

def test_custom_thresholds_override_defaults():
    """Custom thresholds dict overrides defaults — lower q50 cutoff makes 35 a Danger"""
    custom = {"q50_danger": 34.0}
    out = fuse_risk(
        hi_q50=35.0,
        hi_q90=36.0,
        p_high_watch=0.2,
        p_danger=0.05,
        hi_anomaly_p95_excess=0.0,
        horizon_h=24,
        pi_width=1.5,
        thresholds=custom,
    )
    assert out.risk_level == "Danger"
    assert "Q50_DANGER_THRESHOLD" in out.reason_codes


def test_custom_p_danger_gate_can_be_relaxed():
    """Raise p_danger gate so that p=0.4 no longer trips it"""
    custom = {"p_danger_gate": 0.5}
    out = fuse_risk(
        hi_q50=35.0,
        hi_q90=37.0,
        p_high_watch=0.3,
        p_danger=0.4,
        hi_anomaly_p95_excess=0.0,
        horizon_h=72,
        pi_width=1.5,
        thresholds=custom,
    )
    assert "P_DANGER_GATE" not in out.reason_codes


# ---------------------------------------------------------------------------
# horizon_type from horizon_h
# ---------------------------------------------------------------------------

def test_horizon_48_is_awareness():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=48,
        pi_width=1.5,
    )
    assert out.horizon_type == "awareness"


def test_horizon_24_is_operational():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=24,
        pi_width=1.5,
    )
    assert out.horizon_type == "operational"


def test_horizon_72_is_awareness():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=72,
        pi_width=1.5,
    )
    assert out.horizon_type == "awareness"


# ---------------------------------------------------------------------------
# DEFAULT_THRESHOLDS shape
# ---------------------------------------------------------------------------

def test_default_thresholds_has_required_keys():
    required = {
        "q50_danger",
        "q90_danger",
        "p_danger_gate",
        "p_high_watch_gate",
        "pi_width_low",
        "pi_width_high",
        "awareness_horizons",
    }
    assert required.issubset(DEFAULT_THRESHOLDS.keys())
    assert DEFAULT_THRESHOLDS["q50_danger"] == 42.0
    assert DEFAULT_THRESHOLDS["q90_danger"] == 42.0
    assert DEFAULT_THRESHOLDS["p_danger_gate"] == 0.20
    assert DEFAULT_THRESHOLDS["p_high_watch_gate"] == 0.40
    assert DEFAULT_THRESHOLDS["pi_width_low"] == 2.0
    assert DEFAULT_THRESHOLDS["pi_width_high"] == 4.0
    assert 48 in DEFAULT_THRESHOLDS["awareness_horizons"]
    assert 72 in DEFAULT_THRESHOLDS["awareness_horizons"]


def test_returns_risk_output_dataclass():
    out = fuse_risk(
        hi_q50=30.0,
        hi_q90=33.0,
        p_high_watch=0.1,
        p_danger=0.02,
        hi_anomaly_p95_excess=0.0,
        horizon_h=12,
        pi_width=1.5,
    )
    assert isinstance(out, RiskOutput)
    assert out.risk_level in {"Normal", "Caution", "Watch", "Warning", "Danger"}
    assert out.horizon_type in {"operational", "awareness"}
    assert out.uncertainty_flag in {"low", "medium", "high"}
    assert isinstance(out.reason_codes, list)
