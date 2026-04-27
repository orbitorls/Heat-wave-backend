"""Per-lead-time and probabilistic evaluation metrics for heatwave forecasting."""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def per_lead_time_metrics(
    y_true: np.ndarray,  # shape (N, lead_times, ...)
    y_pred: np.ndarray,  # shape (N, lead_times, ...)
    lead_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute RMSE and MAE for each lead-time group.

    Args:
        y_true: Ground truth array, first axis = samples, second = lead time steps.
        y_pred: Predictions with same shape.
        lead_groups: dict mapping group name → list of 0-based lead-time indices.
                     Defaults to {"day1": [0], "day2_3": [1,2], "day4_7": [3,4,5,6]}.

    Returns:
        dict mapping group name → {"rmse": float, "mae": float, "n_steps": int}
    """
    if lead_groups is None:
        lead_groups = {
            "day1": [0],
            "day2_3": [1, 2],
            "day4_7": [3, 4, 5, 6],
        }

    results: Dict[str, Dict[str, float]] = {}
    for group, indices in lead_groups.items():
        # clip indices to available lead times
        valid = [i for i in indices if i < y_true.shape[1]]
        if not valid:
            continue
        t = y_true[:, valid].reshape(-1)
        p = y_pred[:, valid].reshape(-1)
        results[group] = {
            "rmse": rmse(t, p),
            "mae": mae(t, p),
            "n_steps": len(valid),
        }
        LOGGER.info("Lead group %s: RMSE=%.4f MAE=%.4f", group, results[group]["rmse"], results[group]["mae"])

    return results


def crps_ensemble(
    y_true: np.ndarray,  # (N,)
    y_pred_mean: np.ndarray,  # (N,)
    y_pred_std: np.ndarray,  # (N,) — predicted std dev
) -> float:
    """
    Approximate CRPS for a Gaussian predictive distribution.
    CRPS(N(mu, sigma), y) = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.
    """
    from scipy import stats as scipy_stats  # optional dep

    sigma = np.clip(y_pred_std, 1e-6, None)
    z = (y_true - y_pred_mean) / sigma
    crps_vals = sigma * (
        z * (2 * scipy_stats.norm.cdf(z) - 1)
        + 2 * scipy_stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_vals))


def skill_score(metric_model: float, metric_baseline: float) -> float:
    """Skill score: 1 - (model_error / baseline_error). Positive = better than baseline."""
    if metric_baseline == 0:
        return 0.0
    return float(1.0 - metric_model / metric_baseline)


def print_metrics_report(results: Dict[str, Dict[str, float]], title: str = "Lead-Time Metrics") -> None:
    """Print a formatted metrics report to stdout."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for group, m in results.items():
        print(f"  {group:12s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  (steps={m['n_steps']})")
    print(f"{'='*50}\n")
