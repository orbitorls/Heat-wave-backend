from __future__ import annotations

from app.ml.forecast.backends.lgbm_backend import (
    _add_monotone_constraints_if_supported,
    _sanitize_lgbm_params_for_device,
)


def test_sanitize_lgbm_params_clamps_opencl_gpu_max_bin() -> None:
    params = {"max_bin": 511, "num_leaves": 63}

    sanitized = _sanitize_lgbm_params_for_device(params, "gpu")

    assert sanitized == {"max_bin": 255, "num_leaves": 63}
    assert params == {"max_bin": 511, "num_leaves": 63}


def test_sanitize_lgbm_params_keeps_supported_gpu_max_bin() -> None:
    assert _sanitize_lgbm_params_for_device({"max_bin": 255}, "gpu") == {"max_bin": 255}


def test_sanitize_lgbm_params_leaves_cpu_and_cuda_unchanged() -> None:
    params = {"max_bin": 511, "learning_rate": 0.03}

    assert _sanitize_lgbm_params_for_device(params, "cpu") == params
    assert _sanitize_lgbm_params_for_device(params, "cuda") == params


def test_add_monotone_constraints_skips_quantile_objective() -> None:
    params = {"objective": "quantile", "metric": "quantile", "alpha": 0.95}
    feature_list = ["temp_c_lag1h", "rh_lag1h", "hour_sin"]

    constrained = _add_monotone_constraints_if_supported(params, feature_list)

    assert "monotone_constraints" not in constrained
    assert params == {"objective": "quantile", "metric": "quantile", "alpha": 0.95}


def test_add_monotone_constraints_removes_existing_constraints_for_quantile() -> None:
    params = {
        "objective": "quantile",
        "metric": "quantile",
        "monotone_constraints": [1, 1, 0],
    }

    constrained = _add_monotone_constraints_if_supported(params, ["temp_c_lag1h"])

    assert "monotone_constraints" not in constrained
    assert params["monotone_constraints"] == [1, 1, 0]


def test_add_monotone_constraints_keeps_supported_objectives() -> None:
    params = {"objective": "regression", "metric": "l2"}
    feature_list = ["temp_c_lag1h", "rh_lag1h", "hour_sin"]

    constrained = _add_monotone_constraints_if_supported(params, feature_list)

    assert constrained["monotone_constraints"] == [1, 1, 0]
    assert "monotone_constraints" not in params
