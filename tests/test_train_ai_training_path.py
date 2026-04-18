import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Train_Ai import _finalize_event_labels, _resolve_model_identity


def test_finalize_event_labels_preserves_precomputed_anomaly_labels():
    # Generic thresholding would mark every sample positive here.
    train_seq = np.full((3, 3, 2, 2), 40.0, dtype=np.float32)
    val_seq = np.full((2, 3, 2, 2), 40.0, dtype=np.float32)
    test_seq = np.full((2, 3, 2, 2), 40.0, dtype=np.float32)

    precomputed = (
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    )

    y_train, y_val, y_test = _finalize_event_labels(
        event_train_seq=train_seq,
        event_val_seq=val_seq,
        event_test_seq=test_seq,
        threshold_c=35.0,
        min_duration=2,
        min_hot_fraction=0.10,
        precomputed_labels=precomputed,
    )

    np.testing.assert_array_equal(y_train, precomputed[0])
    np.testing.assert_array_equal(y_val, precomputed[1])
    np.testing.assert_array_equal(y_test, precomputed[2])


def test_finalize_event_labels_uses_thresholding_without_precomputed_labels():
    # Only second sample reaches the threshold for long enough duration.
    train_seq = np.array(
        [
            [[[30.0, 30.0], [30.0, 30.0]], [[30.0, 30.0], [30.0, 30.0]], [[30.0, 30.0], [30.0, 30.0]]],
            [[[36.0, 36.0], [36.0, 36.0]], [[36.0, 36.0], [36.0, 36.0]], [[36.0, 36.0], [36.0, 36.0]]],
        ],
        dtype=np.float32,
    )

    y_train, _, _ = _finalize_event_labels(
        event_train_seq=train_seq,
        event_val_seq=train_seq,
        event_test_seq=train_seq,
        threshold_c=35.0,
        min_duration=2,
        min_hot_fraction=0.10,
        precomputed_labels=None,
    )

    np.testing.assert_array_equal(y_train, np.array([0, 1], dtype=np.int32))


def test_resolve_model_identity_random_forest_hp_suffix():
    model = RandomForestClassifier()

    identity = _resolve_model_identity(model, hp_tuning_used=False)
    assert identity["model_type"] == "sklearn_random_forest_classifier"
    assert identity["model_backend"] == "random_forest"
    assert identity["model_backend_base"] == "random_forest"

    hp_identity = _resolve_model_identity(model, hp_tuning_used=True)
    assert hp_identity["model_backend"] == "random_forest_hp_tuned"


def test_resolve_model_identity_xgboost_and_lightgbm_by_class_name():
    xgb_like = type("XGBClassifier", (), {})()
    lgb_like = type("LGBMClassifier", (), {})()

    xgb_identity = _resolve_model_identity(xgb_like, hp_tuning_used=False)
    assert xgb_identity["model_type"] == "sklearn_xgboost_classifier"
    assert xgb_identity["model_backend"] == "xgboost"

    lgb_identity = _resolve_model_identity(lgb_like, hp_tuning_used=False)
    assert lgb_identity["model_type"] == "sklearn_lightgbm_classifier"
    assert lgb_identity["model_backend"] == "lightgbm"
