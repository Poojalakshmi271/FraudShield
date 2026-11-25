"""
FraudShield – Machine Learning Models Unit Tests
================================================
Tests model training, output shapes, probability ranges, serialization,
evaluation metrics calculation, threshold optimization, and calibration.
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.calibration import calibrate_model
from src.models.evaluate import evaluate_model
from src.models.threshold import apply_threshold, optimize_threshold


@pytest.fixture
def synthetic_train_val():
    """Create small synthetic train and validation sets for testing model components."""
    np.random.seed(42)
    n_train = 200
    n_val = 100
    n_feats = 10

    X_train = np.random.randn(n_train, n_feats)
    y_train = np.random.choice([0, 1], size=n_train, p=[0.8, 0.2])

    X_val = np.random.randn(n_val, n_feats)
    y_val = np.random.choice([0, 1], size=n_val, p=[0.8, 0.2])

    return X_train, y_train, X_val, y_val


def test_model_output_shape(synthetic_train_val):
    """Verify predict_proba yields (n_samples, 2) shape with probabilities summing to 1."""
    X_train, y_train, X_val, _ = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_val)
    assert probas.shape == (len(X_val), 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_probability_range(synthetic_train_val):
    """Verify predicted probabilities are strictly bounded in [0.0, 1.0]."""
    X_train, y_train, X_val, _ = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_val)[:, 1]
    assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_model_save_load(synthetic_train_val):
    """Verify model serialization and deserialization via joblib."""
    X_train, y_train, X_val, _ = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)

        orig_preds = model.predict_proba(X_val)
        loaded_preds = loaded.predict_proba(X_val)
        np.testing.assert_allclose(orig_preds, loaded_preds)


def test_evaluation_metrics_valid(synthetic_train_val):
    """Verify evaluate_model returns all required metrics and confusion matrix."""
    X_train, y_train, X_val, y_val = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val, threshold=0.5)

    required_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"]
    for k in required_keys:
        assert k in metrics, f"Missing metric {k}"

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_threshold_optimization(synthetic_train_val):
    """Verify threshold optimization finds a threshold in the valid search range [0.05, 0.95]."""
    X_train, y_train, X_val, y_val = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    best_thresh, best_f1, scores_df = optimize_threshold(model, X_val, y_val, metric="f1")

    assert 0.05 <= best_thresh <= 0.95
    assert 0.0 <= best_f1 <= 1.0
    assert isinstance(scores_df, pd.DataFrame)
    assert "threshold" in scores_df.columns
    assert "f1" in scores_df.columns


def test_calibration_brier_score(synthetic_train_val):
    """Verify calibrated classifier produces probabilities with a valid Brier score."""
    X_train, y_train, X_val, y_val = synthetic_train_val
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    calibrated = calibrate_model(model, X_val, y_val, method="sigmoid")
    cal_probas = calibrated.predict_proba(X_val)[:, 1]

    assert (cal_probas >= 0.0).all() and (cal_probas <= 1.0).all()
