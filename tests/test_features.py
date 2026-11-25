"""
FraudShield – Feature Engineering Unit Tests
============================================
Tests preprocessing, scaling, cyclical hour encoding, and SMOTE resampling.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import FraudPreprocessor
from src.features.resampler import apply_smote, compare_class_distributions


@pytest.fixture
def sample_feature_df():
    """Create a 100-row synthetic raw feature DataFrame."""
    np.random.seed(42)
    n = 100
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(50, n)
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    return {
        "preprocessing": {
            "amount_scaler": "RobustScaler",
            "time_modulo": 86400,
            "log_amount": True,
            "cyclical_encoding": True,
        },
        "resampling": {
            "smote_random_state": 42,
            "smote_k_neighbors": 3,
            "sampling_strategy": "auto",
        },
    }


def test_preprocessor_fit_transform(sample_feature_df, mock_config):
    """Verify preprocessor produces expected columns without altering PCA features."""
    preprocessor = FraudPreprocessor(config=mock_config)
    X_trans = preprocessor.fit_transform(sample_feature_df)

    expected_cols = [f"V{i}" for i in range(1, 29)] + ["hour_sin", "hour_cos", "amount_scaled"]
    assert list(X_trans.columns) == expected_cols
    assert len(X_trans) == len(sample_feature_df)
    # Check that PCA features were preserved exactly
    np.testing.assert_allclose(X_trans["V1"], sample_feature_df["V1"])


def test_preprocessor_no_data_leakage(sample_feature_df, mock_config):
    """Verify preprocessor fits only on train and does not use val statistics."""
    train_df = sample_feature_df.iloc[:70].copy()
    val_df = sample_feature_df.iloc[70:].copy()

    preprocessor = FraudPreprocessor(config=mock_config)
    preprocessor.fit(train_df)

    # Scaler center must be fitted on train
    train_log_amount = np.log1p(train_df[["Amount"]])
    expected_median = float(np.median(train_log_amount))
    assert abs(preprocessor.scaler_.center_[0] - expected_median) < 0.1

    # Transforming val must work without refitting
    val_trans = preprocessor.transform(val_df)
    assert len(val_trans) == len(val_df)


def test_cyclical_encoding_range(sample_feature_df, mock_config):
    """Verify cyclical encoding values remain strictly in [-1, 1]."""
    preprocessor = FraudPreprocessor(config=mock_config)
    X_trans = preprocessor.fit_transform(sample_feature_df)

    assert (X_trans["hour_sin"] >= -1.0).all() and (X_trans["hour_sin"] <= 1.0).all()
    assert (X_trans["hour_cos"] >= -1.0).all() and (X_trans["hour_cos"] <= 1.0).all()


def test_smote_increases_minority(mock_config):
    """Verify SMOTE oversamples the minority class in training data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 5)})
    # 190 legit, 10 fraud
    y = pd.Series([0] * 190 + [1] * 10)

    X_res, y_res = apply_smote(X, y, mock_config, is_training_data=True)

    assert len(y_res) > len(y)
    assert (y_res == 1).sum() == (y_res == 0).sum()  # Balanced by default


def test_smote_rejects_non_training(mock_config):
    """Verify SMOTE raises an error if flagged as non-training data (leakage prevention)."""
    n = 50
    X = pd.DataFrame({"V1": np.random.randn(n)})
    y = pd.Series([0] * 45 + [1] * 5)

    with pytest.raises(ValueError, match="must only be called with training data"):
        apply_smote(X, y, mock_config, is_training_data=False)
