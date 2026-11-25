"""
FraudShield – Data Splitting & Leakage Prevention Unit Tests
============================================================
Verifies stratified splitting proportions, preservation of fraud ratios,
and strictly disjoint split indices.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitter import get_split_stats, split_data, verify_no_overlap


@pytest.fixture
def synthetic_dataset():
    """Create a 1,000-row synthetic dataset with a ~5% fraud rate."""
    np.random.seed(42)
    n = 1000
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(100, n)
    # Exactly 50 fraud (5%) and 950 legitimate (95%)
    data["Class"] = [1] * 50 + [0] * 950
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


@pytest.fixture
def test_config():
    return {
        "splitting": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratify": True,
            "random_seed": 42,
        },
        "data": {
            "target_column": "Class",
        },
    }


def test_split_ratios(synthetic_dataset, test_config):
    """Verify split sizes are approximately 70% train, 15% val, 15% test."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(synthetic_dataset, test_config)

    total = len(synthetic_dataset)
    assert abs(len(X_train) / total - 0.70) < 0.02
    assert abs(len(X_val) / total - 0.15) < 0.02
    assert abs(len(X_test) / total - 0.15) < 0.02


def test_stratification(synthetic_dataset, test_config):
    """Verify fraud rate is preserved across train, val, and test splits."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(synthetic_dataset, test_config)

    orig_rate = synthetic_dataset["Class"].mean()
    train_rate = y_train.mean()
    val_rate = y_val.mean()
    test_rate = y_test.mean()

    # Preserved within 2% absolute tolerance
    assert abs(train_rate - orig_rate) < 0.02
    assert abs(val_rate - orig_rate) < 0.02
    assert abs(test_rate - orig_rate) < 0.02


def test_no_index_overlap(synthetic_dataset, test_config):
    """Verify strictly disjoint indices between splits (no leakage)."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(synthetic_dataset, test_config)

    # Must pass without raising AssertionError
    no_overlap = verify_no_overlap(X_train, X_val, X_test)
    assert no_overlap is True


def test_get_split_stats(synthetic_dataset, test_config):
    """Verify split statistics extraction."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(synthetic_dataset, test_config)
    stats = get_split_stats(y_train, y_val, y_test)

    assert "train" in stats
    assert "val" in stats
    assert "test" in stats
    assert stats["train"]["total"] == len(y_train)
    assert stats["val"]["total"] == len(y_val)
    assert stats["test"]["total"] == len(y_test)
