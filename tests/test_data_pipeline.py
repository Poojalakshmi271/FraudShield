"""
FraudShield – Data Pipeline Unit Tests
======================================
Tests dataset validation, missing values, schema checks, and sample extraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataValidationReport, get_sample_transactions, validate_dataset


@pytest.fixture
def valid_synthetic_df():
    """Create a minimal valid synthetic credit card dataset."""
    np.random.seed(42)
    n = 200
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(50, n)
    data["Class"] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    return pd.DataFrame(data)


def test_validate_dataset_valid(valid_synthetic_df):
    """Verify validation passes on a correctly structured dataset."""
    report = validate_dataset(valid_synthetic_df, check_dimensions=False)
    assert isinstance(report, DataValidationReport)
    assert report.validation_passed is True
    assert sum(report.missing_counts.values()) == 0
    assert set(report.class_distribution.keys()) == {0, 1}


def test_validate_dataset_missing_column(valid_synthetic_df):
    """Verify validation fails if a required column (e.g., 'Class') is missing."""
    df_missing = valid_synthetic_df.drop(columns=["Class"])
    report = validate_dataset(df_missing, check_dimensions=False)
    assert report.validation_passed is False
    assert any("Class" in msg for msg in report.messages)


def test_validate_dataset_missing_values(valid_synthetic_df):
    """Verify validation identifies missing/null values."""
    df_null = valid_synthetic_df.copy()
    df_null.loc[0, "Amount"] = np.nan
    report = validate_dataset(df_null, check_dimensions=False)
    assert sum(report.missing_counts.values()) > 0
    assert any("missing values detected" in msg for msg in report.messages)


def test_validate_dataset_invalid_class(valid_synthetic_df):
    """Verify validation fails if classes other than 0 and 1 exist."""
    df_invalid = valid_synthetic_df.copy()
    df_invalid.loc[0, "Class"] = 99
    report = validate_dataset(df_invalid, check_dimensions=False)
    assert report.validation_passed is False
    assert any("Unexpected classes in target column" in msg for msg in report.messages)


def test_get_sample_transactions(valid_synthetic_df):
    """Verify extraction of sample transactions per class."""
    legit_samples = get_sample_transactions(valid_synthetic_df, n=3, fraud=False)
    fraud_samples = get_sample_transactions(valid_synthetic_df, n=3, fraud=True)

    assert len(legit_samples) <= 3
    assert len(fraud_samples) <= 3
    assert (legit_samples["Class"] == 0).all()
    assert (fraud_samples["Class"] == 1).all()
