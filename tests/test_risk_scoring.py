"""
FraudShield – Risk Scoring Engine Unit Tests
============================================
Tests risk categorization tiers (LOW, MEDIUM, HIGH, CRITICAL), dynamic threshold
updates, mandatory simulation disclaimers, and DataFrame batch scoring.
"""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.risk.scoring_engine import RiskScore, RiskScoringEngine


@pytest.fixture
def mock_config():
    return {
        "risk": {
            "thresholds": {
                "low": 0.20,
                "medium": 0.50,
                "high": 0.80,
            },
            "categories": [
                {
                    "name": "LOW",
                    "label": "Low Risk",
                    "color": "#28a745",
                    "action": "No immediate action required.",
                },
                {
                    "name": "MEDIUM",
                    "label": "Medium Risk",
                    "color": "#ffc107",
                    "action": "Flag for review.",
                },
                {
                    "name": "HIGH",
                    "label": "High Risk",
                    "color": "#fd7e14",
                    "action": "Hold transaction.",
                },
                {
                    "name": "CRITICAL",
                    "label": "Critical Risk",
                    "color": "#dc3545",
                    "action": "Block transaction immediately.",
                },
            ],
            "disclaimer": "DEMONSTRATION RISK CLASSIFICATIONS – SIMULATION ONLY",
        }
    }


@pytest.fixture
def dummy_engine(mock_config):
    """Create a RiskScoringEngine with mocked model and preprocessor."""
    model = MagicMock()
    preprocessor = MagicMock()
    return RiskScoringEngine(model=model, preprocessor=preprocessor, config=mock_config)


def test_risk_low(dummy_engine):
    """Verify probability < 0.20 maps to LOW risk."""
    cat, label, color, action = dummy_engine._classify_risk(0.10)
    assert cat == "LOW"
    assert "Low" in label
    assert color == "#28a745"


def test_risk_medium(dummy_engine):
    """Verify 0.20 <= probability < 0.50 maps to MEDIUM risk."""
    cat, label, color, action = dummy_engine._classify_risk(0.35)
    assert cat == "MEDIUM"
    assert "Medium" in label


def test_risk_high(dummy_engine):
    """Verify 0.50 <= probability < 0.80 maps to HIGH risk."""
    cat, label, color, action = dummy_engine._classify_risk(0.65)
    assert cat == "HIGH"
    assert "High" in label


def test_risk_critical(dummy_engine):
    """Verify probability >= 0.80 maps to CRITICAL risk."""
    cat, label, color, action = dummy_engine._classify_risk(0.92)
    assert cat == "CRITICAL"
    assert "Critical" in label


def test_disclaimer_present(dummy_engine):
    """Verify mandatory simulation disclaimer is present in risk score."""
    score = RiskScore(
        fraud_probability=0.85,
        risk_category="CRITICAL",
        risk_label="Critical Risk",
        risk_color="#dc3545",
        recommended_action="Block transaction",
    )
    assert "SIMULATION ONLY" in score.disclaimer


def test_threshold_update(dummy_engine):
    """Verify dynamic threshold modification updates subsequent risk classifications."""
    dummy_engine.update_thresholds(low=0.10, medium=0.30, high=0.70)
    # 0.15 is now MEDIUM under new threshold (low=0.10)
    cat, _, _, _ = dummy_engine._classify_risk(0.15)
    assert cat == "MEDIUM"

    # 0.75 is now CRITICAL under new threshold (high=0.70)
    cat_crit, _, _, _ = dummy_engine._classify_risk(0.75)
    assert cat_crit == "CRITICAL"
