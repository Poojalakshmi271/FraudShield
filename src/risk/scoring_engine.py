"""
FraudShield Risk Scoring Engine
=================================
Converts raw fraud probability outputs into actionable risk classifications
with configurable thresholds, human-readable labels, and colour codes.

Disclaimer: DEMONSTRATION RISK CLASSIFICATIONS – SIMULATION ONLY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer constant
# ---------------------------------------------------------------------------
_DISCLAIMER = "DEMONSTRATION RISK CLASSIFICATIONS – SIMULATION ONLY"


# ---------------------------------------------------------------------------
# RiskScore dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskScore:
    """
    Structured risk assessment result for a single transaction.

    Attributes
    ----------
    fraud_probability : float
        Model-estimated probability that the transaction is fraudulent [0, 1].
    risk_category : str
        One of: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    risk_label : str
        Human-readable label (e.g. "High Risk").
    risk_color : str
        Hex colour string for UI rendering (e.g. "#fd7e14").
    recommended_action : str
        Recommended operational response for this risk level.
    top_features : list of dict
        Up to 10 feature contributions: {feature, value, contribution}.
    disclaimer : str
        Mandatory disclaimer string.
    """

    fraud_probability: float
    risk_category: str
    risk_label: str
    risk_color: str
    recommended_action: str
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = _DISCLAIMER


# ---------------------------------------------------------------------------
# RiskScoringEngine
# ---------------------------------------------------------------------------

class RiskScoringEngine:
    """
    Converts model fraud-probability outputs into structured risk assessments.

    Parameters
    ----------
    model : sklearn estimator (or calibrated wrapper)
        Trained model with a ``predict_proba`` method.
    preprocessor : FraudPreprocessor (or any transformer with transform())
        Fitted preprocessing pipeline.
    config : dict
        Parsed config.yaml content.  Risk thresholds are read from
        ``config['risk']['thresholds']``.
    threshold : float
        Probability cut-off for binary fraud/non-fraud classification (default 0.5).
        Does NOT affect risk category boundaries.
    """

    # Default risk category definitions used when config is incomplete
    _DEFAULT_CATEGORIES = {
        "LOW":      {"label": "Low Risk",      "color": "#28a745", "action": "No immediate action required. Transaction may proceed normally."},
        "MEDIUM":   {"label": "Medium Risk",    "color": "#ffc107", "action": "Flag for review. Consider requesting additional verification."},
        "HIGH":     {"label": "High Risk",      "color": "#fd7e14", "action": "Hold transaction. Initiate manual review immediately."},
        "CRITICAL": {"label": "Critical Risk",  "color": "#dc3545", "action": "Block transaction. Escalate to fraud investigation team."},
    }

    def __init__(
        self,
        model,
        preprocessor,
        config: dict,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.threshold = threshold

        # Parse risk thresholds from config
        risk_cfg = config.get("risk", {})
        thresholds_cfg = risk_cfg.get("thresholds", {})
        self._low_threshold: float = float(thresholds_cfg.get("low", 0.20))
        self._medium_threshold: float = float(thresholds_cfg.get("medium", 0.50))
        self._high_threshold: float = float(thresholds_cfg.get("high", 0.80))

        # Build category lookup from config categories list (or fall back to defaults)
        self._category_meta: Dict[str, Dict[str, str]] = {}
        categories_list = risk_cfg.get("categories", [])
        if categories_list:
            for cat in categories_list:
                name = cat.get("name", "").upper()
                self._category_meta[name] = {
                    "label":  cat.get("label", name),
                    "color":  cat.get("color", "#6c757d"),
                    "action": cat.get("action", "Review transaction."),
                }
        else:
            self._category_meta = dict(self._DEFAULT_CATEGORIES)

        logger.info(
            "RiskScoringEngine initialised. Thresholds: low=%.2f, medium=%.2f, high=%.2f.",
            self._low_threshold,
            self._medium_threshold,
            self._high_threshold,
        )

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _classify_risk(self, probability: float) -> Tuple[str, str, str, str]:
        """
        Map a fraud probability to a risk (category, label, color, action).

        Boundaries (defaults from config):
          [0,     low)     → LOW
          [low,   medium)  → MEDIUM
          [medium, high)   → HIGH
          [high,  1.0]     → CRITICAL
        """
        if probability < self._low_threshold:
            cat = "LOW"
        elif probability < self._medium_threshold:
            cat = "MEDIUM"
        elif probability < self._high_threshold:
            cat = "HIGH"
        else:
            cat = "CRITICAL"

        meta = self._category_meta.get(cat, self._DEFAULT_CATEGORIES[cat])
        return cat, meta["label"], meta["color"], meta["action"]

    def update_thresholds(self, low: float, medium: float, high: float) -> None:
        """
        Dynamically update risk category boundaries (e.g. from a dashboard slider).

        Parameters
        ----------
        low    : float  Upper boundary for LOW risk.
        medium : float  Upper boundary for MEDIUM risk.
        high   : float  Upper boundary for HIGH risk.
        """
        if not (0.0 < low < medium < high <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < low < medium < high ≤ 1. "
                f"Got low={low}, medium={medium}, high={high}."
            )
        self._low_threshold = low
        self._medium_threshold = medium
        self._high_threshold = high
        logger.info("Risk thresholds updated: low=%.2f, medium=%.2f, high=%.2f.", low, medium, high)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_transaction(self, transaction_dict: Dict[str, Any]) -> RiskScore:
        """
        Score a single transaction.

        Parameters
        ----------
        transaction_dict : dict
            Raw feature key-value pairs (pre-preprocessing).

        Returns
        -------
        RiskScore
        """
        # Build a single-row DataFrame
        df = pd.DataFrame([transaction_dict])

        # Preprocess
        try:
            X = self.preprocessor.transform(df)
        except Exception as exc:
            logger.warning("Preprocessor failed; attempting to use raw features. Error: %s", exc)
            X = df

        # Predict
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            prob = float(self.model.predict(X)[0])

        prob = float(np.clip(prob, 0.0, 1.0))
        category, label, color, action = self._classify_risk(prob)

        # Top feature contributions (best-effort using coef_ or feature_importances_)
        top_features = self._get_top_features(X, transaction_dict)

        logger.debug("Transaction scored: prob=%.4f, category=%s.", prob, category)
        return RiskScore(
            fraud_probability=prob,
            risk_category=category,
            risk_label=label,
            risk_color=color,
            recommended_action=action,
            top_features=top_features,
            disclaimer=_DISCLAIMER,
        )

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Rows of raw (pre-preprocessing) transaction features.

        Returns
        -------
        pd.DataFrame
            Original columns plus: fraud_probability, risk_category, risk_label.
        """
        result_df = df.copy()

        # Preprocess
        try:
            X = self.preprocessor.transform(df)
        except Exception as exc:
            logger.warning("Preprocessor failed; using raw features. Error: %s", exc)
            X = df

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict(X).astype(float)

        probs = np.clip(probs, 0.0, 1.0)
        categories, labels, _, _ = zip(*[self._classify_risk(p) for p in probs])

        result_df["fraud_probability"] = probs
        result_df["risk_category"] = list(categories)
        result_df["risk_label"] = list(labels)

        logger.info("Scored %d transactions.", len(df))
        return result_df

    # ------------------------------------------------------------------
    # Aggregation / Distribution
    # ------------------------------------------------------------------

    def get_risk_distribution(self, probabilities: Union[List[float], np.ndarray]) -> Dict[str, int]:
        """
        Count transactions per risk category.

        Parameters
        ----------
        probabilities : array-like of float
            Fraud probability for each transaction.

        Returns
        -------
        dict  {category_name: count}  e.g. {'LOW': 980, 'MEDIUM': 15, ...}
        """
        distribution: Dict[str, int] = {cat: 0 for cat in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}
        for p in probabilities:
            category, _, _, _ = self._classify_risk(float(p))
            distribution[category] += 1
        return distribution

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_top_features(
        self,
        X_transformed: Union[pd.DataFrame, np.ndarray],
        raw_dict: Dict[str, Any],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Extract approximate top feature contributions.

        Uses model.coef_ for linear models, model.feature_importances_ for
        tree models, or falls back to raw feature values when neither is
        available.
        """
        if isinstance(X_transformed, pd.DataFrame):
            feature_names = list(X_transformed.columns)
            x_arr = X_transformed.values.flatten()
        else:
            x_arr = np.asarray(X_transformed).flatten()
            feature_names = [f"feature_{i}" for i in range(len(x_arr))]

        # Try to get model importances / coefficients
        contributions: Optional[np.ndarray] = None
        inner_model = self.model
        # Unwrap CalibratedClassifierCV if needed
        if hasattr(inner_model, "calibrated_classifiers_"):
            try:
                inner_model = inner_model.calibrated_classifiers_[0].estimator
            except (AttributeError, IndexError):
                pass

        if hasattr(inner_model, "coef_"):
            coef = np.asarray(inner_model.coef_).flatten()
            if len(coef) == len(x_arr):
                contributions = np.abs(coef * x_arr)
        elif hasattr(inner_model, "feature_importances_"):
            fi = np.asarray(inner_model.feature_importances_).flatten()
            if len(fi) == len(x_arr):
                contributions = fi

        if contributions is None:
            # Fallback: sort by absolute raw value
            contributions = np.abs(x_arr)

        # Pick top_n
        order = np.argsort(contributions)[::-1][:top_n]
        result: List[Dict[str, Any]] = []
        for idx in order:
            feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            raw_val = raw_dict.get(feat, float(x_arr[idx]) if idx < len(x_arr) else None)
            result.append(
                {
                    "feature": feat,
                    "value": raw_val,
                    "contribution": float(contributions[idx]),
                }
            )
        return result


# ---------------------------------------------------------------------------
# Standalone convenience function
# ---------------------------------------------------------------------------

def score_single_transaction(
    features_dict: Dict[str, Any],
    model,
    preprocessor,
    config: dict,
    threshold: float = 0.5,
) -> RiskScore:
    """
    Convenience wrapper – score one transaction without instantiating the engine.

    Parameters
    ----------
    features_dict : dict
        Raw transaction feature key-value pairs.
    model : sklearn estimator
        Trained classification model.
    preprocessor : transformer
        Fitted preprocessing pipeline.
    config : dict
        Parsed YAML config (must contain 'risk' section).
    threshold : float
        Binary classification probability threshold.

    Returns
    -------
    RiskScore
    """
    engine = RiskScoringEngine(
        model=model,
        preprocessor=preprocessor,
        config=config,
        threshold=threshold,
    )
    return engine.score_transaction(features_dict)
