"""
FraudShield Feature Preprocessor
=================================
Handles all feature engineering for the credit-card fraud detection pipeline:
  - log1p + RobustScaler on Amount (fitted on training data only)
  - Cyclical time-of-day encoding (hour_sin / hour_cos)
  - V1–V28 PCA features passed through unchanged
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config relative to project root (cwd) or an absolute path."""
    path = Path(config_path)
    if not path.is_absolute():
        # Walk up from this file to find the project root that contains the config
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
            full = candidate / config_path
            if full.exists():
                path = full
                break
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_V_COLS = [f"V{i}" for i in range(1, 29)]
_ENGINEERED_COLS = ["hour_sin", "hour_cos", "amount_scaled"]
_FEATURE_ORDER = _V_COLS + _ENGINEERED_COLS  # 28 + 3 = 31 features


# ---------------------------------------------------------------------------
# FraudPreprocessor
# ---------------------------------------------------------------------------

class FraudPreprocessor:
    """
    Stateful preprocessing pipeline for FraudShield.

    Workflow
    --------
    1. ``fit(X_train)``      – compute scaler statistics from training data only.
    2. ``transform(X)``      – apply the fitted transformations to any split.
    3. ``fit_transform(X)``  – convenience wrapper (fit + transform on the same data).

    Feature engineering applied
    ---------------------------
    - **Amount**: log1p (if ``log_amount: true`` in config) → RobustScaler.
    - **Time**: modulo 86400 → hour-of-day → sine / cosine cyclical encoding.
    - **V1–V28**: passed through unchanged.
    - Original ``Time`` and ``Amount`` columns are dropped.
    """

    def __init__(self, config: Optional[dict] = None, config_path: str = "config/config.yaml") -> None:
        if config is None:
            config = _load_config(config_path)
        self.config = config
        prep_cfg = config.get("preprocessing", {})

        self.log_amount: bool = prep_cfg.get("log_amount", True)
        self.cyclical_encoding: bool = prep_cfg.get("cyclical_encoding", True)
        self.time_modulo: int = int(prep_cfg.get("time_modulo", 86400))

        self._scaler: RobustScaler = RobustScaler()
        self._fitted: bool = False
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> "FraudPreprocessor":
        """
        Fit the preprocessor on training data **only**.

        Parameters
        ----------
        X_train : pd.DataFrame
            Raw training features (must contain 'Time', 'Amount', V1–V28).

        Returns
        -------
        self
        """
        self._validate_input(X_train)

        amount_values = X_train["Amount"].values.reshape(-1, 1)
        if self.log_amount:
            amount_values = np.log1p(amount_values)

        self._scaler.fit(amount_values)
        self._fitted = True

        # Determine column order on a small dummy transform to be consistent
        self._feature_names = _FEATURE_ORDER.copy()

        logger.info(
            "FraudPreprocessor fitted on %d training samples. "
            "log_amount=%s, cyclical_encoding=%s",
            len(X_train),
            self.log_amount,
            self.cyclical_encoding,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features (Time, Amount, V1–V28 at minimum).

        Returns
        -------
        pd.DataFrame
            Engineered features with consistent column ordering.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "FraudPreprocessor must be fitted before calling transform(). "
                "Call fit(X_train) first."
            )
        self._validate_input(X)

        result = X.copy()

        # --- Amount feature ---
        amount_values = result["Amount"].values.reshape(-1, 1)
        if self.log_amount:
            amount_values = np.log1p(amount_values)
        scaled_amount = self._scaler.transform(amount_values).flatten()
        result["amount_scaled"] = scaled_amount

        # --- Time-of-day features ---
        time_seconds = result["Time"].values % self.time_modulo  # 0–86399 s within a day
        hour_of_day = time_seconds / 3600.0  # 0.0–23.999...

        if self.cyclical_encoding:
            result["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
            result["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)
        else:
            result["hour_sin"] = hour_of_day  # fallback: raw hour
            result["hour_cos"] = hour_of_day

        # --- Drop originals ---
        result = result.drop(columns=["Time", "Amount"])

        # --- Enforce consistent column ordering ---
        present_v_cols = [c for c in _V_COLS if c in result.columns]
        engineered = [c for c in _ENGINEERED_COLS if c in result.columns]
        ordered_cols = present_v_cols + engineered

        # Include any unexpected extra columns at the end (graceful handling)
        extra = [c for c in result.columns if c not in ordered_cols]
        if extra:
            logger.warning("Unexpected columns carried through: %s", extra)
            ordered_cols = ordered_cols + extra

        result = result[ordered_cols]

        logger.debug("transform() produced shape %s", result.shape)
        return result

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on X_train then immediately transform it.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data only.

        Returns
        -------
        pd.DataFrame
            Transformed training features.
        """
        return self.fit(X_train).transform(X_train)

    @property
    def scaler_(self) -> RobustScaler:
        """Expose the fitted internal RobustScaler."""
        return self._scaler

    def get_feature_names(self) -> List[str]:
        """
        Return the ordered list of output feature column names.

        Returns
        -------
        List[str]
            Column names in the same order as returned by ``transform()``.

        Raises
        ------
        RuntimeError
            If called before the preprocessor has been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "FraudPreprocessor must be fitted before calling get_feature_names()."
            )
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the fitted preprocessor to disk using joblib.

        Parameters
        ----------
        path : str
            File path (e.g. ``'models/preprocessor.joblib'``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("FraudPreprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FraudPreprocessor":
        """
        Load a previously saved preprocessor from disk.

        Parameters
        ----------
        path : str
            File path to a joblib-serialised ``FraudPreprocessor``.

        Returns
        -------
        FraudPreprocessor
        """
        instance: FraudPreprocessor = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Loaded object is {type(instance)}, expected FraudPreprocessor."
            )
        logger.info("FraudPreprocessor loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Check that mandatory raw columns are present."""
        required = {"Time", "Amount"} | set(_V_COLS)
        missing = required - set(X.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing)}"
            )
        if X.empty:
            raise ValueError("Input DataFrame is empty.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"FraudPreprocessor("
            f"log_amount={self.log_amount}, "
            f"cyclical_encoding={self.cyclical_encoding}, "
            f"time_modulo={self.time_modulo}, "
            f"status={status})"
        )
